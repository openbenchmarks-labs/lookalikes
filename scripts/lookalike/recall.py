"""TAM-recall computation core for the lookalike benchmark.

This module is intentionally **pure and dependency-free** (stdlib only) so the
recall math can be unit-tested in isolation and re-derived by any third party.
It does NOT make network calls and does NOT import any vendor client — the
reference (gold) set must be vendor-independent (see RECALL_METHODOLOGY.md §5.1).

What lives here:
  • domain canonicalization + name normalization (the matching key)
  • GoldCompany / GoldSet dataclasses + JSON loading + content hashing
  • match_candidates(): ranked vendor results → matched gold hits (deterministic)
  • metrics: recall@k, R-precision, hit@k, precision@k, coverage
  • fairness audit: findability distribution, audit-vendor-only, unique contribution

Run the self-test (no API keys, no network):

    python3 scripts/lookalike/recall.py --selftest

Everything returns percentages rounded to 2 dp to match the existing
`precision_at_k` convention in common.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Domain canonicalization                                                     #
# --------------------------------------------------------------------------- #

# Common multi-part public suffixes. Not exhaustive (a full public-suffix list
# would be a dependency); covers the suffixes that actually show up in B2B
# vendor output. Extend as needed — additions only ever make matching more
# correct, never less deterministic.
_MULTI_PART_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk",
        "com.au", "net.au", "org.au", "co.nz", "co.jp", "co.kr",
        "com.br", "com.mx", "com.ar", "com.sg", "com.hk", "com.tr",
        "co.in", "co.za", "co.il", "com.cn", "com.tw",
    }
)

_NAME_SUFFIX_RE = re.compile(
    r"\b(inc|inc\.|llc|l\.l\.c\.|ltd|ltd\.|limited|corp|corp\.|"
    r"corporation|co|co\.|gmbh|s\.a\.|sa|plc|ag|nv|bv|pty|holdings|group)\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def canonical_domain(raw: str | None) -> str | None:
    """Normalize a URL or hostname to its registrable domain (eTLD+1).

    Examples:
        https://www.Brex.com/pricing  -> brex.com
        api.openfunnel.dev        -> openfunnel.dev
        foo.co.uk                     -> foo.co.uk
        ""                            -> None
    """
    if not raw or not isinstance(raw, str):
        return None
    host = raw.strip().lower()
    if not host:
        return None
    if "://" in host:
        host = host.split("://", 1)[1]
    # strip userinfo, path, query, fragment, port
    host = host.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split(":", 1)[0]
    host = host.split("@")[-1]
    if host.startswith("www."):
        host = host[4:]
    host = host.strip(".")
    if not host or "." not in host:
        return host or None
    parts = host.split(".")
    last_two = ".".join(parts[-2:])
    if last_two in _MULTI_PART_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


def normalize_name(raw: str | None) -> str:
    """Lowercase, strip legal suffixes + punctuation. Used only as a fallback
    match key when a vendor returns no domain (see methodology §6.3)."""
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = _NAME_SUFFIX_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub("", s)
    return s


# --------------------------------------------------------------------------- #
# Gold-set dataclasses                                                         #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class GoldCompany:
    """One known-similar company in a seed's reference set. `domains` are the
    canonical aliases; a vendor result matching ANY counts as a hit."""

    name: str
    domains: tuple[str, ...]
    sources: tuple[str, ...] = ()
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def name_key(self) -> str:
        return normalize_name(self.name)


@dataclasses.dataclass
class GoldSet:
    """A frozen, vendor-independent reference set for one seed.

    `as_of` + `version_hash` pin the exact contents for reproducibility. The
    domain/name indexes are built once for O(1) matching."""

    seed_slug: str
    category: str
    as_of: str
    companies: list[GoldCompany]
    filters: dict[str, Any] = dataclasses.field(default_factory=dict)
    version_hash: str = ""

    def __post_init__(self) -> None:
        self._by_domain: dict[str, GoldCompany] = {}
        self._by_name: dict[str, GoldCompany] = {}
        for c in self.companies:
            for d in c.domains:
                cd = canonical_domain(d)
                if cd:
                    self._by_domain.setdefault(cd, c)
            if c.name_key:
                self._by_name.setdefault(c.name_key, c)
        if not self.version_hash:
            self.version_hash = self.compute_hash()

    @property
    def size(self) -> int:
        return len(self.companies)

    def compute_hash(self) -> str:
        """Deterministic content hash over (name, sorted domains). The freeze
        proof — recorded in the snapshot so a result is tied to an exact set."""
        payload = sorted(
            [
                [c.name.strip().lower(), sorted(canonical_domain(d) or d for d in c.domains)]
                for c in self.companies
            ]
        )
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def lookup_domain(self, domain: str | None) -> GoldCompany | None:
        cd = canonical_domain(domain)
        return self._by_domain.get(cd) if cd else None

    def lookup_name(self, name: str | None) -> GoldCompany | None:
        return self._by_name.get(normalize_name(name)) if name else None


# --------------------------------------------------------------------------- #
# Candidate adapter — works with common.Candidate or any name/domain/rank obj  #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Cand:
    name: str
    domain: str | None
    rank: int


def _as_candidates(items: Sequence[Any]) -> list[_Cand]:
    """Accept common.Candidate, dicts, or _Cand. Rank is taken from the object
    if present, else the (1-based) position in the list."""
    out: list[_Cand] = []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            name, domain, rank = it.get("name") or "", it.get("domain"), it.get("rank")
        else:
            name = getattr(it, "name", "") or ""
            domain = getattr(it, "domain", None)
            rank = getattr(it, "rank", None)
        out.append(_Cand(name=name, domain=domain, rank=int(rank) if rank else i + 1))
    return out


# --------------------------------------------------------------------------- #
# Matching                                                                     #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class MatchHit:
    rank: int
    gold_name: str
    method: str  # "domain" | "alias" | "name"
    candidate_domain: str | None


def match_candidates(candidates: Sequence[Any], gold: GoldSet) -> list[MatchHit]:
    """Map ranked vendor candidates onto gold companies, deterministically.

    Rules (methodology §6):
      • primary key = canonical registrable domain
      • a gold company is credited at most once, at the earliest rank it appears
      • name fallback only when a candidate has no domain (flagged method="name")
    Returns one MatchHit per *distinct* gold company found, ordered by rank.
    """
    cands = _as_candidates(candidates)
    credited: dict[str, MatchHit] = {}  # gold name_key -> first hit
    for c in cands:
        gold_co: GoldCompany | None = None
        method = ""
        cd = canonical_domain(c.domain)
        if cd:
            gold_co = gold.lookup_domain(cd)
            if gold_co is not None:
                first_domain = canonical_domain(gold_co.domains[0]) if gold_co.domains else None
                method = "domain" if cd == first_domain else "alias"
        else:
            gold_co = gold.lookup_name(c.name)
            if gold_co is not None:
                method = "name"
        if gold_co is None:
            continue
        key = gold_co.name_key or gold_co.name
        if key in credited:
            continue  # already credited at an earlier (better) rank
        credited[key] = MatchHit(
            rank=c.rank, gold_name=gold_co.name, method=method, candidate_domain=cd
        )
    return sorted(credited.values(), key=lambda h: h.rank)


# --------------------------------------------------------------------------- #
# Metrics — all return percent (0-100) rounded to 2dp, or None when undefined  #
# --------------------------------------------------------------------------- #


def _pct(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(100.0 * numer / denom, 2)


def recall_at_k(hits: Sequence[MatchHit], gold_size: int, k: int) -> float | None:
    """Fraction of the gold set found within the top K results."""
    found = sum(1 for h in hits if h.rank <= k)
    return _pct(found, gold_size)


def r_precision(hits: Sequence[MatchHit], gold_size: int) -> float | None:
    """Precision/recall crossover: relevant found within top-R, over R."""
    if gold_size <= 0:
        return None
    found = sum(1 for h in hits if h.rank <= gold_size)
    return _pct(found, gold_size)


def hit_at_k(hits: Sequence[MatchHit], k: int) -> int:
    """1 if at least one gold company appears in the top K, else 0."""
    return 1 if any(h.rank <= k for h in hits) else 0


def precision_at_k(relevant_flags: Sequence[bool], k: int) -> float | None:
    """Top-of-list cleanliness from judge labels (kept for parity with the
    existing precision benchmark). Independent of the gold set."""
    top = list(relevant_flags)[:k]
    if not top:
        return None
    return _pct(sum(1 for r in top if r), len(top))


def recall_curve(hits: Sequence[MatchHit], gold_size: int, ks: Iterable[int]) -> dict[int, float | None]:
    """Recall@K for several K from a single matched-hit list (the efficient
    'fetch once, score many' path)."""
    return {k: recall_at_k(hits, gold_size, k) for k in ks}


# --------------------------------------------------------------------------- #
# Fairness audit (methodology §7)                                              #
# --------------------------------------------------------------------------- #


def fairness_audit(
    per_vendor_hits: dict[str, list[MatchHit]],
    gold: GoldSet,
    *,
    audit_vendor: str = "openfunnel",
) -> dict[str, Any]:
    """Evidence that the reference set is not biased toward the audit vendor.

    `per_vendor_hits` maps vendor_slug -> hits for ONE seed. Returns:
      • findability: gold companies found by >=2 vendors (count + pct)
      • audit_vendor_only: gold companies found ONLY by the audit vendor
      • unique_contribution: per-vendor gold companies no other vendor found
    """
    finders: dict[str, set[str]] = {}  # gold name_key -> set(vendor)
    for vendor, hits in per_vendor_hits.items():
        for h in hits:
            finders.setdefault(normalize_name(h.gold_name), set()).add(vendor)

    found_keys = list(finders.keys())
    found_by_2plus = sum(1 for vs in finders.values() if len(vs) >= 2)
    audit_vendor_only = sorted(
        k for k, vs in finders.items() if vs == {audit_vendor}
    )
    unique: dict[str, int] = {}
    for vendor in per_vendor_hits:
        unique[vendor] = sum(
            1 for vs in finders.values() if vs == {vendor}
        )
    return {
        "gold_size": gold.size,
        "gold_found_by_any": len(found_keys),
        "gold_found_by_2plus": found_by_2plus,
        "found_by_2plus_pct": _pct(found_by_2plus, len(found_keys)) if found_keys else None,
        "audit_vendor": audit_vendor,
        "audit_vendor_only_count": len(audit_vendor_only),
        "audit_vendor_only_companies": audit_vendor_only,
        "unique_contribution": unique,
    }


# --------------------------------------------------------------------------- #
# Gold-set JSON loading                                                        #
# --------------------------------------------------------------------------- #


def load_goldset(path: Path | str) -> GoldSet:
    """Load a frozen gold set from its JSON file (schema in
    data/lookalike-tam/gold/README.md). Verifies the stored hash if present."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    companies = [
        GoldCompany(
            name=c["name"],
            domains=tuple(c.get("domains") or ([c["domain"]] if c.get("domain") else ())),
            sources=tuple(c.get("sources") or ()),
            attrs=c.get("attrs") or {},
        )
        for c in data.get("companies", [])
    ]
    gs = GoldSet(
        seed_slug=data["seed_slug"],
        category=data.get("category", ""),
        as_of=data.get("as_of", ""),
        companies=companies,
        filters=data.get("filters") or {},
        version_hash=data.get("version_hash", ""),
    )
    stored = data.get("version_hash")
    if stored and stored != gs.compute_hash():
        raise ValueError(
            f"gold-set hash mismatch for {p.name}: stored={stored} "
            f"computed={gs.compute_hash()} — the file was edited without "
            "re-freezing. Re-run the freeze step."
        )
    return gs


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    """Synthetic, deterministic checks for the metric math + matching rules."""
    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # --- canonicalization ---
    check("canon scheme+www+path", canonical_domain("https://www.Brex.com/x"), "brex.com")
    check("canon subdomain", canonical_domain("api.openfunnel.dev"), "openfunnel.dev")
    check("canon multipart", canonical_domain("foo.co.uk"), "foo.co.uk")
    check("canon deep multipart", canonical_domain("a.b.foo.co.uk"), "foo.co.uk")
    check("canon empty", canonical_domain(""), None)
    check("name norm", normalize_name("Brex, Inc."), "brex")

    # --- gold set of 5 companies ---
    gold = GoldSet(
        seed_slug="ramp",
        category="b2b-saas",
        as_of="2026-06-08",
        companies=[
            GoldCompany("Brex", ("brex.com", "getbrex.com"), ("g2:spend", "crunchbase:ramp")),
            GoldCompany("Mercury", ("mercury.com",), ("g2:banking", "crunchbase:ramp")),
            GoldCompany("Airbase", ("airbase.com",), ("g2:spend",)),
            GoldCompany("Divvy", ("getdivvy.com",), ("g2:spend",)),
            GoldCompany("Pleo", ("pleo.io",), ("g2:spend",)),
        ],
    )
    check("gold size", gold.size, 5)

    # --- first synthetic vendor: finds Brex(rank1 via alias), Mercury(rank3), Pleo(rank8) ---
    cands_a = [
        {"name": "Brex", "domain": "https://www.getbrex.com", "rank": 1},
        {"name": "Noise Co", "domain": "noise.com", "rank": 2},
        {"name": "Mercury", "domain": "mercury.com", "rank": 3},
        {"name": "Other", "domain": "other.io", "rank": 4},
        {"name": "Pleo", "domain": "pleo.io", "rank": 8},
    ]
    hits_a = match_candidates(cands_a, gold)
    check("A hits count", len(hits_a), 3)
    check("A brex alias method", hits_a[0].method, "alias")
    check("A mercury domain method", hits_a[1].method, "domain")
    check("A recall@1", recall_at_k(hits_a, gold.size, 1), 20.0)   # 1/5
    check("A recall@3", recall_at_k(hits_a, gold.size, 3), 40.0)   # 2/5
    check("A recall@10", recall_at_k(hits_a, gold.size, 10), 60.0) # 3/5
    check("A r-precision", r_precision(hits_a, gold.size), 40.0)   # R=5: ranks<=5 -> Brex,Mercury = 2/5
    check("A hit@3", hit_at_k(hits_a, 3), 1)
    curve_a = recall_curve(hits_a, gold.size, (1, 3, 10))
    check("A curve", curve_a, {1: 20.0, 3: 40.0, 10: 60.0})

    # --- dedup: same gold company twice keeps earliest rank ---
    cands_dup = [
        {"name": "Brex", "domain": "brex.com", "rank": 2},
        {"name": "Brex (dupe)", "domain": "getbrex.com", "rank": 5},
    ]
    hits_dup = match_candidates(cands_dup, gold)
    check("dedup count", len(hits_dup), 1)
    check("dedup keeps earliest rank", hits_dup[0].rank, 2)

    # --- name fallback only when no domain ---
    cands_name = [{"name": "Airbase", "domain": None, "rank": 1}]
    hits_name = match_candidates(cands_name, gold)
    check("name fallback hit", len(hits_name), 1)
    check("name fallback method", hits_name[0].method, "name")

    # --- precision@k from judge flags (independent of gold) ---
    check("precision@4", precision_at_k([True, False, True, True], 4), 75.0)

    # --- fairness audit ---
    # vendor B finds only Brex; openfunnel finds Brex + Divvy(unique)
    hits_of = match_candidates(
        [
            {"name": "Brex", "domain": "brex.com", "rank": 1},
            {"name": "Divvy", "domain": "getdivvy.com", "rank": 2},
        ],
        gold,
    )
    hits_b = match_candidates([{"name": "Brex", "domain": "brex.com", "rank": 1}], gold)
    audit = fairness_audit({"openfunnel": hits_of, "exa": hits_b}, gold)
    check("audit found_by_any", audit["gold_found_by_any"], 2)   # Brex, Divvy
    check("audit found_by_2plus", audit["gold_found_by_2plus"], 1)  # Brex
    check("audit audit_vendor_only", audit["audit_vendor_only_count"], 1)        # Divvy
    check("audit unique openfunnel", audit["unique_contribution"]["openfunnel"], 1)
    check("audit unique exa", audit["unique_contribution"]["exa"], 0)

    # --- hash determinism ---
    g2 = GoldSet("ramp", "b2b-saas", "2026-06-08", list(gold.companies))
    check("hash stable", g2.compute_hash(), gold.compute_hash())

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("recall.py self-test: all checks passed ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run deterministic self-test")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
