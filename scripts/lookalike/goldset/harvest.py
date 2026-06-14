"""Automated, multi-signal gold-set harvester (open sources only).

Builds a `<seed>.raw.json` working file programmatically from license-clean
sources, ready for `build.py freeze`. NEVER queries a benchmarked vendor and
NEVER lets an LLM invent members (the optional gate can only DROP, never add).

Signals per company (methodology §5–6):
  • wikidata  — membership in the seed's market industry roster (P452)
  • wikipedia — the company has its own English Wikipedia article (an
                independent editorial process)
A company with both clears the >=2 Tier-B corroboration bar at freeze time,
which doubles as a notability/reliability filter.

Usage:
  python3 scripts/lookalike/goldset/harvest.py \
      --seed postscript --category ecommerce \
      --industry "marketing automation" --industry "email marketing"

  # optionally derive industries from the seed's own Wikidata entity:
  python3 ... --seed klaviyo --category ecommerce --seed-entity "Klaviyo"

  # optional LLM same-kind gate (drops off-topic industry-mates; needs Azure key):
  python3 ... --gate

Then:
  python3 scripts/lookalike/goldset/build.py freeze --seed postscript
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lookalike import recall as R  # noqa: E402
from lookalike.goldset import sources as src, wikidata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = ROOT / "data" / "lookalike-tam" / "gold"
RAW_CACHE = GOLD_DIR / "_raw"


def _today() -> str:
    return datetime.date.today().isoformat()


def _self_match(name: str, domains: list[str], seed_name: str, seed_domain: str | None) -> bool:
    if R.normalize_name(name) == R.normalize_name(seed_name):
        return True
    sd = R.canonical_domain(seed_domain)
    return bool(sd and any(R.canonical_domain(d) == sd for d in domains))


def _alive(domain: str) -> bool:
    """Best-effort liveness check. Tolerant: network failure != dead."""
    url = f"https://{domain}"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": wikidata.UA})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status < 500
    except Exception:  # noqa: BLE001
        return True  # don't drop on transient/anti-bot errors


def _llm_gate(seed_name: str, seed_desc: str | None, candidates: list[dict]) -> set[str]:
    """Optional same-kind filter. Returns the set of qids to KEEP. Drops
    off-topic industry-mates. The LLM can only remove, never add. No-op (keeps
    all) if the Azure judge client can't be constructed."""
    try:
        from lookalike.judge import Judge
        from lookalike.common import Candidate, Seed
        judge = Judge()
    except Exception as exc:  # noqa: BLE001
        print(f"  gate: skipped (no judge client: {exc})")
        return {c["qid"] for c in candidates}

    seed = Seed(seed_slug="seed", seed_name=seed_name, seed_domain=None, description=seed_desc, category="")
    keep: set[str] = set()
    for c in candidates:
        cand = Candidate(name=c["name"], domain=(c["domains"][0] if c["domains"] else None),
                         description=None)
        try:
            verdict = judge.score_candidate(seed, cand)
            if verdict.relevant:
                keep.add(c["qid"])
        except Exception:  # noqa: BLE001
            keep.add(c["qid"])  # fail-open: don't silently drop on error
    print(f"  gate: kept {len(keep)}/{len(candidates)} after same-kind filter")
    return keep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", required=True)
    p.add_argument("--seed-name", default=None)
    p.add_argument("--seed-domain", default=None)
    p.add_argument("--category", required=True, choices=list(src.SOURCES))
    p.add_argument("--industry", action="append", default=[], help="market/industry name (repeatable)")
    p.add_argument("--seed-entity", default=None, help="resolve the seed's own Wikidata entity to derive industries")
    p.add_argument("--limit", type=int, default=300, help="max companies per industry query")
    p.add_argument("--gate", action="store_true", help="LLM same-kind filter (drops off-topic; needs Azure key)")
    p.add_argument("--check-liveness", action="store_true", help="HEAD each domain, drop 5xx")
    args = p.parse_args()

    seed_name = args.seed_name or args.seed
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_CACHE / args.seed).mkdir(parents=True, exist_ok=True)

    # 1. Resolve industry QIDs
    industry_qids: dict[str, str] = {}  # qid -> label
    if args.seed_entity:
        ent = wikidata.search_entity(args.seed_entity, kind="item")
        if ent and ent["qid"]:
            for q in wikidata.industries_of(ent["qid"]):
                industry_qids[q] = q
            print(f"seed entity {ent['label']} ({ent['qid']}) → industries {list(industry_qids)}")
    for name in args.industry:
        ent = wikidata.search_entity(name, kind="item")
        if ent and ent["qid"]:
            industry_qids[ent["qid"]] = f"{ent['label']} ({name})"
            print(f"industry '{name}' → {ent['label']} {ent['qid']}")
        else:
            print(f"industry '{name}' → no Wikidata entity, skipped", file=sys.stderr)

    if not industry_qids:
        print("no industries resolved — pass --industry or --seed-entity", file=sys.stderr)
        return 1

    # 2. Roster each industry; merge by canonical domain
    merged: dict[str, dict] = {}  # qid -> company record
    raw_payloads: dict[str, list] = {}
    for qid in industry_qids:
        companies = wikidata.companies_in_industry(qid, limit=args.limit)
        raw_payloads[qid] = companies
        print(f"  {qid}: {len(companies)} companies")
        for c in companies:
            rec = merged.setdefault(c["qid"], {
                "name": c["name"], "qid": c["qid"], "domains": [],
                "has_enwiki": False, "enwiki_url": None, "industries": [],
            })
            for d in c["domains"]:
                if d not in rec["domains"]:
                    rec["domains"].append(d)
            rec["has_enwiki"] = rec["has_enwiki"] or c["has_enwiki"]
            rec["enwiki_url"] = rec["enwiki_url"] or c["enwiki_url"]
            if qid not in rec["industries"]:
                rec["industries"].append(qid)

    # cache raw payloads for reproducibility
    (RAW_CACHE / args.seed / "wikidata.json").write_text(
        json.dumps(raw_payloads, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Drop ones with no domain (can't be matched) + self-exclusion
    candidates = [
        c for c in merged.values()
        if c["domains"] and not _self_match(c["name"], c["domains"], seed_name, args.seed_domain)
    ]
    print(f"merged → {len(candidates)} candidates with domains (post self-exclusion)")

    # 4. Optional LLM same-kind gate (drop-only)
    if args.gate:
        keep = _llm_gate(seed_name, None, candidates)
        candidates = [c for c in candidates if c["qid"] in keep]

    # 5. Optional liveness
    if args.check_liveness:
        candidates = [c for c in candidates if _alive(c["domains"][0])]
        print(f"liveness → {len(candidates)} live")

    # 6. Build sources per company + raw file
    companies_out = []
    for c in candidates:
        srcs = ["wikidata"]
        if c["has_enwiki"]:
            srcs.append("wikipedia")
        companies_out.append({
            "name": c["name"],
            "domains": [R.canonical_domain(d) or d for d in c["domains"]],
            "sources": srcs,
            "attrs": {"confidence": len(srcs), "wikidata_qid": c["qid"],
                      "enwiki_url": c["enwiki_url"], "industries": c["industries"]},
        })
    companies_out.sort(key=lambda c: (-c["attrs"]["confidence"], c["name"]))

    raw = {
        "seed_slug": args.seed,
        "seed_name": seed_name,
        "seed_domain": args.seed_domain,
        "category": args.category,
        "as_of": _today(),
        "filters": {},
        "version_hash": "",
        "sources_used": (
            [{"id": f"wikidata:{q}", "tier": "B", "name": lbl,
              "url": f"https://www.wikidata.org/wiki/{q}", "captured_at": _today()}
             for q, lbl in industry_qids.items()]
            + [{"id": "wikipedia", "tier": "B", "name": "English Wikipedia article presence",
                "url": "https://en.wikipedia.org", "captured_at": _today()}]
        ),
        "_harvest": {
            "method": "auto (wikidata industry roster + enwiki-article signal)",
            "industries": industry_qids,
            "gate_applied": bool(args.gate),
            "raw_cache": str((RAW_CACHE / args.seed).relative_to(ROOT)),
        },
        "companies": companies_out,
    }
    out = GOLD_DIR / f"{args.seed}.raw.json"
    out.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    n_two = sum(1 for c in companies_out if c["attrs"]["confidence"] >= 2)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"  {len(companies_out)} candidates · {n_two} with >=2 sources (will survive freeze)")
    print(f"  next: python3 scripts/lookalike/goldset/build.py freeze --seed {args.seed} --allow-rejects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
