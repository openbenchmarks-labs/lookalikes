"""Gold-set builder CLI: harvest → (human QA) → freeze → verify.

    python3 scripts/lookalike/goldset/build.py harvest --seed ramp --category b2b-saas
    python3 scripts/lookalike/goldset/build.py freeze  --seed ramp
    python3 scripts/lookalike/goldset/build.py verify

Stages (methodology §5):
  harvest  Emit a raw working file `<seed>.raw.json` pre-filled with the
           category's sanctioned sources. For "open" sources (Wikidata) it can
           auto-populate companies; "manual" sources are filled in by a human
           from the published rosters. NEVER queries a benchmarked vendor.
  freeze   Apply the inclusion rule (>=1 Tier-A OR >=2 Tier-B per company),
           compute the content hash, and write the frozen `<seed>.gold.json`.
  verify   Load every `*.gold.json`, re-verify hashes, and re-check the
           inclusion rule. Use in CI so a hand-edited gold file fails loudly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the `lookalike` package importable whether run as a script or module.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lookalike.recall import GoldCompany, GoldSet, canonical_domain, normalize_name, load_goldset  # noqa: E402
from lookalike.goldset import sources as src  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = ROOT / "data" / "lookalike-tam" / "gold"


def _raw_path(seed: str) -> Path:
    return GOLD_DIR / f"{seed}.raw.json"


def _manual_path(seed: str) -> Path:
    return GOLD_DIR / f"{seed}.manual.json"


def _gold_path(seed: str) -> Path:
    return GOLD_DIR / f"{seed}.gold.json"


def _company_key(c: dict) -> str:
    """Identity for merging across layers: first canonical domain, else name."""
    for d in c.get("domains") or ([c["domain"]] if c.get("domain") else []):
        cd = canonical_domain(d)
        if cd:
            return f"d:{cd}"
    return f"n:{normalize_name(c.get('name', ''))}"


def _merge_layers(*layers: list[dict]) -> list[dict]:
    """Union companies across layers by identity, accumulating sources/domains."""
    merged: dict[str, dict] = {}
    for companies in layers:
        for c in companies:
            key = _company_key(c)
            cur = merged.get(key)
            if cur is None:
                merged[key] = {
                    "name": c.get("name", "?"),
                    "domains": list(dict.fromkeys(c.get("domains") or ([c["domain"]] if c.get("domain") else []))),
                    "sources": list(dict.fromkeys(c.get("sources") or [])),
                    "attrs": dict(c.get("attrs") or {}),
                }
            else:
                for d in c.get("domains") or []:
                    if d not in cur["domains"]:
                        cur["domains"].append(d)
                for s in c.get("sources") or []:
                    if s not in cur["sources"]:
                        cur["sources"].append(s)
                cur["attrs"].update(c.get("attrs") or {})
    return list(merged.values())


def cmd_harvest(args: argparse.Namespace) -> int:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    category = args.category
    if category not in src.SOURCES:
        print(f"unknown category '{category}'. known: {', '.join(src.SOURCES)}", file=sys.stderr)
        return 1

    companies: list[dict] = []
    if args.wikidata_qid:
        try:
            from lookalike.goldset import wikidata
            fetched = wikidata.companies_in_industry(args.wikidata_qid)
            for c in fetched:
                companies.append(
                    {"name": c["name"], "domains": c["domains"], "sources": ["wikidata:industry"]}
                )
            print(f"wikidata: fetched {len(companies)} companies for {args.wikidata_qid}")
        except Exception as exc:  # noqa: BLE001
            print(f"wikidata harvest failed ({exc}); emitting empty template", file=sys.stderr)

    raw = {
        "seed_slug": args.seed,
        "seed_name": args.seed,
        "category": category,
        "as_of": args.as_of or "",
        "filters": {},
        "version_hash": "",
        "sources_used": src.category_template_sources(category),
        "_instructions": (
            "Fill `sources_used[].url` + `captured_at`. Add companies from the "
            "WHOLE published roster of each sanctioned source (no subsetting). "
            "Each company needs >=1 Tier-A source OR >=2 Tier-B sources in its "
            "`sources`. Then run: build.py freeze --seed " + args.seed
        ),
        "companies": companies,
    }
    out = _raw_path(args.seed)
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  (edit it, then `freeze`)")
    return 0


def _layer_files(seed: str) -> list[Path]:
    """All input layers for a seed (manual/g2/raw/…), excluding the gold file.
    Ordered so richer-metadata layers win on conflicts: manual, g2, raw, others."""
    order = {"manual": 0, "g2": 1, "raw": 2}
    files = [f for f in GOLD_DIR.glob(f"{seed}.*.json") if not f.name.endswith(".gold.json")]
    return sorted(files, key=lambda f: order.get(f.name.split(".")[-2], 9))


def cmd_freeze(args: argparse.Namespace) -> int:
    layers = _layer_files(args.seed)
    if not layers:
        print(f"no input layers ({args.seed}.*.json) in {GOLD_DIR}; run `harvest` first", file=sys.stderr)
        return 1

    # Merge every layer (manual gated-source paste, G2 API, open-source auto, …).
    # A company's sources accumulate across layers, so cross-layer agreement
    # strengthens corroboration.
    loaded = [(f, json.loads(f.read_text(encoding="utf-8"))) for f in layers]
    data = loaded[0][1]  # highest-priority layer for top-level metadata
    category = next((d.get("category") for _, d in loaded if d.get("category")), "")

    companies = _merge_layers(*[d.get("companies", []) for _, d in loaded])
    sources_used = [s for _, d in loaded for s in (d.get("sources_used") or [])]
    breakdown = ", ".join(f"{f.name.split('.')[-2]}={len(d.get('companies', []))}" for f, d in loaded)
    print(f"merged {len(loaded)} layer(s) → {len(companies)} companies ({breakdown})")

    kept: list[dict] = []
    rejected: list[tuple[str, str]] = []
    for c in companies:
        ok, reason = src.qualifies(category, list(c.get("sources") or []))
        if ok:
            kept.append(c)
        else:
            rejected.append((c.get("name", "?"), reason))

    if rejected:
        print(f"inclusion rule rejected {len(rejected)} compan(ies):")
        for name, reason in rejected:
            print(f"  ✗ {name}: {reason}")
        if not args.allow_rejects:
            print("\nfix sources or pass --allow-rejects to drop them and continue.", file=sys.stderr)
            return 1

    gs = GoldSet(
        seed_slug=data.get("seed_slug", args.seed),
        category=category,
        as_of=data.get("as_of", ""),
        companies=[
            GoldCompany(
                name=c["name"],
                domains=tuple(c.get("domains") or ([c["domain"]] if c.get("domain") else ())),
                sources=tuple(c.get("sources") or ()),
                attrs=c.get("attrs") or {},
            )
            for c in kept
        ],
        filters=data.get("filters") or {},
    )
    seed_domain = next((d.get("seed_domain") for _, d in loaded if d.get("seed_domain")), None)
    description = next((d.get("description") for _, d in loaded if d.get("description")), None)
    payload = {
        "seed_slug": gs.seed_slug,
        "seed_name": data.get("seed_name", gs.seed_slug),
        "seed_domain": seed_domain,
        "description": description,
        "category": gs.category,
        "as_of": gs.as_of,
        "filters": gs.filters,
        "version_hash": gs.compute_hash(),
        "sources_used": sources_used,
        "companies": [
            {"name": c.name, "domains": list(c.domains), "sources": list(c.sources), "attrs": c.attrs}
            for c in gs.companies
        ],
    }
    out = _gold_path(args.seed)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"froze {out}  ({gs.size} companies, hash={gs.compute_hash()})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    files = sorted(GOLD_DIR.glob("*.gold.json"))
    if not files:
        print(f"no gold files in {GOLD_DIR}", file=sys.stderr)
        return 1
    bad = 0
    for f in files:
        try:
            gs = load_goldset(f)  # raises on hash mismatch
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {f.name}: {exc}")
            bad += 1
            continue
        # re-check inclusion rule
        offenders = []
        for c in gs.companies:
            ok, reason = src.qualifies(gs.category, list(c.sources))
            if not ok:
                offenders.append((c.name, reason))
        if offenders:
            bad += 1
            print(f"  ✗ {f.name}: {len(offenders)} compan(ies) fail inclusion rule")
            for name, reason in offenders[:5]:
                print(f"      - {name}: {reason}")
        else:
            print(f"  ✓ {f.name}: {gs.size} companies, hash ok, inclusion ok ({gs.category})")
    if bad:
        print(f"\n{bad} file(s) failed verification", file=sys.stderr)
        return 1
    print(f"\nall {len(files)} gold file(s) verified ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="emit a raw working file for a seed")
    h.add_argument("--seed", required=True)
    h.add_argument("--category", required=True, choices=list(src.SOURCES))
    h.add_argument("--as-of", default="")
    h.add_argument("--wikidata-qid", default=None, help="optional industry QID to auto-populate (open source)")
    h.add_argument("--force", action="store_true")
    h.set_defaults(func=cmd_harvest)

    f = sub.add_parser("freeze", help="apply inclusion rule + hash; write gold file")
    f.add_argument("--seed", required=True)
    f.add_argument("--allow-rejects", action="store_true", help="drop non-qualifying companies and continue")
    f.set_defaults(func=cmd_freeze)

    v = sub.add_parser("verify", help="re-verify all gold files (hashes + inclusion rule)")
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
