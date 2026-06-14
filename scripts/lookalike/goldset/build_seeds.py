"""Build + freeze every TAM-recall gold set from the seed registry.

One-time, cached pipeline (methodology §5):
  for each seed in data/lookalike-tam/seeds.json:
    1. g2_rapidapi  → fetch the exhaustive G2 category roster, resolve every
                      product's company domain (cached to _g2cache/), write
                      <seed>.g2.json   (Tier-A `g2cat:<slug>`)
    2. build freeze → merge all layers, apply inclusion rule, hash, write
                      <seed>.gold.json

Re-runs are cheap: resolved product details are cached on disk, so only new
products cost an API call.

Usage:
  python3 scripts/lookalike/goldset/build_seeds.py            # all seeds
  python3 scripts/lookalike/goldset/build_seeds.py --seeds postscript,recharge
  python3 scripts/lookalike/goldset/build_seeds.py --max-details 50   # quota guard
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
SEEDS = ROOT / "data" / "lookalike-tam" / "seeds.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", help="comma-separated seed slugs (default: all)")
    p.add_argument("--max-details", type=int, default=2000, help="cap on G2 detail calls per seed")
    p.add_argument("--skip-fetch", action="store_true", help="only re-freeze (use existing layers)")
    args = p.parse_args()

    registry = json.loads(SEEDS.read_text(encoding="utf-8"))["seeds"]
    wanted = {s.strip() for s in args.seeds.split(",")} if args.seeds else None
    seeds = [s for s in registry if not wanted or s["seed_slug"] in wanted]
    if not seeds:
        print("no seeds matched", file=sys.stderr)
        return 1

    failures: list[str] = []
    for s in seeds:
        slug = s["seed_slug"]
        print(f"\n{'='*70}\n{slug} — {s['seed_name']} → G2 '{s['g2_category_slug']}'\n{'='*70}")
        if not args.skip_fetch:
            rc = subprocess.run([
                sys.executable, str(HERE / "g2_rapidapi.py"),
                "--seed", slug, "--seed-name", s["seed_name"], "--seed-domain", s["seed_domain"],
                "--description", s.get("description", ""), "--category", s["category"],
                "--category-slug", s["g2_category_slug"], "--max-details", str(args.max_details),
            ]).returncode
            if rc != 0:
                failures.append(f"{slug}: fetch rc={rc}")
                continue
        rc = subprocess.run([
            sys.executable, str(HERE / "build.py"), "freeze", "--seed", slug, "--allow-rejects",
        ]).returncode
        if rc != 0:
            failures.append(f"{slug}: freeze rc={rc}")

    print(f"\n{'='*70}\ndone — {len(seeds) - len(failures)}/{len(seeds)} seeds built")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
