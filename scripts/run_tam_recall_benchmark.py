"""Run the TAM-recall benchmark end-to-end.

Unlike the precision benchmark, recall needs **no LLM judge** — a "hit" is
deterministic membership in a frozen, vendor-independent gold set, matched by
canonical domain (see scripts/lookalike/RECALL_METHODOLOGY.md). That makes the
scoring loop reproducible and model-free.

For each seed that has a frozen gold set (data/lookalike-tam/gold/<seed>.gold.json):
  1. For each vendor, sweep its configs at fetch depth N=100.
  2. Match the returned ranked list against the gold set (recall.match_candidates).
  3. Keep the config with the best Recall@100. Compute Recall@10/50/100,
     R-Precision, Hit@100, returned_count (+ the vendor's max-return ceiling).
  4. Persist the matched-hit map per cell so every recall number is hand-checkable.
  5. Run the cross-vendor fairness audit per seed.
  6. Aggregate the per-vendor leaderboard and write data/latest-lookalike-tam.json.

Usage:
  python3 scripts/run_tam_recall_benchmark.py --mock        # offline, no API keys
  python3 scripts/run_tam_recall_benchmark.py               # live vendor APIs
  python3 scripts/run_tam_recall_benchmark.py --only openfunnel,exa
  python3 scripts/run_tam_recall_benchmark.py --seeds postscript
  python3 scripts/run_tam_recall_benchmark.py --fetch-depth 100
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from lookalike.common import (
    Candidate,
    RunResult,
    Seed,
    capture_http_calls,
    now_iso,
    raw_http_call_to_dict,
)
from lookalike import recall as R
from lookalike.runners import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "lookalike-tam" / "gold"
SEEDS_PATH = ROOT / "data" / "lookalike-tam" / "seeds.json"
SNAPSHOT_PATH = ROOT / "data" / "latest-lookalike-tam.json"
RUNS_DIR = ROOT / "data" / "lookalike-tam-runs"

FETCH_DEPTH = 100
RECALL_KS = (10, 50, 100)
AUDIT_VENDOR = "openfunnel"
THROTTLE_SEC = 0.4
ALL_VENDORS = list(REGISTRY.keys())

# Display metadata for the leaderboard (kept here so the snapshot is self-
# describing without importing the frontend constants).
VENDOR_NAMES = {slug: REGISTRY[slug].VENDOR_NAME for slug in ALL_VENDORS}


def parse_csv(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [s.strip().lower() for s in arg.split(",") if s.strip()] or None


# --------------------------------------------------------------------------- #
# Gold-set discovery                                                          #
# --------------------------------------------------------------------------- #


def discover_goldsets(seed_filter: list[str] | None) -> list[R.GoldSet]:
    out: list[R.GoldSet] = []
    for f in sorted(GOLD_DIR.glob("*.gold.json")):
        gs = R.load_goldset(f)  # raises on hash mismatch — fail loudly
        if seed_filter and gs.seed_slug not in seed_filter:
            continue
        out.append(gs)
    return out


def _load_firmographics() -> dict[str, dict[str, Any]]:
    """Map seed_slug -> firmographic hints from the seed registry. These are
    public/TAM-derived (never gold-derived) and let runners use a vendor's
    documented firmographic filter surface. Empty if the registry is absent."""
    if not SEEDS_PATH.exists():
        return {}
    data = R.json.loads(SEEDS_PATH.read_text("utf-8"))
    return {
        s["seed_slug"]: s["firmographics"]
        for s in data.get("seeds", [])
        if s.get("firmographics")
    }


_FIRMOGRAPHICS = _load_firmographics()


def seed_from_goldset(gs: R.GoldSet) -> Seed:
    """Reconstruct the vendor input Seed from the gold set file. The seed's own
    domain is taken from the gold file's seed metadata if present; otherwise we
    leave it None and vendors fall back to name. Firmographic hints (if any)
    come from the seed registry, not the gold set."""
    data = R.json.loads((GOLD_DIR / f"{gs.seed_slug}.gold.json").read_text("utf-8"))
    return Seed(
        seed_slug=gs.seed_slug,
        seed_name=data.get("seed_name", gs.seed_slug),
        seed_domain=data.get("seed_domain"),
        description=data.get("description"),
        category=gs.category,
        firmographics=_FIRMOGRAPHICS.get(gs.seed_slug),
    )


# --------------------------------------------------------------------------- #
# Mock — TAM-aware so --mock produces a meaningful, differentiated leaderboard #
# --------------------------------------------------------------------------- #


def _vendor_skill(vendor: str, seed_slug: str) -> float:
    """Deterministic per-(vendor, seed) recall skill in [0.15, 0.95].

    The audit vendor gets a mild mock-only edge so the offline leaderboard is
    legible, while every vendor still finds a different, overlapping subset.
    """
    h = hashlib.sha256(f"{vendor}|{seed_slug}".encode()).digest()
    base = 0.15 + (int.from_bytes(h[:2], "big") / 65535.0) * 0.7
    if vendor == AUDIT_VENDOR:
        base = min(0.95, base + 0.12)
    return base


def mock_run(gs: R.GoldSet, vendor: str, depth: int) -> RunResult:
    """Build a fake ranked list: a deterministic subset of gold companies
    interleaved with noise, so recall metrics + matching are exercised offline."""
    skill = _vendor_skill(vendor, gs.seed_slug)
    n_gold = max(1, round(skill * gs.size))
    # pick which gold companies this vendor finds (stable per vendor/seed)
    order = sorted(
        gs.companies,
        key=lambda c: hashlib.sha256(f"{vendor}|{gs.seed_slug}|{c.name}".encode()).hexdigest(),
    )
    found = order[:n_gold]

    candidates: list[Candidate] = []
    gi = 0
    for rank in range(1, depth + 1):
        # sprinkle gold hits across the list with a vendor-specific cadence
        place_gold = gi < len(found) and (rank % 2 == 0 or skill > 0.7)
        if place_gold:
            c = found[gi]
            gi += 1
            candidates.append(Candidate(name=c.name, domain=c.domains[0] if c.domains else None, rank=rank))
        else:
            candidates.append(Candidate(name=f"Noise {vendor} {rank}", domain=f"noise-{vendor}-{rank}.example", rank=rank))
    return RunResult(
        seed_slug=gs.seed_slug,
        provider_slug=vendor,
        config_name="mock",
        config={"name": "mock"},
        candidates=candidates,
        latency_ms=120 + (hash((vendor, gs.seed_slug)) % 400),
        cost_usd=round(0.002 * (depth / 10), 4),
    )


# --------------------------------------------------------------------------- #
# Scoring one (seed, vendor) cell                                             #
# --------------------------------------------------------------------------- #


def score_cell(
    gs: R.GoldSet,
    seed: Seed,
    vendor: str,
    depth: int,
    *,
    mock: bool,
) -> tuple[dict[str, Any] | None, list[R.MatchHit], list[dict[str, Any]], str | None]:
    """Returns (cell, best_hits, raw_attempts, error). Sweeps configs, keeps
    the one with best Recall@max(K)."""
    mod = REGISTRY[vendor]
    configs = [{"name": "mock"}] if mock else list(mod.CONFIGS)
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_hits: list[R.MatchHit] = []
    best_recall = -1.0
    last_error: str | None = None

    for config in configs:
        try:
            if mock:
                run = mock_run(gs, vendor, depth)
                http_dicts: list[dict[str, Any]] = []
            else:
                with capture_http_calls() as calls:
                    run = mod.run(seed, depth, config)
                http_dicts = [raw_http_call_to_dict(c) for c in calls]
        except Exception as exc:  # noqa: BLE001
            last_error = f"{config.get('name','?')}: {exc}"
            attempts.append({"config": dict(config), "result": {"status": "vendor_crash", "error": str(exc)}})
            time.sleep(0 if mock else THROTTLE_SEC)
            continue

        if run.error and not run.candidates:
            last_error = f"{config.get('name','?')}: {run.error}"
            attempts.append({"config": dict(config), "vendor_calls": http_dicts,
                             "result": {"status": "vendor_error", "error": run.error}})
            time.sleep(0 if mock else THROTTLE_SEC)
            continue

        hits = R.match_candidates(run.candidates, gs)
        max_k = max(RECALL_KS)
        recall_max = R.recall_at_k(hits, gs.size, max_k) or 0.0
        cell = {
            "returned_count": len(run.candidates),
            "matched_count": len(hits),
            "recall_at_k": {k: R.recall_at_k(hits, gs.size, k) for k in RECALL_KS},
            "r_precision": R.r_precision(hits, gs.size),
            "hit_at_max": R.hit_at_k(hits, max_k),
            "latency_ms": run.latency_ms,
            "cost_usd": run.cost_usd,
            "config_used": {"name": run.config_name, **(run.config or {})},
        }
        attempts.append({
            "config": dict(config),
            "vendor_calls": http_dicts,
            "matched_hits": [dataclass_hit(h) for h in hits],
            "result": {"status": "ok", **{k: cell[k] for k in ("returned_count", "matched_count", "recall_at_k", "r_precision")}},
        })
        if recall_max > best_recall:
            best_recall, best, best_hits = recall_max, cell, hits
        time.sleep(0 if mock else THROTTLE_SEC)

    return best, best_hits, attempts, last_error


def dataclass_hit(h: R.MatchHit) -> dict[str, Any]:
    return {"rank": h.rank, "gold_name": h.gold_name, "method": h.method, "candidate_domain": h.candidate_domain}


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def _avg(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def build_leaderboard(seed_vendor_cells: list[dict[str, Any]], total_seeds: int) -> list[dict[str, Any]]:
    by_vendor: dict[str, list[dict[str, Any]]] = {}
    for c in seed_vendor_cells:
        by_vendor.setdefault(c["provider_slug"], []).append(c)

    rows: list[dict[str, Any]] = []
    for vendor, cells in by_vendor.items():
        scored = [c for c in cells if c.get("recall_at_10") is not None]
        latencies = [c["latency_ms"] for c in cells if isinstance(c.get("latency_ms"), int)]
        rows.append({
            "provider_slug": vendor,
            "provider_name": VENDOR_NAMES.get(vendor, vendor),
            "fetch_depth": FETCH_DEPTH,
            "seeds_scored": len(scored),
            "total_seeds": total_seeds,
            "avg_recall_at_10": _avg([c["recall_at_10"] for c in scored]),
            "avg_recall_at_50": _avg([c["recall_at_50"] for c in scored]),
            "avg_recall_at_100": _avg([c["recall_at_100"] for c in scored]),
            "avg_r_precision": _avg([c["r_precision"] for c in scored if c["r_precision"] is not None]),
            "hit_at_100_rate": _avg([100.0 * c["hit_at_100"] for c in scored]),
            "total_unique_relevant": sum(int(c.get("matched_count") or 0) for c in scored),
            "max_returned": max((int(c.get("returned_count") or 0) for c in cells), default=0),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "total_cost_usd": round(sum(float(c.get("cost_usd") or 0.0) for c in cells), 4),
        })

    rows.sort(key=lambda r: (-1 if r["avg_recall_at_100"] is None else r["avg_recall_at_100"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def persist_cell_detail(gs: R.GoldSet, vendor: str, attempts: list[dict[str, Any]]) -> None:
    out_dir = RUNS_DIR / gs.seed_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed_slug": gs.seed_slug,
        "provider_slug": vendor,
        "gold_version_hash": gs.version_hash,
        "gold_size": gs.size,
        "fetch_depth": FETCH_DEPTH,
        "captured_at": now_iso(),
        "attempts": attempts,
    }
    (out_dir / f"{vendor}.json").write_text(R.json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    global FETCH_DEPTH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated vendor slugs")
    parser.add_argument("--seeds", help="comma-separated seed slugs (must have gold sets)")
    parser.add_argument("--mock", action="store_true", help="offline TAM-aware mock (no API keys)")
    parser.add_argument("--fetch-depth", type=int, default=FETCH_DEPTH)
    parser.add_argument("--dry-run", action="store_true", help="run but don't write the snapshot")
    args = parser.parse_args()
    FETCH_DEPTH = args.fetch_depth

    goldsets = discover_goldsets(parse_csv(args.seeds))
    if not goldsets:
        print(f"no gold sets found in {GOLD_DIR} (build with goldset/build.py)", file=sys.stderr)
        return 1
    vendors = [v for v in ALL_VENDORS if (not parse_csv(args.only)) or v in parse_csv(args.only)]

    print(f"TAM-recall — seeds={len(goldsets)}, vendors={len(vendors)}, "
          f"fetch_depth={FETCH_DEPTH}, mock={args.mock}")

    seed_meta: list[dict[str, Any]] = []
    gold_version_by_seed: dict[str, str] = {}
    seed_vendor_cells: list[dict[str, Any]] = []
    fairness: list[dict[str, Any]] = []
    failures: list[str] = []

    for gs in goldsets:
        seed = seed_from_goldset(gs)
        gold_version_by_seed[gs.seed_slug] = gs.version_hash
        seed_meta.append({
            "seed_slug": gs.seed_slug, "seed_name": seed.seed_name,
            "seed_domain": seed.seed_domain, "category": gs.category,
            "gold_size": gs.size, "filters": gs.filters,
        })
        print(f"\n[{gs.seed_slug}] {seed.seed_name} ({gs.category}) — gold={gs.size} ({gs.version_hash})")

        per_vendor_hits: dict[str, list[R.MatchHit]] = {}
        for vendor in vendors:
            try:
                cell, hits, attempts, err = score_cell(gs, seed, vendor, FETCH_DEPTH, mock=args.mock)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                failures.append(f"{gs.seed_slug}/{vendor}: {exc}")
                cell, hits, attempts, err = None, [], [], str(exc)

            if not args.dry_run and attempts:
                persist_cell_detail(gs, vendor, attempts)

            per_vendor_hits[vendor] = hits
            base = {
                "seed_slug": gs.seed_slug, "seed_name": seed.seed_name, "category": gs.category,
                "provider_slug": vendor, "provider_name": VENDOR_NAMES.get(vendor, vendor),
                "gold_size": gs.size,
            }
            if cell is None:
                seed_vendor_cells.append({**base, "returned_count": 0, "matched_count": None,
                                          "recall_at_10": None, "recall_at_50": None, "recall_at_100": None,
                                          "r_precision": None, "hit_at_100": None, "latency_ms": None,
                                          "cost_usd": None, "config_used": None, "error": err})
                print(f"  {VENDOR_NAMES.get(vendor, vendor):<14}  FAIL  {err}")
                if err:
                    failures.append(f"{gs.seed_slug}/{vendor}: {err}")
                continue

            seed_vendor_cells.append({**base,
                "returned_count": cell["returned_count"], "matched_count": cell["matched_count"],
                "recall_at_10": cell["recall_at_k"][10], "recall_at_50": cell["recall_at_k"][50],
                "recall_at_100": cell["recall_at_k"][100], "r_precision": cell["r_precision"],
                "hit_at_100": cell["hit_at_max"], "latency_ms": cell["latency_ms"],
                "cost_usd": cell["cost_usd"], "config_used": cell["config_used"], "error": None})
            print(f"  {VENDOR_NAMES.get(vendor, vendor):<14}  "
                  f"R@10={cell['recall_at_k'][10]}%  R@50={cell['recall_at_k'][50]}%  "
                  f"R@100={cell['recall_at_k'][100]}%  ({cell['matched_count']}/{gs.size})")

        audit = R.fairness_audit(per_vendor_hits, gs, audit_vendor=AUDIT_VENDOR)
        audit["seed_slug"] = gs.seed_slug
        fairness.append(audit)

    leaderboard = build_leaderboard(seed_vendor_cells, total_seeds=len(goldsets))

    print("\n=== leaderboard (avg Recall@100) ===")
    for r in leaderboard:
        print(f"  #{r['rank']} {r['provider_name']:<14} R@10={r['avg_recall_at_10']}  "
              f"R@50={r['avg_recall_at_50']}  R@100={r['avg_recall_at_100']}  "
              f"hit@100={r['hit_at_100_rate']}%")

    snapshot = {
        "dataset_slug": "tam-recall-v1",
        "dataset_name": "TAM Recall Benchmark",
        "generated_at": now_iso(),
        "fetch_depth": FETCH_DEPTH,
        "recall_ks": list(RECALL_KS),
        "audit_vendor": AUDIT_VENDOR,
        "gold_version_by_seed": gold_version_by_seed,
        "seeds": seed_meta,
        "leaderboard": leaderboard,
        "seed_vendors": seed_vendor_cells,
        "fairness": fairness,
    }

    if args.dry_run:
        print("\n--dry-run: snapshot not written")
        return 0

    SNAPSHOT_PATH.write_text(R.json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {SNAPSHOT_PATH.relative_to(ROOT)}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures[:20]:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
