"""Cost is the list price of what the vendor sent, and the leaderboard derives relevant per $1 at
100 and p50 / p99 latency from the cells.

  PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_cost.py
"""
from lookalike.common import recompute_leaderboard
from lookalike.generic_runner import compute_cost
from lookalike.spec_loader import load_spec

failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    print(("  ok: " if cond else "  FAIL: ") + msg)
    failures += 0 if cond else 1


ce, disco, seltz = load_spec("companyenrich"), load_spec("discolike"), load_spec("seltz")
check(abs(compute_cost(ce, 100, 100) - 4.9) < 1e-9, "per-row vendor billed on all 100 rows sent")
check(abs(compute_cost(ce, 120, 100) - 4.9) < 1e-9, "rows past the depth asked for are not billed")
check(abs(compute_cost(disco, 100, 100) - (0.18 + 0.35)) < 1e-9, "flat + per-row vendor")
check(compute_cost(seltz, 100, 100) == 0.005 and compute_cost(seltz, 7, 100) == 0.005, "flat vendor ignores row count")
check(compute_cost(load_spec("lusha"), 50, 100) is None, "unpriced vendor has no cost")

cell = lambda seed, lat, rel, cost: {"dataset_slug": "d", "seed_slug": seed, "provider_slug": "v", "provider_name": "V",
                                     "precision_at_k": rel, "relevant_count_at_100": rel, "relevant_count": rel,
                                     "latency_ms": lat, "cost_usd": cost, "returned_count": 100}
snap = {"seeds": [{}] * 3, "seed_vendors": [cell("a", 1000, 50, 0.5), cell("b", 2000, 70, 0.5), cell("c", 9000, 30, 0.5)],
        "leaderboard": []}
recompute_leaderboard(snap, 100)
row = snap["leaderboard"][0]
check(row["relevant_per_usd_at_100"] == 100.0, f"relevant per $1 = 150 / $1.50 (got {row['relevant_per_usd_at_100']})")
check(row["p50_latency_ms"] == 2000 and row["p99_latency_ms"] == 9000, "p50 / p99 latency, nearest rank")

print("ok" if failures == 0 else f"\n{failures} failure(s)")
raise SystemExit(1 if failures else 0)
