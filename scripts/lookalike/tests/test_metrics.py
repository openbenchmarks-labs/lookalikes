"""P2 metric-registry checks (offline).

1. The `precision_at_k` metric must reproduce the `precision_at_k` already stored
   in every slim fixture file (byte-compatible with the old JudgedRun property).
2. nDCG / MAP / MRR stay in range; pooled recall does the right set math.

Run: PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_metrics.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

from lookalike.common import Candidate, JudgedCandidate
from lookalike import metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
with open(os.path.join(REPO_ROOT, "data", "latest-lookalike.json"), encoding="utf-8") as _snapshot_file:
    DATASET_DIR = os.path.join(REPO_ROOT, "data", "lookalike-runs", json.load(_snapshot_file)["dataset_slug"])


def _judged_from_slim(cands: list[dict]) -> list[JudgedCandidate]:
    out = []
    for c in cands:
        out.append(
            JudgedCandidate(
                candidate=Candidate(
                    name=c["name"], domain=c.get("domain"), description=c.get("description"),
                    rank=c.get("rank"), extra=c.get("extra") or {},
                ),
                relevant=bool(c["relevant"]),
                rationale=c.get("rationale", ""),
            )
        )
    return out


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*", "*.json"))):
        if path.endswith(".raw.json"):
            continue
        d = json.load(open(path, encoding="utf-8"))
        if "candidates" not in d or d.get("precision_at_k") is None:
            continue
        judged = _judged_from_slim(d["candidates"])
        k = d.get("k", 10)
        val, _ = metrics.METRIC_REGISTRY["precision_at_k"].fn(judged, k)
        if val != d["precision_at_k"]:
            failures.append(
                f"{os.path.relpath(path, DATASET_DIR)}: precision_at_k metric {val} != stored {d['precision_at_k']}"
            )
        # range checks on the rank-aware metrics
        for key in ("ndcg_at_k", "map_at_k", "mrr"):
            mv, _ = metrics.METRIC_REGISTRY[key].fn(judged, k)
            if mv is not None and not (0.0 <= mv <= 1.0):
                failures.append(f"{os.path.relpath(path, DATASET_DIR)}: {key}={mv} out of [0,1]")
        checked += 1

    # synthetic recall pool check
    judged = _judged_from_slim([
        {"name": "A", "domain": "a.com", "relevant": True},
        {"name": "B", "domain": "b.com", "relevant": False},
        {"name": "C", "domain": "c.com", "relevant": True},
    ])
    pool = {"a.com", "c.com", "z.com"}  # 3 distinct relevant across all vendors
    rv, detail = metrics.compute_seed_context_metric("recall_at_k", judged, 10, pool)
    if rv != round(100.0 * 2 / 3, 2):
        failures.append(f"recall synthetic: got {rv} ({detail}), expected {round(100.0*2/3,2)}")

    # primary metric sanity
    if metrics.primary_metric().key != "precision_at_k":
        failures.append("primary_metric() is not precision_at_k")

    print(f"checked {checked} slim fixtures across the registry")
    if failures:
        print(f"\n{len(failures)} FAILURE(s):")
        for f in failures[:40]:
            print("  - " + f)
        return 1
    print("PASS — precision_at_k metric matches every stored cell; nDCG/MAP/MRR in range; recall pool math correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
