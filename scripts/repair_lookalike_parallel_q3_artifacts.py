#!/usr/bin/env python3
"""Restore truncated Parallel Q3 slim artifacts from their captured judge calls.

The July Q3 release snapshot retained aggregate metrics for several Parallel
runs but their slim artifacts contained only a prefix of the candidates.  The
raw artifacts retain both the normalized candidate list and the original
per-candidate GPT-5.6 verdicts, so this repair is deterministic and makes no
vendor or LLM calls.

Run with --dry-run first.  Without it, this updates only the affected local
slim JSON files and ``data/lookalike-2026-q3.json``.  Persist with
``backfill_lookalike_to_db.py`` separately, after review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lookalike.common import recompute_leaderboard


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# The OSS repository publishes the current Q3 release as its canonical
# `latest-lookalike.json` snapshot.
SNAPSHOT_PATH = DATA / "latest-lookalike.json"
RUNS = DATA / "lookalike-runs" / "lookalike-2026-q3"


def _precision(candidates: list[dict], cutoff: int) -> float:
    scored = candidates[:cutoff]
    return round(100 * sum(bool(c["relevant"]) for c in scored) / cutoff, 2)


def _restore_cell(cell: dict, *, write: bool) -> bool:
    if cell["provider_slug"] != "parallel" or cell.get("returned_count") is None:
        return False
    slim_path = RUNS / cell["seed_slug"] / "parallel.json"
    raw_path = slim_path.with_suffix(".raw.json")
    slim = json.loads(slim_path.read_text())
    expected = int(cell["returned_count"])
    if len(slim.get("candidates", [])) >= expected:
        return False
    raw = json.loads(raw_path.read_text())
    attempt = next(
        (
            item
            for item in raw.get("attempts", [])
            if item.get("config", {}).get("name") == raw.get("winning_config_name")
        ),
        None,
    )
    if not attempt:
        raise RuntimeError(f"{cell['seed_slug']}: winning raw attempt not found")
    extracted = attempt.get("extracted_candidates") or []
    calls = attempt.get("judge_calls") or []
    verdict_by_rank = {
        call.get("for_candidate_rank"): call.get("parsed_response")
        for call in calls
        if call.get("error") is None and isinstance(call.get("parsed_response"), dict)
    }
    if len(extracted) != expected or len(verdict_by_rank) != expected:
        raise RuntimeError(
            f"{cell['seed_slug']}: expected {expected}; raw candidates={len(extracted)}, "
            f"usable verdicts={len(verdict_by_rank)}"
        )

    candidates = []
    for candidate in extracted:
        verdict = verdict_by_rank.get(candidate.get("rank"))
        if verdict is None or "relevant" not in verdict or not verdict.get("rationale"):
            raise RuntimeError(f"{cell['seed_slug']}: missing verdict for rank {candidate.get('rank')}")
        candidates.append({**candidate, "relevant": bool(verdict["relevant"]), "rationale": verdict["rationale"]})
    candidates.sort(key=lambda item: int(item.get("rank") or 0))

    result = attempt.get("result") or {}
    relevant_count = sum(bool(candidate["relevant"]) for candidate in candidates)
    if result.get("relevant_count") != relevant_count or cell.get("relevant_count") != relevant_count:
        raise RuntimeError(f"{cell['seed_slug']}: reconstructed relevance does not match frozen metrics")
    restored = {
        "dataset_slug": "lookalike-2026-q3",
        "seed_slug": cell["seed_slug"],
        "provider_slug": "parallel",
        "config_name": raw.get("winning_config_name") or "",
        "config": attempt.get("config") or {},
        "k": int(result.get("k") or cell.get("k") or 100),
        "relevant_count": relevant_count,
        # Precision@K retains the requested K=100 denominator even when a
        # provider returns fewer results.  This is the historical release
        # convention captured in the raw attempt result and frozen snapshot.
        "precision_at_k": round(100 * relevant_count / int(result.get("k") or cell.get("k") or 100), 2),
        "relevant_count_at_10": sum(bool(candidate["relevant"]) for candidate in candidates[:10]),
        "relevant_count_at_25": sum(bool(candidate["relevant"]) for candidate in candidates[:25]),
        "relevant_count_at_100": relevant_count,
        "precision_at_10": _precision(candidates, 10),
        "precision_at_25": _precision(candidates, 25),
        "precision_at_100": round(100 * relevant_count / 100, 2),
        "latency_ms": result.get("latency_ms"),
        "cost_usd": result.get("cost_usd"),
        "duplicates_removed": result.get("duplicates_removed") or 0,
        "max_results": None,
        "judge_model": raw.get("judge_model") or cell.get("judge_model"),
        "judged_at": result.get("judged_at") or cell.get("judged_at"),
        "candidates": candidates,
        "error": result.get("error"),
    }
    for key in (
        "precision_at_k", "relevant_count_at_10", "relevant_count_at_25", "relevant_count_at_100",
        "precision_at_10", "precision_at_25", "precision_at_100", "latency_ms", "duplicates_removed",
    ):
        if cell.get(key) != restored[key]:
            raise RuntimeError(f"{cell['seed_slug']}: {key} differs from frozen snapshot")
    if write:
        slim_path.write_text(json.dumps(restored, indent=2) + "\n")
    print(f"{cell['seed_slug']}: {len(slim.get('candidates', []))} -> {expected} candidates")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    restored = sum(_restore_cell(cell, write=not args.dry_run) for cell in snapshot["seed_vendors"])
    # A completed release is a valid no-op: the script should remain usable as
    # a regression check after publication.  Any non-zero partial count still
    # signals an unexpected, incomplete repair.
    if restored not in (0, 18):
        raise RuntimeError(f"expected 18 truncated Parallel cells or a complete no-op, found {restored}")
    if not args.dry_run:
        recompute_leaderboard(snapshot, int(snapshot.get("k") or 100))
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"{'would restore' if args.dry_run else 'restored'} {restored} Parallel artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
