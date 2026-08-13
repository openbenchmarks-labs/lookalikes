"""Re-judge the lookalike benchmark from saved raw audit files — no vendor calls.

Every `data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json` stores, per
config attempt, the exact candidates that were handed to the judge
(`extracted_candidates`). This driver rebuilds those runs, scores them with the
CURRENT judge panel + prompt, re-picks each cell's winning config, and updates
the snapshot, per-cell detail files, raw audit files, and leaderboard exactly
like a live run — minus the vendor fetch.

Intended flow for a re-baseline after a judge/prompt change:
  1. Re-fetch only the vendors whose INPUT changed (NL-query vendors):
       PYTHONPATH=scripts .venv/bin/python scripts/run_lookalike_benchmark.py \
           --only exa,parallel --judges gpt-5.6-terra,kimi-k3,claude-opus-5
  2. Re-judge everyone else from disk (this script):
       PYTHONPATH=scripts .venv/bin/python scripts/rejudge_lookalike_from_raw.py \
           --skip-vendors exa,parallel --judges gpt-5.6-terra,kimi-k3,claude-opus-5

Run order note: list panel judges cheapest-first — the gated tiebreak seat
(last) is only consulted on disagreements.

Writes the new judge calls back into the same raw files (vendor_calls and
extracted_candidates are preserved verbatim — the HTTP evidence is untouched).
Use --dry-run to score without writing anything.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import traceback
from pathlib import Path
from typing import Any

from lookalike.common import (
    Candidate,
    JudgedRun,
    RunResult,
    Seed,
    load_dotenv,
    now_iso,
    persist_run_detail,
    persist_run_raw,
    raw_judge_call_to_dict,
    read_seeds,
    read_snapshot,
    recompute_leaderboard,
    upsert_seed_vendor_cell,
    write_snapshot,
)
from lookalike.judge import JudgePanel, capture_judge_calls, resolve_judge_models

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "lookalike-runs"

import json
import os


def parse_csv(v: str | None) -> list[str]:
    return [t.strip() for t in v.split(",") if t.strip()] if v else []


def slim_max_results(raw_path: Path) -> int | None:
    """Endpoint result cap from the sibling slim artifact.

    Raw files do not store max_results, so the cap normally rides in on the
    prior snapshot cell. A vendor whose cells are being judged into the
    snapshot for the first time has no prior cell, and losing the cap would
    silently score a capped endpoint at every cutoff it cannot reach. The slim
    file records it, so fall back to that before giving up.
    """
    slim = raw_path.with_name(raw_path.name.replace(".raw.json", ".json"))
    if not slim.is_file():
        return None
    try:
        return json.loads(slim.read_text(encoding="utf-8")).get("max_results")
    except Exception:  # noqa: BLE001
        return None


def rebuild_run(
    seed: Seed,
    provider_slug: str,
    attempt: dict[str, Any],
    file_k: int,
    prior_cell: dict[str, Any] | None,
    fallback_max_results: int | None = None,
) -> RunResult:
    """Reconstruct the RunResult a config attempt produced, from its persisted
    extracted_candidates + result metadata."""
    res = attempt.get("result") or {}
    candidates = [
        Candidate(
            name=c.get("name") or "",
            domain=c.get("domain"),
            description=c.get("description"),
            rank=c.get("rank"),
            extra=c.get("extra") or {},
        )
        for c in attempt.get("extracted_candidates") or []
    ]
    return RunResult(
        seed_slug=seed.seed_slug,
        provider_slug=provider_slug,
        config_name=(attempt.get("config") or {}).get("name") or "?",
        config=attempt.get("config") or {},
        candidates=candidates,
        latency_ms=res.get("latency_ms") or 0,
        cost_usd=res.get("cost_usd"),
        requested_k=res.get("k") or file_k,
        duplicates_removed=res.get("duplicates_removed") or 0,
        # max_results isn't stored in raw files; carry it over from the prior
        # snapshot cell (e.g. PredictLeads' 25-result ceiling), falling back to
        # the slim artifact for a vendor that has no prior cell yet.
        max_results=(prior_cell or {}).get("max_results") or fallback_max_results,
    )


def rejudge_file(
    raw_path: Path,
    seed: Seed,
    judge: JudgePanel,
    prior_cell: dict[str, Any] | None,
) -> tuple[JudgedRun | None, str | None, list[dict[str, Any]]]:
    """Re-judge every judgeable attempt in one raw file. Returns
    (best_judged, error, new_attempts) — the same contract as the
    orchestrator's run_vendor_for_seed()."""
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    file_k = raw.get("k") or 100
    provider_slug = raw["provider_slug"]

    judged_runs: list[JudgedRun] = []
    new_attempts: list[dict[str, Any]] = []
    last_error: str | None = None

    for attempt in raw.get("attempts") or []:
        # Attempts that never produced candidates (vendor errors, empty
        # results) are carried over untouched — there is nothing to re-judge.
        if not attempt.get("extracted_candidates"):
            new_attempts.append(
                {**attempt, "judge_calls": attempt.get("judge_calls") or []}
            )
            status = (attempt.get("result") or {}).get("status")
            last_error = f"{(attempt.get('config') or {}).get('name', '?')}: {status}"
            continue

        run = rebuild_run(
            seed, provider_slug, attempt, file_k, prior_cell,
            fallback_max_results=slim_max_results(raw_path),
        )
        try:
            with capture_judge_calls() as judge_calls:
                judged = judge.score_run(seed, run)
            new_judge_calls = [raw_judge_call_to_dict(j) for j in judge_calls]
        except Exception as exc:  # noqa: BLE001
            new_attempts.append(
                {
                    **{k: attempt[k] for k in ("config", "vendor_calls", "extracted_candidates")},
                    "judge_calls": [raw_judge_call_to_dict(j) for j in judge_calls],
                    "result": {
                        "status": "judge_crash",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": run.latency_ms,
                    },
                }
            )
            last_error = f"{run.config_name}: judge failed — {exc}"
            continue

        new_attempts.append(
            {
                "config": attempt.get("config") or {},
                "vendor_calls": attempt.get("vendor_calls") or [],
                "extracted_candidates": attempt.get("extracted_candidates"),
                "judge_calls": new_judge_calls,
                "result": {
                    "status": "ok",
                    "k": judged.k,
                    "relevant_count": judged.relevant_count,
                    "precision_at_k": judged.precision_at_k,
                    "latency_ms": run.latency_ms,
                    "judged_at": judged.judged_at,
                    "cost_usd": run.cost_usd,
                    "duplicates_removed": run.duplicates_removed,
                    "error": None,
                    "rejudged_from": raw.get("judge_model"),
                },
            }
        )
        judged_runs.append(judged)

    if not judged_runs:
        return None, last_error or "no judgeable attempts", new_attempts

    # Same winner selection as the live orchestrator.
    from run_lookalike_benchmark import best_judged

    return best_judged(judged_runs), None, new_attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", help="comma-separated seed slugs (default: all in snapshot)")
    parser.add_argument("--only", help="comma-separated vendor slugs to re-judge")
    parser.add_argument(
        "--skip-vendors",
        default="exa,parallel",
        help="vendors NOT to re-judge from disk because they need a live re-fetch "
        "(default: exa,parallel — their NL query changed). Pass '' to re-judge all.",
    )
    parser.add_argument("--judges", default=None, help="comma-separated judge model ids (panel)")
    parser.add_argument("--judge-model", default=None, help="single judge model id")
    parser.add_argument("--mock", action="store_true", help="mock judges (offline, free)")
    parser.add_argument("--dry-run", action="store_true", help="score but write nothing")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip cells already judged by this exact panel (for restarting an interrupted run)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="cells at once (default 4)")
    parser.add_argument("--judge-concurrency", type=int, default=0, help="judge calls per cell")
    args = parser.parse_args()

    load_dotenv()
    snapshot = read_snapshot()
    dataset_slug = snapshot.get("dataset_slug") or "lookalike"
    k = snapshot.get("k") or 100
    raw_dir = RUNS_DIR / dataset_slug
    if not raw_dir.is_dir():
        print(f"no raw runs at {raw_dir}", file=sys.stderr)
        return 1

    seeds = read_seeds()
    want_seeds = set(parse_csv(args.seeds))
    if want_seeds:
        seeds = [s for s in seeds if s.seed_slug in want_seeds]
    skip = set(parse_csv(args.skip_vendors))
    only = set(parse_csv(args.only))

    prompt_version = os.environ.get("LOOKALIKE_JUDGE_PROMPT") or "v1"
    if args.mock:
        judge = JudgePanel(models=["mock"] * 3, mock=True, prompt_version=prompt_version,
                           concurrency=args.judge_concurrency)
    else:
        models = resolve_judge_models(args.judges, args.judge_model)
        judge = JudgePanel(models=models, mock=False, prompt_version=prompt_version,
                           concurrency=args.judge_concurrency)

    prior_cells: dict[tuple[str, str], dict[str, Any]] = {
        (c["seed_slug"], c["provider_slug"]): c for c in snapshot.get("seed_vendors", [])
    }

    # Work list: every (seed, vendor raw file) present on disk.
    work: list[tuple[Seed, Path, str]] = []
    for seed in seeds:
        seed_dir = raw_dir / seed.seed_slug
        if not seed_dir.is_dir():
            print(f"[{seed.seed_slug}] no raw dir — skipped")
            continue
        for f in sorted(seed_dir.glob("*.raw.json")):
            vendor = f.name.replace(".raw.json", "")
            if vendor in skip or (only and vendor not in only):
                continue
            if args.resume:
                prior = prior_cells.get((seed.seed_slug, vendor))
                if prior and prior.get("judge_model") == judge.label() and prior.get("precision_at_k") is not None:
                    continue
            work.append((seed, f, vendor))

    print(
        f"re-judging {dataset_slug} from raw — cells={len(work)}, judge={judge.label()}, "
        f"prompt={prompt_version}, skip={sorted(skip) or '-'}, dry_run={args.dry_run}"
    )

    total = failed = 0

    def handle(seed: Seed, vendor: str, result: Any) -> None:
        nonlocal total, failed
        tag = f"[{seed.seed_slug}] {vendor:<14}"
        if isinstance(result, BaseException):
            traceback.print_exception(type(result), result, result.__traceback__)
            print(f"{tag} CRASH  {result}")
            failed += 1
            return
        judged, err, attempts = result
        prov_name = (prior_cells.get((seed.seed_slug, vendor)) or {}).get("provider_name") or vendor
        if judged is None:
            upsert_seed_vendor_cell(
                snapshot, dataset_slug=dataset_slug, seed=seed,
                provider_slug=vendor, provider_name=prov_name, k=k, judged=None, error=err,
            )
            print(f"{tag} FAIL   {err}")
            failed += 1
            return
        upsert_seed_vendor_cell(
            snapshot, dataset_slug=dataset_slug, seed=seed,
            provider_slug=vendor, provider_name=prov_name, k=k, judged=judged,
        )
        if not args.dry_run:
            persist_run_detail(dataset_slug, judged)
            persist_run_raw(
                dataset_slug, seed=seed, provider_slug=vendor, provider_name=prov_name,
                k=k, judge_model=judge.label(),
                winning_config_name=judged.run.config_name, attempts=attempts,
            )
        print(
            f"{tag} P@{judged.k}={judged.precision_at_k}%  "
            f"({judged.relevant_count}/{judged.k} relevant)  config={judged.run.config_name}"
        )
        total += 1
        # Checkpoint after every cell so an interrupted run resumes via --resume.
        # handle() always runs on the main thread, so this write is safe.
        if not args.dry_run:
            write_snapshot(snapshot)

    def run_one(seed: Seed, path: Path) -> tuple[JudgedRun | None, str | None, list[dict[str, Any]]]:
        vendor = path.name.replace(".raw.json", "")
        return rejudge_file(path, seed, judge, prior_cells.get((seed.seed_slug, vendor)))

    if args.concurrency <= 1 or len(work) <= 1:
        for seed, path, vendor in work:
            try:
                res: Any = run_one(seed, path)
            except Exception as exc:  # noqa: BLE001
                res = exc
            handle(seed, vendor, res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(run_one, seed, path): (seed, vendor) for seed, path, vendor in work}
            for fut in concurrent.futures.as_completed(futs):
                seed, vendor = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = exc
                handle(seed, vendor, res)

    snapshot["k"] = k
    snapshot["judge_model"] = judge.label()
    snapshot["generated_at"] = now_iso()
    recompute_leaderboard(snapshot, k)

    if args.dry_run:
        print(f"\n--dry-run, nothing written. rejudged={total}, failed={failed}")
        return 0 if failed == 0 else 1

    write_snapshot(snapshot)
    print(f"\nsnapshot written. rejudged={total}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
