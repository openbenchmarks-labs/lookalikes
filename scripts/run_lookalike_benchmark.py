"""Run the lookalike benchmark end-to-end.

For each (seed, vendor) pair:
  1. Sweep every config the runner declares (default + tuned variants).
  2. Take K candidates per config.
  3. LLM-judge each candidate (binary relevance + rationale).
  4. Keep the run with the highest Precision@K. Persist:
       • the aggregate cell (precision, latency, cost, winning config)
         into `data/latest-lookalike.json#seed_vendors`
       • the full per-candidate detail into
         `data/lookalike-runs/<dataset>/<seed>/<vendor>.json`
  5. Re-aggregate the leaderboard rows from the refreshed cells.

Usage:
  python3 scripts/run_lookalike_benchmark.py                    # all vendors, real APIs + real judge
  python3 scripts/run_lookalike_benchmark.py --mock              # offline smoke run (no API keys needed)
  python3 scripts/run_lookalike_benchmark.py --only openfunnel,exa
  python3 scripts/run_lookalike_benchmark.py --seeds stripe,modal
  python3 scripts/run_lookalike_benchmark.py --k 10              # override default K
  python3 scripts/run_lookalike_benchmark.py --judge-model gpt-5
  python3 scripts/run_lookalike_benchmark.py --retry-failed     # only re-run cells that
                                                                #   previously errored
                                                                #   (or are missing) —
                                                                #   useful after topping
                                                                #   up vendor credits

Env (required when running live; ignored under --mock):
  AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY + _URL    judge (Azure-hosted gpt-5)
  OPENFUNNEL_API_KEY
  OCEAN_API_KEY
  PARALLEL_API_KEY
  EXA_API_KEY
  LUSHA_API_KEY
  PREDICT_LEADS_API_TOKEN + PREDICT_LEADS_API_KEY
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
import traceback
from typing import Any

from lookalike.common import (
    DEFAULT_K,
    JudgedRun,
    RunResult,
    Seed,
    capture_http_calls,
    persist_run_detail,
    persist_run_raw,
    raw_http_call_to_dict,
    raw_judge_call_to_dict,
    read_seeds,
    read_snapshot,
    recompute_leaderboard,
    upsert_seed_vendor_cell,
    write_snapshot,
)
from lookalike.judge import JudgePanel, capture_judge_calls, resolve_judge_models
from lookalike.metrics import primary_metric
from lookalike.runners import REGISTRY
from lookalike.runners import mock as mock_runner


THROTTLE_SEC = 0.4
ALL_VENDORS = list(REGISTRY.keys())


def parse_csv(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    out = [s.strip().lower() for s in arg.split(",") if s.strip()]
    return out or None


def _resolve_int(cli_val: int | None, env_name: str, default: int) -> int:
    """CLI flag wins, then env var, then default. Floored at 1."""
    if cli_val is not None:
        return max(1, cli_val)
    raw = os.environ.get(env_name)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def select_seeds(all_seeds: list[Seed], only: list[str] | None) -> list[Seed]:
    if not only:
        return all_seeds
    wanted = set(only)
    return [s for s in all_seeds if s.seed_slug in wanted or s.seed_name.lower() in wanted]


def select_vendors(only: list[str] | None) -> list[str]:
    if not only:
        return ALL_VENDORS
    wanted = set(only)
    return [v for v in ALL_VENDORS if v in wanted]


def best_judged(runs: list[JudgedRun]) -> JudgedRun:
    """Pick the run with the highest primary metric (Precision@K by default).
    Ties broken by:
      1. more judged candidates,
      2. lower latency.
    Routing selection through the metric registry means changing the primary
    metric (e.g. to nDCG) retunes best-of with no other code change."""
    pm = primary_metric()

    def key(j: JudgedRun) -> tuple[float, int, int]:
        val = pm.fn(j.judged, j.k)[0] or 0.0
        return (-val, -j.k, j.run.latency_ms)
    return sorted(runs, key=key)[0]


def get_runner(slug: str, mock: bool):
    if mock:
        # Build a per-vendor closure so the mock matrix differs by vendor.
        provider_name = REGISTRY[slug].VENDOR_NAME
        return {
            "VENDOR_SLUG": slug,
            "VENDOR_NAME": provider_name,
            "CONFIGS": mock_runner.CONFIGS,
            "run": mock_runner.make_runner(slug, provider_name),
        }
    mod = REGISTRY[slug]
    return {
        "VENDOR_SLUG": mod.VENDOR_SLUG,
        "VENDOR_NAME": mod.VENDOR_NAME,
        "CONFIGS": mod.CONFIGS,
        "run": mod.run,
    }


def run_vendor_for_seed(
    runner: dict[str, Any], seed: Seed, k: int, judge: JudgePanel
) -> tuple[JudgedRun | None, str | None, list[dict[str, Any]]]:
    """Sweep every config the runner declares and return:
        (best_judged, error, attempts)
    `attempts` is the per-config audit trail (HTTP req/resp + judge
    prompt/resp) ready to be persisted to the `*.raw.json` file. One
    entry per sweep config, regardless of success."""
    judged_runs: list[JudgedRun] = []
    attempts: list[dict[str, Any]] = []
    last_error: str | None = None

    for config in runner["CONFIGS"]:
        attempt: dict[str, Any] = {
            "config": dict(config),
            "vendor_calls": [],
            "extracted_candidates": [],
            "judge_calls": [],
            "result": {"status": "pending"},
        }

        # Vendor leg — capture every HTTP call the runner makes for this
        # config (preflights like OpenFunnel's lookup-companies + the
        # main search call all land in the same buffer).
        try:
            with capture_http_calls() as http_calls:
                run: RunResult = runner["run"](seed, k, config)
            attempt["vendor_calls"] = [raw_http_call_to_dict(c) for c in http_calls]
        except Exception as exc:  # noqa: BLE001
            attempt["result"] = {
                "status": "vendor_crash",
                "error": f"{type(exc).__name__}: {exc}",
            }
            attempts.append(attempt)
            last_error = f"{config.get('name', '?')}: {exc}"
            time.sleep(THROTTLE_SEC)
            continue

        if run.error and not run.candidates:
            attempt["result"] = {
                "status": "vendor_error",
                "error": run.error,
                "latency_ms": run.latency_ms,
            }
            attempts.append(attempt)
            last_error = f"{config.get('name', '?')}: {run.error}"
            time.sleep(THROTTLE_SEC)
            continue

        if not run.candidates:
            attempt["result"] = {
                "status": "no_candidates",
                "latency_ms": run.latency_ms,
            }
            attempts.append(attempt)
            last_error = f"{config.get('name', '?')}: 0 candidates"
            time.sleep(THROTTLE_SEC)
            continue

        # Snapshot the normalized candidates that were handed to the judge.
        # This is the bridge between "raw vendor response" and "judge input".
        attempt["extracted_candidates"] = [
            {
                "name": c.name,
                "domain": c.domain,
                "description": c.description,
                "rank": c.rank,
                "extra": c.extra,
            }
            for c in run.candidates
        ]

        # Judge leg — capture every LLM round-trip (one per candidate).
        try:
            with capture_judge_calls() as judge_calls:
                judged = judge.score_run(seed, run)
            attempt["judge_calls"] = [raw_judge_call_to_dict(j) for j in judge_calls]
        except Exception as exc:  # noqa: BLE001
            attempt["judge_calls"] = [raw_judge_call_to_dict(j) for j in judge_calls]
            attempt["result"] = {
                "status": "judge_crash",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": run.latency_ms,
            }
            attempts.append(attempt)
            last_error = f"{config.get('name', '?')}: judge failed — {exc}"
            time.sleep(THROTTLE_SEC)
            continue

        attempt["result"] = {
            "status": "ok",
            "k": judged.k,
            "relevant_count": judged.relevant_count,
            "precision_at_k": judged.precision_at_k,
            "latency_ms": run.latency_ms,
            "judged_at": judged.judged_at,
            "cost_usd": run.cost_usd,
            "error": None,
        }
        attempts.append(attempt)
        judged_runs.append(judged)
        time.sleep(THROTTLE_SEC)

    if not judged_runs:
        return None, last_error or "no successful runs", attempts
    return best_judged(judged_runs), None, attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated vendor slugs")
    parser.add_argument("--seeds", help="comma-separated seed slugs or names")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"override K (default = snapshot.k or {DEFAULT_K})",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the offline mock runners + mock judge (no API keys required)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="override the (single) judge model id (default: gpt-5.4-mini from judge.py)",
    )
    parser.add_argument(
        "--judges",
        default=None,
        help=(
            "multi-judge panel: comma-separated model ids (e.g. "
            "gpt-5.4-mini,gpt-5.2,o4-mini); a candidate is relevant by majority vote. "
            "Under --mock, pass an integer count (e.g. 3) or a comma list. "
            "Falls back to LOOKALIKE_JUDGE_MODELS env, then --judge-model, then the default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run everything but don't write the snapshot",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "skip (seed, vendor) pairs that already have a non-null Precision@K — "
            "only re-run cells that previously failed or are missing entirely. "
            "Combine with --only / --seeds to narrow further."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "pin the sweep to a single named runner config (e.g. "
            "`seed_only_agentic` for OpenFunnel). When set, runners whose "
            "CONFIGS list has no matching name are skipped for that cell."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="max (seed, vendor) cells to run at once (default 4; env LOOKALIKE_CELL_CONCURRENCY; 1 = sequential).",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="max judge LLM calls in flight PER cell (default 3; env LOOKALIKE_JUDGE_CONCURRENCY). Live ceiling = concurrency x this.",
    )
    args = parser.parse_args()

    snapshot = read_snapshot()
    seeds = select_seeds(read_seeds(), parse_csv(args.seeds))
    vendors = select_vendors(parse_csv(args.only))
    k = args.k or snapshot.get("k") or DEFAULT_K

    if not seeds:
        print("no seeds matched filter", file=sys.stderr)
        return 1
    if not vendors:
        print("no vendors matched filter", file=sys.stderr)
        return 1

    cell_concurrency = _resolve_int(args.concurrency, "LOOKALIKE_CELL_CONCURRENCY", 4)
    judge_concurrency = _resolve_int(args.judge_concurrency, "LOOKALIKE_JUDGE_CONCURRENCY", 3)
    prompt_version = os.environ.get("LOOKALIKE_JUDGE_PROMPT") or "v1"
    if args.mock:
        jt = (args.judges or "").strip()
        n_mock = int(jt) if jt.isdigit() else (len([t for t in jt.split(",") if t.strip()]) or 1)
        judge = JudgePanel(models=["mock"] * n_mock, mock=True, prompt_version=prompt_version,
                           concurrency=judge_concurrency)
    else:
        models = resolve_judge_models(args.judges, args.judge_model)
        judge = JudgePanel(models=models, mock=False, prompt_version=prompt_version,
                           concurrency=judge_concurrency)

    dataset_slug = snapshot.get("dataset_slug") or "lookalike"

    # Build a quick lookup of previously persisted cells so --retry-failed can
    # skip pairs that already have a non-null Precision@K. A cell whose
    # precision_at_k is None either failed last time or was never attempted.
    existing_cells: dict[tuple[str, str], dict[str, Any]] = {
        (c["seed_slug"], c["provider_slug"]): c
        for c in snapshot.get("seed_vendors", [])
    }

    print(
        f"running lookalike benchmark — k={k}, judge={judge.label()}, "
        f"seeds={len(seeds)}, vendors={len(vendors)}, mock={args.mock}, "
        f"retry_failed={args.retry_failed}, prompt={prompt_version}, "
        f"cells_at_once={cell_concurrency}, judges_at_once_per_cell={judge_concurrency}"
    )

    total_cells = 0
    failed_cells = 0
    skipped_cells = 0
    failures: list[dict[str, str]] = []  # {seed, vendor, error}

    # Build the work list up front so --retry-failed / --config skips are counted
    # without spawning workers. Each item is the (seed, resolved-runner) to run.
    work: list[tuple[Seed, dict[str, Any]]] = []
    for seed in seeds:
        for vendor_slug in vendors:
            runner = get_runner(vendor_slug, args.mock)
            tag = f"[{seed.seed_slug}] {runner['VENDOR_NAME']:<14}"

            prior = existing_cells.get((seed.seed_slug, vendor_slug))
            if args.retry_failed and prior and prior.get("precision_at_k") is not None:
                print(
                    f"{tag} SKIP   already judged "
                    f"(P@{prior.get('k', k)}={prior['precision_at_k']}%, "
                    f"config={(prior.get('config_used') or {}).get('name')})"
                )
                skipped_cells += 1
                continue

            if args.config:
                pinned = [c for c in runner["CONFIGS"] if c.get("name") == args.config]
                if not pinned:
                    print(f"{tag} SKIP   no config named '{args.config}'")
                    skipped_cells += 1
                    continue
                runner = {**runner, "CONFIGS": pinned}

            work.append((seed, runner))

    # Persist + log + tally one finished cell. ALWAYS runs on the main thread (in
    # the as_completed loop), so the shared snapshot/leaderboard/counters are never
    # touched concurrently. `result` is (judged, err, attempts) or an Exception.
    def handle(seed: Seed, runner: dict[str, Any], result: Any) -> None:
        nonlocal total_cells, failed_cells
        tag = f"[{seed.seed_slug}] {runner['VENDOR_NAME']:<14}"

        if isinstance(result, BaseException):
            traceback.print_exception(type(result), result, result.__traceback__)
            upsert_seed_vendor_cell(
                snapshot, dataset_slug=dataset_slug, seed=seed,
                provider_slug=runner["VENDOR_SLUG"], provider_name=runner["VENDOR_NAME"],
                k=k, judged=None, error=str(result),
            )
            print(f"{tag} CRASH  {result}")
            failures.append({"seed": seed.seed_slug, "vendor": runner["VENDOR_SLUG"], "error": str(result)})
            failed_cells += 1
            return

        judged, err, attempts = result
        if judged is None:
            upsert_seed_vendor_cell(
                snapshot, dataset_slug=dataset_slug, seed=seed,
                provider_slug=runner["VENDOR_SLUG"], provider_name=runner["VENDOR_NAME"],
                k=k, judged=None, error=err,
            )
            # Persist the raw audit trail even on full failure so the open-source
            # bundle still shows which configs were tried and why each failed.
            if attempts and not args.dry_run:
                persist_run_raw(
                    dataset_slug, seed=seed,
                    provider_slug=runner["VENDOR_SLUG"], provider_name=runner["VENDOR_NAME"],
                    k=k, judge_model=judge.label(), winning_config_name=None, attempts=attempts,
                )
            print(f"{tag} FAIL   {err}")
            failures.append({"seed": seed.seed_slug, "vendor": runner["VENDOR_SLUG"], "error": err or "no successful runs"})
            failed_cells += 1
            return

        upsert_seed_vendor_cell(
            snapshot, dataset_slug=dataset_slug, seed=seed,
            provider_slug=runner["VENDOR_SLUG"], provider_name=runner["VENDOR_NAME"],
            k=k, judged=judged,
        )
        if not args.dry_run:
            persist_run_detail(dataset_slug, judged)
            persist_run_raw(
                dataset_slug, seed=seed,
                provider_slug=runner["VENDOR_SLUG"], provider_name=runner["VENDOR_NAME"],
                k=k, judge_model=judge.label(),
                winning_config_name=judged.run.config_name, attempts=attempts,
            )
        print(
            f"{tag} P@{k}={judged.precision_at_k}%  "
            f"({judged.relevant_count}/{judged.k} relevant)  "
            f"config={judged.run.config_name}  lat={judged.run.latency_ms}ms"
        )
        total_cells += 1

    # Run cells. Workers stay pure (no shared-state writes); each enters its own
    # capture_http_calls()/capture_judge_calls() in its thread, and the nested
    # judge fan-out propagates that context via copy_context() — so audit traces
    # are captured correctly and cells never cross-contaminate.
    if cell_concurrency <= 1 or len(work) <= 1:
        for seed, runner in work:
            try:
                res: Any = run_vendor_for_seed(runner, seed, k, judge)
            except Exception as exc:  # noqa: BLE001
                res = exc
            handle(seed, runner, res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cell_concurrency) as pool:
            fut_meta = {
                pool.submit(run_vendor_for_seed, runner, seed, k, judge): (seed, runner)
                for seed, runner in work
            }
            for fut in concurrent.futures.as_completed(fut_meta):
                seed, runner = fut_meta[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = exc
                handle(seed, runner, res)

    snapshot["k"] = k
    # Canonical judge label: bare model id for a single judge (byte-compatible
    # with today), "majority(n=K): a,b,c" for a panel, "mock-judge" under --mock.
    snapshot["judge_model"] = judge.label()
    recompute_leaderboard(snapshot, k)

    if args.dry_run:
        print(
            f"\n--dry-run, snapshot not written. "
            f"cells_judged={total_cells}, failed={failed_cells}, skipped={skipped_cells}"
        )
        _print_failure_summary(failures)
        return 0

    write_snapshot(snapshot)
    print(
        f"\nwrote data/latest-lookalike.json — cells_judged={total_cells}, "
        f"failed={failed_cells}, skipped={skipped_cells}"
    )
    _print_failure_summary(failures)
    return 0


def _print_failure_summary(failures: list[dict[str, str]]) -> None:
    """Group failures by vendor so the user can see at a glance which keys
    need attention (usually: top up credits, fix scope, re-issue token)."""
    if not failures:
        print("\nno failures — all (seed × vendor) pairs judged")
        return

    by_vendor: dict[str, list[dict[str, str]]] = {}
    for f in failures:
        by_vendor.setdefault(f["vendor"], []).append(f)

    print(f"\n{len(failures)} failed cell(s) across {len(by_vendor)} vendor(s):")
    for vendor, items in sorted(by_vendor.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  [{vendor}]  {len(items)} failure(s)")
        # Dedupe identical error strings so 10× the same auth error reads cleanly.
        by_err: dict[str, list[str]] = {}
        for f in items:
            by_err.setdefault(f["error"], []).append(f["seed"])
        for err, seeds_for_err in by_err.items():
            seeds_preview = ", ".join(seeds_for_err[:4])
            extra = f" (+{len(seeds_for_err) - 4} more)" if len(seeds_for_err) > 4 else ""
            print(f"    seeds: {seeds_preview}{extra}")
            print(f"    error: {err}")

    print(
        "\nto retry only these cells after fixing the underlying issue, run:\n"
        "  npm run lookalike:retry          # all vendors, only failed pairs\n"
        "  python scripts/run_lookalike_benchmark.py --retry-failed --only "
        + ",".join(sorted(by_vendor.keys()))
    )


if __name__ == "__main__":
    raise SystemExit(main())
