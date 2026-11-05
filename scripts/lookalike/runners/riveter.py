"""Riveter runner — one dataset per seed, extended with generated rows.

Riveter is a dataset engine, not a lookalike index: it has no "similar
companies" endpoint and no pre-built company graph. This arm uses the public
API's row generation (the app's "Build a list"), which searches and scrapes
the web at request time. Per seed:

  1. POST /datasets — a dataset whose input columns are the three fields the
     judge reads (Website, Company name, What it does) holding one row: the
     seed itself. Free. The first input column (Website) is the index the
     dataset dedupes on.
  2. POST /datasets/{id}/rows/build — `count` = K more rows for the seed's
     authored lookalike `query`, the same string Exa and Parallel receive.
     The generator is shown the existing row (the seed) and told to skip it.
     Paid: 4 credits per row requested, refunded for rows not found.
  3. GET /runs/{id}?wait=50 — long-poll until the generation run is terminal.
  4. The generator stops when it judges the well dry, often short of K. While
     fewer than K rows have landed and the last build added some, steps 2–3
     repeat without a prompt — the API's "extend": the previous build's
     criteria, the existing rows shown to the generator so it steers away
     from repeats — for the remainder, up to `max_build_rounds` builds.
  5. GET /datasets/{id}/rows — the rows in generation order. The seed row
     (position 1) is dropped; the rest become candidates in rank order.

Latency is the wall time of all calls — minutes rather than seconds, since
the rows are researched per request. Cost is the builds' `credits_used`
summed, at Riveter's Self-Serve rate of $0.025 per credit.

Env: RIVETER_API_KEY (required); RIVETER_API_BASE to point at another host
(default https://api.riveterhq.com/v1); RIVETER_BUILD_TIMEOUT_SEC to cap the
wait for one generation run (default 2400).
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..common import Candidate, RunResult, Seed, http_request, require_env, take_top
from ..generic_runner import _without_duplicates, _without_seed

VENDOR_SLUG = "riveter"
VENDOR_NAME = "Riveter"
CONFIGS: list[dict[str, Any]] = [{"name": "extend_seed_dataset", "max_build_rounds": 3}]
DEFAULT_MAX_BUILD_ROUNDS = 3

DEFAULT_API_BASE = "https://api.riveterhq.com/v1"
USD_PER_CREDIT = 0.025  # Self-Serve rate, https://www.riveterhq.com/pricing
DEFAULT_BUILD_TIMEOUT_SEC = 2400
POLL_WAIT_SEC = 50          # server-side long-poll per GET /runs/{id}
POLL_HTTP_TIMEOUT_SEC = 75  # must outlast the server's wait
TERMINAL_RUN_STATUSES = {"success", "stopped"}

# The input columns of the per-seed dataset, in order. The first is the index
# column (what the dataset dedupes on), and all three are what the judge reads.
WEBSITE_COLUMN = "Website"
NAME_COLUMN = "Company name"
DESCRIPTION_COLUMN = "What it does"
INPUT_COLUMNS = [WEBSITE_COLUMN, NAME_COLUMN, DESCRIPTION_COLUMN]

# Directory / profile hosts that are never a company's own website. Same list
# as the Exa, Parallel, and Seltz hooks, so a LinkedIn URL never lands in
# `domain` and skews the cross-vendor dedupe.
AGGREGATOR_HOSTS = {
    "linkedin.com", "tracxn.com", "platform.tracxn.com", "crunchbase.com",
    "wikipedia.org", "en.wikipedia.org", "bloomberg.com", "pitchbook.com",
}


class RiveterApiError(Exception):
    """A non-2xx answer from the Riveter API, with the envelope's message."""


def _host(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    host = value.strip().replace("https://", "").replace("http://", "").split("/")[0].lower()
    host = host[4:] if host.startswith("www.") else host
    if not host or any(host == h or host.endswith("." + h) for h in AGGREGATOR_HOSTS):
        return None
    return host


def _api_base() -> str:
    """The `/v1` root. RIVETER_API_BASE may name the host alone
    (`http://localhost:3000`) or the root itself (`https://api.riveterhq.com/v1`)."""
    base = (os.environ.get("RIVETER_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _build_timeout_sec() -> int:
    raw = os.environ.get("RIVETER_BUILD_TIMEOUT_SEC")
    return int(raw) if raw and raw.isdigit() else DEFAULT_BUILD_TIMEOUT_SEC


def _error_message(status: int, payload: Any) -> str:
    detail = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        return f"HTTP {status}: {detail.get('type')}: {detail.get('message')}"
    return f"HTTP {status}: {str(payload)[:200]}"


def _call(method: str, path: str, headers: dict[str, str], body: Any = None, timeout: int | None = None) -> Any:
    kwargs: dict[str, Any] = {"headers": headers, "body": body}
    if timeout is not None:
        kwargs["timeout"] = timeout
    status, payload, _ = http_request(method, f"{_api_base()}{path}", **kwargs)
    if status >= 300:
        raise RiveterApiError(_error_message(status, payload))
    return payload


def lookalike_prompt(seed: Seed) -> str:
    """The seed's authored lookalike query, verbatim — the same string Exa
    and Parallel receive. Composed from the seed only when a seed has none."""
    if seed.query and seed.query.strip():
        return seed.query.strip()
    who = f"{seed.seed_name} ({seed.seed_domain})" if seed.seed_domain else seed.seed_name
    identity = seed.short_description or seed.description
    return f"Find companies similar to {who}: {identity}" if identity else f"Find companies similar to {who}"


def seed_row(seed: Seed) -> dict[str, str]:
    return {
        WEBSITE_COLUMN: seed.seed_domain or "",
        NAME_COLUMN: seed.seed_name,
        DESCRIPTION_COLUMN: seed.short_description or seed.description or "",
    }


def candidates_from_rows(rows: list[Any]) -> list[Candidate]:
    """Every row after the seed row (position 1), in position order, as a
    Candidate. A row without a company name is dropped; a row whose website is
    a directory page keeps its name and loses the domain."""
    ordered = sorted((r for r in rows if isinstance(r, dict)), key=lambda r: r.get("position") or 0)
    candidates: list[Candidate] = []
    for row in ordered:
        if (row.get("position") or 0) <= 1:
            continue
        inputs = row.get("input") if isinstance(row.get("input"), dict) else {}
        name = inputs.get(NAME_COLUMN)
        if not isinstance(name, str) or not name.strip():
            continue
        description = inputs.get(DESCRIPTION_COLUMN)
        candidates.append(
            Candidate(
                name=name.strip(),
                domain=_host(inputs.get(WEBSITE_COLUMN)),
                description=description.strip() if isinstance(description, str) and description.strip() else None,
            )
        )
    return candidates


def _wait_for_run(run_id: str, headers: dict[str, str]) -> dict[str, Any]:
    deadline = time.monotonic() + _build_timeout_sec()
    while True:
        run = _call("GET", f"/runs/{run_id}?wait={POLL_WAIT_SEC}", headers, timeout=POLL_HTTP_TIMEOUT_SEC)
        if not isinstance(run, dict):
            raise RiveterApiError(f"unexpected run payload: {str(run)[:200]}")
        if run.get("status") in TERMINAL_RUN_STATUSES:
            return run
        if time.monotonic() >= deadline:
            raise RiveterApiError(f"run {run_id} still {run.get('status')} after {_build_timeout_sec()} s")


def _fetch_rows(dataset_id: str, headers: dict[str, str]) -> list[Any]:
    rows: list[Any] = []
    cursor: str | None = None
    while True:
        query = "?limit=1000" + (f"&cursor={cursor}" if cursor else "")
        page = _call("GET", f"/datasets/{dataset_id}/rows{query}", headers)
        if not isinstance(page, dict):
            raise RiveterApiError(f"unexpected rows payload: {str(page)[:200]}")
        rows.extend(page.get("rows") or [])
        cursor = page.get("next_cursor")
        if not cursor:
            return rows


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
    return {
        "run_id": run.get("id"),
        "run_status": run.get("status"),
        "stopped_reason": run.get("stopped_reason"),
        "app_url": run.get("app_url"),
        "credits_used": run.get("credits_used"),
        "requested_rows": progress.get("requested_rows"),
        "generated_rows": progress.get("generated_rows"),
        "added_rows": progress.get("added_rows"),
        "skipped_duplicate_rows": progress.get("skipped_duplicate_rows"),
    }


def _credits_used(builds: list[dict[str, Any]]) -> float:
    return sum(b["credits_used"] for b in builds if isinstance(b.get("credits_used"), (int, float)))


def _cost_usd(builds: list[dict[str, Any]]) -> float | None:
    if not any(isinstance(b.get("credits_used"), (int, float)) for b in builds):
        return None
    return round(_credits_used(builds) * USD_PER_CREDIT, 4)


def _result(seed: Seed, config: dict[str, Any], candidates: list[Candidate], latency_ms: int, *,
            k: int, out_config: dict[str, Any], cost_usd: float | None = None, error: str | None = None,
            duplicates_removed: int = 0) -> RunResult:
    return RunResult(
        seed_slug=seed.seed_slug, provider_slug=VENDOR_SLUG, config_name=config["name"], config=out_config,
        candidates=candidates, latency_ms=latency_ms, cost_usd=cost_usd, error=error, requested_k=k,
        duplicates_removed=duplicates_removed,
    )


def _create_seed_dataset(seed: Seed, headers: dict[str, str]) -> str:
    dataset = _call("POST", "/datasets", headers, body={
        "name": f"Lookalikes: {seed.seed_name}",
        "input_columns": INPUT_COLUMNS,
        "rows": [seed_row(seed)],
    })
    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if not dataset_id:
        raise RiveterApiError(f"dataset create returned no id: {str(dataset)[:200]}")
    return dataset_id


def _build_rows(dataset_id: str, headers: dict[str, str], *, count: int, prompt: str | None) -> dict[str, Any]:
    """One generation run to a terminal state. `prompt` on the first build; the
    later ones omit it and extend the previous build's criteria."""
    body: dict[str, Any] = {"count": count}
    if prompt is not None:
        body["prompt"] = prompt
    build = _call("POST", f"/datasets/{dataset_id}/rows/build", headers, body=body)
    run_id = (build.get("run") or {}).get("id") if isinstance(build, dict) else None
    if not run_id:
        raise RiveterApiError(f"rows/build returned no run: {str(build)[:200]}")
    return _run_summary(_wait_for_run(run_id, headers))


def _rows_landed(build: dict[str, Any]) -> int | None:
    added = build.get("added_rows")
    return added if isinstance(added, int) else None


def _build_until_k(dataset_id: str, headers: dict[str, str], *, k: int, prompt: str, max_rounds: int) -> list[dict[str, Any]]:
    """Build K rows, then extend for the remainder while a build lands rows and
    the dataset is still short of K. Every build's summary is returned."""
    builds: list[dict[str, Any]] = []
    landed = 0
    for round_index in range(max_rounds):
        build = _build_rows(dataset_id, headers, count=k - landed, prompt=prompt if round_index == 0 else None)
        builds.append(build)
        added = _rows_landed(build)
        if added is None or added == 0 or build["run_status"] != "success":
            break
        landed += added
        if landed >= k:
            break
    return builds


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    env = require_env("RIVETER_API_KEY")
    headers = {"Authorization": f"Bearer {env['RIVETER_API_KEY']}"}
    max_rounds = int(config.get("max_build_rounds") or DEFAULT_MAX_BUILD_ROUNDS)
    prompt = lookalike_prompt(seed)
    out_config: dict[str, Any] = {**config, "prompt": prompt, "count": k, "input_columns": INPUT_COLUMNS, "builds": []}
    started = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        dataset_id = _create_seed_dataset(seed, headers)
        out_config["dataset_id"] = dataset_id
        builds = _build_until_k(dataset_id, headers, k=k, prompt=prompt, max_rounds=max_rounds)
        out_config["builds"] = builds
        out_config["credits_used"] = _credits_used(builds)
        rows = _fetch_rows(dataset_id, headers)
    except RiveterApiError as exc:
        return _result(seed, config, [], elapsed_ms(), k=k, out_config=out_config, error=str(exc))

    latency_ms = elapsed_ms()
    candidates = _without_seed(candidates_from_rows(rows), seed)
    candidates, duplicates_removed = _without_duplicates(candidates)
    error = None
    if not candidates and builds[-1]["run_status"] == "stopped":
        error = f"run stopped ({builds[-1]['stopped_reason']}) before any row landed"
    return _result(
        seed, config, take_top(candidates, k), latency_ms, k=k, out_config=out_config,
        cost_usd=_cost_usd(builds), error=error, duplicates_removed=duplicates_removed,
    )
