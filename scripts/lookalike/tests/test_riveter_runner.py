"""Offline tests for the Riveter runner.

No network: `http_request` is replaced by a scripted fake that answers the
four-call sequence (create dataset → build rows → poll run → list rows) and
records every request. Asserts that
  - the dataset is created with the three judge-facing input columns and the
    seed as its only row, and the build receives the seed's verbatim `query`
    with `count` = K,
  - polling continues until the run is terminal, and the dataset is extended
    (a build without a prompt, for the remainder) while a round lands rows and
    the dataset is short of K, up to `max_build_rounds`,
  - the seed row is dropped, rows become candidates in position order with
    normalized domains, aggregator URLs lose the domain, and the generic
    seed-removal + dedupe passes apply,
  - cost is credits_used × $0.025 and the audit config carries the run ids,
  - an API error is returned as an error RunResult (no crash), and a seed
    without an authored query gets a composed prompt.

Run:  PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_riveter_runner.py
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Set outright, not setdefault: a shell that sourced a real .env must not leak into the checks.
os.environ["RIVETER_API_KEY"] = "test-key"
os.environ["RIVETER_API_BASE"] = "https://riveter.test/v1"

import lookalike.common as common  # noqa: E402
from lookalike.common import Seed  # noqa: E402
from lookalike.runners import riveter  # noqa: E402

SEED = Seed(
    "toast", "Toast", "toasttab.com", "Restaurant point-of-sale platform", "hospitality",
    short_description="the restaurant point-of-sale and operations platform",
    anchor="restaurant point-of-sale platform", capabilities=["payments", "online ordering"],
    query="Find companies similar to Toast (toasttab.com): restaurant point-of-sale platforms.",
)
SEED_NO_QUERY = Seed("bare", "Bare Co", "bare.com", "Bare makes widgets", "saas")


def _row(position: int, website: str | None, name: str | None, description: str | None = None) -> dict[str, Any]:
    return {"id": f"row_{position}", "position": position,
            "input": {"Website": website, "Company name": name, "What it does": description}, "cells": {}}


class ScriptedApi:
    """Stands in for common.http_request: records calls, answers from a script
    keyed on (method, path) with a queue per key."""

    def __init__(self, script: dict[tuple[str, str], list[tuple[int, Any]]]) -> None:
        self.script = {key: list(answers) for key, answers in script.items()}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=60):
        path = url.removeprefix("https://riveter.test/v1")
        self.calls.append({"method": method, "path": path, "headers": headers, "body": body, "timeout": timeout})
        answers = self.script.get((method, path))
        if not answers:
            raise AssertionError(f"unscripted call {method} {path}")
        status, payload = answers.pop(0) if len(answers) > 1 else answers[0]
        return status, payload, 5


def _run(script: dict, seed: Seed = SEED, k: int = 100) -> tuple[Any, ScriptedApi]:
    api = ScriptedApi(script)
    riveter.http_request = api  # type: ignore[assignment]
    try:
        return riveter.run(seed, k, riveter.CONFIGS[0]), api
    finally:
        riveter.http_request = common.http_request  # type: ignore[assignment]


def _finished_run(run_id: str, *, requested: int, added: int, credits: float, status: str = "success") -> dict[str, Any]:
    return {"id": run_id, "status": status, "credits_used": credits, "app_url": f"https://app.riveter.test/runs/{run_id}",
            "progress": {"requested_rows": requested, "generated_rows": added + 1, "added_rows": added, "skipped_duplicate_rows": 1}}


# Three build rounds: the first lands 4 rows, the extension 2, the third none.
HAPPY_PATH = {
    ("POST", "/datasets"): [(201, {"id": "ds_1", "name": "Lookalikes: Toast"})],
    ("POST", "/datasets/ds_1/rows/build"): [
        (201, {"id": "ds_1", "run": {"id": "run_1", "status": "enqueued"}}),
        (201, {"id": "ds_1", "run": {"id": "run_2", "status": "enqueued"}}),
        (201, {"id": "ds_1", "run": {"id": "run_3", "status": "enqueued"}}),
    ],
    ("GET", "/runs/run_1?wait=50"): [
        (200, {"id": "run_1", "status": "enqueued", "credits_used": 0}),
        (200, _finished_run("run_1", requested=100, added=4, credits=16.0)),
    ],
    ("GET", "/runs/run_2?wait=50"): [(200, _finished_run("run_2", requested=96, added=2, credits=8.0))],
    ("GET", "/runs/run_3?wait=50"): [(200, _finished_run("run_3", requested=94, added=0, credits=0.0))],
    ("GET", "/datasets/ds_1/rows?limit=1000"): [(200, {"rows": [
        _row(1, "toasttab.com", "Toast", "the seed row"),
        _row(2, "https://www.squareup.com/us/en/restaurants", "Square for Restaurants", "POS for restaurants"),
        _row(3, "https://www.linkedin.com/company/lightspeed", "Lightspeed", None),
        _row(4, "toasttab.com", "Toast POS", "the seed again, under its domain"),
        _row(5, "squareup.com", "Square for Restaurants", "a repeat of row 2"),
        _row(6, None, None, "a row without a name"),
        _row(7, "clover.com", "Clover", "  POS and payments  "),
    ], "next_cursor": None})],
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def test_requests() -> None:
    print("\n[requests]")
    _, api = _run(HAPPY_PATH)
    create = api.calls[0]
    check(create["body"]["input_columns"] == ["Website", "Company name", "What it does"], "dataset has the three judge-facing input columns, Website first (the index)")
    check(create["body"]["rows"] == [{"Website": "toasttab.com", "Company name": "Toast",
                                      "What it does": "the restaurant point-of-sale and operations platform"}], "the seed is the dataset's only row")
    check(create["headers"]["Authorization"] == "Bearer test-key", "bearer auth from RIVETER_API_KEY")
    builds = [c for c in api.calls if c["path"] == "/datasets/ds_1/rows/build"]
    check(len(builds) == 3, "a build, then extensions for the remainder, until a round lands nothing (3 rounds max)")
    check(builds[0]["body"] == {"prompt": SEED.query, "count": 100}, "the first build receives the seed's verbatim query and count = K")
    check(builds[1]["body"] == {"count": 96} and builds[2]["body"] == {"count": 94}, "the extensions carry no prompt and ask for the remainder")
    polls = [c for c in api.calls if c["path"].startswith("/runs/run_1")]
    check(len(polls) == 2, "polling continues until the run is terminal")
    check(all(c["timeout"] > riveter.POLL_WAIT_SEC for c in polls), "the poll's HTTP timeout outlasts the server-side wait")
    check(api.calls[-1]["path"] == "/datasets/ds_1/rows?limit=1000", "rows are read from the dataset once the last run is terminal")
    order = [c["path"].split("?")[0] for c in api.calls]
    check(order.index("/runs/run_1") < order.index("/datasets/ds_1/rows/build", 2), "an extension starts only after the previous run is terminal")


def test_candidates() -> None:
    print("\n[candidates]")
    result, _ = _run(HAPPY_PATH)
    check(result.error is None, "no error on the happy path")
    names = [c.name for c in result.candidates]
    check(names == ["Square for Restaurants", "Lightspeed", "Clover"], f"seed row, seed-domain row, repeat, and nameless row are dropped: {names}")
    check([c.rank for c in result.candidates] == [1, 2, 3], "candidates are ranked in generation order")
    check(result.candidates[0].domain == "squareup.com", "website URLs are normalized to a bare host")
    check(result.candidates[1].domain is None, "a LinkedIn URL does not become the domain")
    check(result.candidates[2].description == "POS and payments", "descriptions are passed to the judge, trimmed")
    check(result.duplicates_removed == 1, "the dedupe pass is counted")
    check(result.requested_k == 100 and result.max_results is None, "Riveter serves the full K")
    check(result.cost_usd == 0.6, "cost is the builds' credits_used summed × $0.025")
    check(result.config["dataset_id"] == "ds_1" and [b["run_id"] for b in result.config["builds"]] == ["run_1", "run_2", "run_3"], "the audit config carries the dataset id and every build's run id")
    check([b["added_rows"] for b in result.config["builds"]] == [4, 2, 0] and result.config["credits_used"] == 24.0, "the audit config carries each build's row counters and the credits total")
    check(result.config["prompt"] == SEED.query, "the audit config carries the prompt sent")


def test_rounds_stop_at_k() -> None:
    print("\n[rounds stop at k]")
    script = {
        **HAPPY_PATH,
        ("GET", "/runs/run_1?wait=50"): [(200, _finished_run("run_1", requested=5, added=5, credits=20.0))],
    }
    result, api = _run(script, k=5)
    check(len([c for c in api.calls if c["path"].endswith("/rows/build")]) == 1, "no extension once K rows have landed")
    check(len(result.candidates) == 3 and result.requested_k == 5, "candidates are capped at K")
    single = {**HAPPY_PATH}
    api_single = ScriptedApi(single)
    riveter.http_request = api_single  # type: ignore[assignment]
    try:
        riveter.run(SEED, 100, {"name": "one_build", "max_build_rounds": 1})
    finally:
        riveter.http_request = common.http_request  # type: ignore[assignment]
    check(len([c for c in api_single.calls if c["path"].endswith("/rows/build")]) == 1, "max_build_rounds: 1 is a single build")


def test_api_error() -> None:
    print("\n[api error]")
    script = {("POST", "/datasets"): [(403, {"error": {"type": "feature_not_enabled", "message": "Datasets are not enabled on this account"}})]}
    result, api = _run(script)
    check(result.error == "HTTP 403: feature_not_enabled: Datasets are not enabled on this account", f"the envelope's error is the RunResult error: {result.error}")
    check(result.candidates == [] and len(api.calls) == 1, "nothing else is called after a refused create")


def test_stopped_without_rows() -> None:
    print("\n[stopped run]")
    script = {
        **HAPPY_PATH,
        ("GET", "/runs/run_1?wait=50"): [(200, {"id": "run_1", "status": "stopped", "stopped_reason": "credit_limit", "credits_used": 0})],
        ("GET", "/datasets/ds_1/rows?limit=1000"): [(200, {"rows": [_row(1, "toasttab.com", "Toast")], "next_cursor": None})],
    }
    result, _ = _run(script)
    check(result.error == "run stopped (credit_limit) before any row landed", "a stopped run with no rows is an error cell")


def test_prompt_fallback() -> None:
    print("\n[prompt fallback]")
    check(riveter.lookalike_prompt(SEED) == SEED.query, "an authored query is sent verbatim")
    check(riveter.lookalike_prompt(SEED_NO_QUERY) == "Find companies similar to Bare Co (bare.com): Bare makes widgets", "a seed without a query gets a composed prompt")


def main() -> int:
    test_requests()
    test_candidates()
    test_rounds_stop_at_k()
    test_api_error()
    test_stopped_without_rows()
    test_prompt_fallback()
    print()
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("all riveter runner checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
