"""Offline regression oracle for the spec/generic-runner refactor (P1).

Each data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json fixture records, per
swept config: the literal vendor request(s) AND the `extracted_candidates` the
ORIGINAL per-vendor runner produced. This harness mocks `http_request` to replay
the recorded responses through the NEW generic runner and asserts:
  1. the requests it issues (method, url, body) match the recorded calls, and
  2. the candidates it normalizes match the recorded `extracted_candidates`.

No network, no API keys. Run:
  PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/replay_fixtures.py
Exits non-zero if any attempt mismatches.
"""
from __future__ import annotations

import glob
import json
import os
import sys

# Dummy creds so require_env() passes; http_request is mocked so they're unused.
for _k in (
    "OCEAN_API_KEY", "EXA_API_KEY", "PARALLEL_API_KEY",
    "PREDICT_LEADS_API_KEY", "PREDICT_LEADS_API_TOKEN", "LUSHA_API_KEY",
):
    os.environ.setdefault(_k, "test-key")

from lookalike import generic_runner  # noqa: E402
from lookalike.common import Seed  # noqa: E402
from lookalike.runners import REGISTRY  # noqa: E402

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "lookalike-runs", "lookalike-2026-q2",
)


class Replayer:
    """Stands in for common.http_request, returning recorded responses in order
    and recording the requests it was asked to make."""

    def __init__(self) -> None:
        self.queue: list[tuple[int, object, int]] = []
        self.calls: list[dict[str, object]] = []

    def load(self, vendor_calls: list[dict]) -> None:
        self.queue = [(vc["response_status"], vc["response_body"], vc.get("elapsed_ms", 0)) for vc in vendor_calls]
        self.calls = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=60):
        self.calls.append({"method": method.upper(), "url": url, "request_body": body})
        if not self.queue:
            raise AssertionError(f"runner made an unexpected extra HTTP call: {method} {url}")
        return self.queue.pop(0)


def _cand_dicts(run) -> list[dict]:
    return [
        {"name": c.name, "domain": c.domain, "description": c.description, "rank": c.rank, "extra": c.extra}
        for c in run.candidates
    ]


def _compare_calls(actual: list[dict], expected: list[dict]) -> list[str]:
    diffs: list[str] = []
    if len(actual) != len(expected):
        diffs.append(f"call count: got {len(actual)}, expected {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a["method"] != e["method"]:
            diffs.append(f"call[{i}].method: {a['method']} != {e['method']}")
        if a["url"] != e["url"]:
            diffs.append(f"call[{i}].url:\n      got: {a['url']}\n      exp: {e['url']}")
        if a["request_body"] != e.get("request_body"):
            diffs.append(f"call[{i}].body:\n      got: {json.dumps(a['request_body'])}\n      exp: {json.dumps(e.get('request_body'))}")
    return diffs


def main() -> int:
    replayer = Replayer()
    generic_runner.http_request = replayer        # type: ignore[assignment]

    totals = {"pass": 0, "fail": 0, "skipped": 0}
    per_vendor: dict[str, dict[str, int]] = {}
    failures: list[str] = []

    for vendor in sorted(REGISTRY.keys()):
        runner = REGISTRY[vendor]
        per_vendor[vendor] = {"pass": 0, "fail": 0, "skipped": 0}
        for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*", f"{vendor}.raw.json"))):
            fix = json.load(open(path, encoding="utf-8"))
            si = fix.get("seed_input") or {}
            if "category" not in si:
                continue
            seed = Seed(
                seed_slug=fix["seed_slug"], seed_name=si["seed_name"],
                seed_domain=si.get("seed_domain"), description=si.get("description"),
                category=si["category"],
            )
            k = fix.get("k", 10)
            for ai, attempt in enumerate(fix["attempts"]):
                calls = attempt["vendor_calls"]
                if not calls:  # vendor_crash with no recorded HTTP — can't replay
                    totals["skipped"] += 1
                    per_vendor[vendor]["skipped"] += 1
                    continue
                status = (attempt.get("result") or {}).get("status")
                replayer.load(calls)
                where = f"{vendor}/{seed.seed_slug}#{ai}({attempt['config'].get('name')})"
                try:
                    run = runner.run(seed, k, attempt["config"])
                except Exception as exc:  # noqa: BLE001
                    totals["fail"] += 1
                    per_vendor[vendor]["fail"] += 1
                    failures.append(f"{where}: raised {type(exc).__name__}: {exc}")
                    continue

                diffs = _compare_calls(replayer.calls, calls)
                if status == "ok":
                    got = _cand_dicts(run)
                    exp = attempt["extracted_candidates"]
                    if got != exp:
                        if len(got) != len(exp):
                            diffs.append(f"candidate count: got {len(got)}, expected {len(exp)}")
                        for ci, (g, e) in enumerate(zip(got, exp)):
                            if g != e:
                                for f in ("name", "domain", "description", "rank", "extra"):
                                    if g[f] != e[f]:
                                        diffs.append(f"cand[{ci}].{f}: got {json.dumps(g[f])[:160]} != exp {json.dumps(e[f])[:160]}")
                                break  # first differing candidate is enough signal

                if diffs:
                    totals["fail"] += 1
                    per_vendor[vendor]["fail"] += 1
                    failures.append(f"{where}:\n    " + "\n    ".join(diffs[:6]))
                else:
                    totals["pass"] += 1
                    per_vendor[vendor]["pass"] += 1

    print("=== replay regression: new generic runner vs recorded fixtures ===")
    for vendor, c in per_vendor.items():
        print(f"  {vendor:<14} pass={c['pass']:<3} fail={c['fail']:<3} skipped(crash)={c['skipped']}")
    print(f"  {'TOTAL':<14} pass={totals['pass']:<3} fail={totals['fail']:<3} skipped(crash)={totals['skipped']}")
    if failures:
        print(f"\n{len(failures)} MISMATCH(es):")
        for f in failures[:40]:
            print("  - " + f)
        return 1
    print("\nALL REPLAYED ATTEMPTS MATCH — generic runner is byte-compatible with the old runners.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
