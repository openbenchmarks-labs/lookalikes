"""Seltz runner — thin binding to specs/seltz.yaml + the generic runner.

Matches https://github.com/openbenchmarks-labs/lookalikes/pull/8: POST /search
with x-api-key, body {query: <anchor for caps>, scope: companies, max_results: k}.
The Markdown `content` of each result is parsed into a Candidate by
hooks/seltz.py. Seltz can serve 100 results, so the benchmark's K=100 is
always requested in full.
"""

from __future__ import annotations

from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("seltz")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_from_spec(_SPEC, seed, k, config)
