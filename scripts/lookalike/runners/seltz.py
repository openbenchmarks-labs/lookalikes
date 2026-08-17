"""Seltz runner — thin binding to specs/seltz.yaml + the generic runner.

Seltz is a web-search API; its `companies` scope returns lookalike companies for
a natural-language query. The query is the seed's own `description`, the scope is
always `companies`, and the Markdown `content` of each result is parsed into a
Candidate by the transform hook (hooks/seltz.py). Seltz can serve 100 results, so the benchmark's K=100 is always returned in full.
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
