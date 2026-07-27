"""PredictLeads runner — thin binding to specs/predictleads.yaml + generic runner.

Quirks (hooks/predictleads.py): optional paging query, and the JSON:API
data[]↔included[] join that resolves each similarity to its company.
"""
from __future__ import annotations

from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("predictleads")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs
MAX_RESULTS = 25


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    result = run_from_spec(_SPEC, seed, k, config)
    # The published lookalike endpoint's default one-page result depth is 25.
    # It may occasionally return a couple more records, but it cannot support
    # a comparable P@100 evaluation without changing the configured endpoint.
    result.max_results = MAX_RESULTS
    return result
