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


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_from_spec(_SPEC, seed, k, config)
