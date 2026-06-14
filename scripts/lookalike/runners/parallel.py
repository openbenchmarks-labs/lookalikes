"""Parallel runner — thin binding to specs/parallel.yaml + generic runner.

Quirks (hooks/parallel.py): natural-language objective construction, and
routing aggregator hosts (LinkedIn/Crunchbase/…) out of the company domain.
"""
from __future__ import annotations

from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("parallel")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_from_spec(_SPEC, seed, k, config)
