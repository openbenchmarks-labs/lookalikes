"""Lusha runner — thin binding to specs/lusha.yaml + generic runner.

NOT in REGISTRY (the lookalike endpoint needs 5-100 seeds per request, which
doesn't fit the one-seed-per-cell unit). Kept importable + spec-bound for a
future multi-seed cell design. Fully declarative; no hooks.
"""
from __future__ import annotations

from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("lusha")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_from_spec(_SPEC, seed, k, config)
