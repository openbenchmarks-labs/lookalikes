"""OpenFunnel runner — thin binding to specs/openfunnel.yaml + generic runner.

The request shape, field mapping, and config sweep are declared in the YAML
spec; the two quirks (domain canonicalization preflight + conditional query
param) live in hooks/openfunnel.py. See scripts/lookalike/specs/openfunnel.yaml.
"""
from __future__ import annotations

from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("openfunnel")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_from_spec(_SPEC, seed, k, config)
