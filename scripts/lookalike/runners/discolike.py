"""Discolike runner — thin binding to its domain-lookalike spec."""
from __future__ import annotations

import time
from typing import Any

from ..common import RunResult, Seed
from ..generic_runner import run_from_spec
from ..spec_loader import load_spec

_SPEC = load_spec("discolike")
VENDOR_SLUG = _SPEC.slug
VENDOR_NAME = _SPEC.name
CONFIGS: list[dict[str, Any]] = _SPEC.configs

# /v1/discover is rate-limited per API key over a moving 60s window (429) and
# answers 503 "please retry" while vector search is briefly unavailable; both
# wait and retry instead of failing the cell.
RETRYABLE_PREFIXES = ("HTTP 429", "HTTP 503")
MAX_ATTEMPTS = 6
BACKOFF_SEC = 20


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    for _ in range(MAX_ATTEMPTS - 1):
        result = run_from_spec(_SPEC, seed, k, config)
        if not (result.error or "").startswith(RETRYABLE_PREFIXES):
            return result
        time.sleep(BACKOFF_SEC)
    return run_from_spec(_SPEC, seed, k, config)
