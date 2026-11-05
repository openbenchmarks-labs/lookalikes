"""Model-only baseline — GPT-5.6 Sol (Azure NEXTGEN OpenAI-v1 deployment).

Same pipeline as the Claude Opus 5.5 baseline (`model_baseline.py`): one call
generates the ranked lookalikes, then each candidate's description is written
strictly from a scrape of the domain the model identified. Only the generator
transport differs — chat completions against the Azure OpenAI-v1 deployment
that also serves the gpt-6-luna judge.

GPT-5.6 is a reasoning model: reasoning tokens count against
max_completion_tokens, so each call gets headroom on top of the pipeline's
nominal budget, and reasoning effort defaults to low (override with
LOOKALIKE_SOL_REASONING_EFFORT).
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

from ..common import RunResult, Seed, http_request, require_env
from . import model_baseline as base

VENDOR_SLUG = "gpt-5-6-sol"
VENDOR_NAME = "GPT-5.6 Sol (model baseline)"
MODEL_ID = "gpt-5.6-sol"

CONFIGS: list[dict[str, Any]] = [
    {"name": "model_scraped_desc"},
]

REASONING_HEADROOM = 4096


def _complete(system: str, user: str, max_tokens: int, usage_totals: dict[str, int]) -> str:
    env = require_env(
        "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY", "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL"
    )
    url = env["AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL"].rstrip("/") + "/chat/completions"
    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens + REASONING_HEADROOM,
        "reasoning_effort": os.environ.get("LOOKALIKE_SOL_REASONING_EFFORT", "low"),
    }
    headers = {"Authorization": f"Bearer {env['AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY']}"}

    last: tuple[int, Any] = (0, None)
    for attempt in range(base.BEDROCK_RETRIES):
        status, payload, _elapsed = http_request("POST", url, headers=headers, body=body)
        if status == 200 and isinstance(payload, dict):
            usage = payload.get("usage") or {}
            with base._USAGE_LOCK:
                usage_totals["inputTokens"] = usage_totals.get("inputTokens", 0) + int(
                    usage.get("prompt_tokens") or 0
                )
                usage_totals["outputTokens"] = usage_totals.get("outputTokens", 0) + int(
                    usage.get("completion_tokens") or 0
                )
                usage_totals["totalTokens"] = usage_totals.get("totalTokens", 0) + int(
                    usage.get("total_tokens") or 0
                )
            choices = payload.get("choices") or []
            text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
            if text.strip():
                return text
            raise RuntimeError(
                "sol returned 200 with no text content "
                f"(finish_reason={choices[0].get('finish_reason') if choices else None})"
            )
        last = (status, payload)
        if status not in base.RETRYABLE_STATUSES:
            break
        time.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"sol chat completion failed (HTTP {last[0]}): {str(last[1])[:300]}")


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return base.run_pipeline(
        seed, k, config, slug=VENDOR_SLUG, model_label=MODEL_ID, complete=_complete
    )
