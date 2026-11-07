"""Model-only baseline — Kimi K3 (Fireworks OpenAI-compatible endpoint).

Same pipeline as the Claude Opus 5.5 baseline (`model_baseline.py`): one call
generates the ranked lookalikes, then each candidate's description is written
strictly from a scrape of the domain the model identified. Only the generator
transport differs — chat completions against Fireworks, the same serving path
the kimi-k3 judge endpoint uses (FIREWORKS_BASE_URL / FIREWORKS_KIMI_K3_MODEL
overrides apply here too).

Kimi K3 reasons before answering and its reasoning tokens count against
max_tokens, so each call gets headroom on top of the pipeline's nominal budget.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from ..common import RunResult, Seed, record_vendor_call, require_env
from . import model_baseline as base

VENDOR_SLUG = "kimi-k3"
VENDOR_NAME = "Kimi K3 (model baseline)"
DEFAULT_MODEL = "accounts/fireworks/models/kimi-k3"
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

CONFIGS: list[dict[str, Any]] = [
    {"name": "model_scraped_desc"},
]

REASONING_HEADROOM = 8192
# Max silence between streamed chunks before the call is treated as stalled.
STREAM_IDLE_TIMEOUT_SEC = 120


def _model_id() -> str:
    return os.environ.get("FIREWORKS_KIMI_K3_MODEL") or DEFAULT_MODEL


def _stream_chat(url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, Any]:
    """POST a streaming chat completion and reassemble it into the shape of a
    non-streaming response. Streaming keeps a long generation (reasoning + a
    110-row list) from hitting a whole-response timeout: the socket timeout
    applies per read, so a slow but steady stream never trips it. The
    reassembled response is added to the raw audit trail (auth never stored)."""
    req_body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        url,
        data=json.dumps(req_body).encode("utf-8"),
        method="POST",
        headers={**headers, "Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.monotonic()
    content: list[str] = []
    reasoning_chars = 0
    finish_reason = None
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=STREAM_IDLE_TIMEOUT_SEC) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_chars += len(delta["reasoning_content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            err: Any = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            err = str(exc)
        record_vendor_call(
            method="POST", url=url, request_body=req_body, response_status=status,
            response_body=err, elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return status, err
    payload = {
        "streamed": True,
        "choices": [{"message": {"content": "".join(content)}, "finish_reason": finish_reason}],
        "usage": usage,
        "reasoning_chars": reasoning_chars,
    }
    record_vendor_call(
        method="POST", url=url, request_body=req_body, response_status=status,
        response_body=payload, elapsed_ms=int((time.monotonic() - started) * 1000),
    )
    return status, payload


def _complete(system: str, user: str, max_tokens: int, usage_totals: dict[str, int]) -> str:
    env = require_env("FIREWORKS_API_KEY")
    url = (os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = {
        "model": _model_id(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens + REASONING_HEADROOM,
    }
    headers = {"Authorization": f"Bearer {env['FIREWORKS_API_KEY']}"}

    last: tuple[int, Any] = (0, None)
    for attempt in range(base.BEDROCK_RETRIES):
        try:
            status, payload = _stream_chat(url, headers, body)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            last = (0, f"{type(exc).__name__}: {exc}")
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
            continue
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
                "kimi returned 200 with no text content "
                f"(finish_reason={choices[0].get('finish_reason') if choices else None})"
            )
        last = (status, payload)
        if status not in base.RETRYABLE_STATUSES:
            break
        time.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"kimi chat completion failed (HTTP {last[0]}): {str(last[1])[:300]}")


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return base.run_pipeline(
        seed, k, config, slug=VENDOR_SLUG, model_label=_model_id(), complete=_complete
    )
