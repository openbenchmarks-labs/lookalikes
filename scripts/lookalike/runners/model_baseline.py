"""Model-only baseline — Claude Opus 5.5 (Amazon Bedrock) plays the vendor.

No lookalike API is called. The pipeline per (seed, config):

  1. One Bedrock Converse call asks the model for the top-K lookalike
     companies (name + registrable domain), given the same input the
     text-query vendors receive: the seed's authored query, falling back to
     name / domain / short description.
  2. For every candidate the model returned, the candidate's description is
     NOT taken from the model's parametric memory: the homepage of the domain
     the model identified is fetched (plain HTTP, no JS rendering) and a
     second Converse call writes a 1-3 sentence description strictly from the
     scraped text. A domain that does not resolve, or returns no usable text,
     leaves the candidate with no description — the judge sees the miss
     honestly, exactly as it would for a vendor that returned a dead domain.

Every Bedrock round-trip goes through `common.http_request`, so the literal
model calls (with the Authorization header redacted) land in the same
`*.raw.json` audit trail as any vendor's HTTP calls. Website scrapes are
recorded compactly via `record_vendor_call` (status + first chars of text)
rather than dumping full HTML into the artifact.

Env:
  AWS_BEARER_TOKEN_BEDROCK   Bedrock API key (bearer token)
  AWS_REGION                 e.g. us-east-1
  LOOKALIKE_MODEL_BASELINE_ID      override model id
                                   (default us.anthropic.claude-opus-5-5)
  LOOKALIKE_MODEL_DESC_CONCURRENCY scrape+describe fan-out per cell (default 4)
"""
from __future__ import annotations

import concurrent.futures
import gzip
import html
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from contextvars import copy_context
from html.parser import HTMLParser
from typing import Any

from ..common import (
    Candidate,
    RunResult,
    Seed,
    http_request,
    record_vendor_call,
    require_env,
    take_top,
)
from ..generic_runner import _without_duplicates, _without_seed

VENDOR_SLUG = "claude-opus-5-5"
VENDOR_NAME = "Claude Opus 5.5 (model baseline)"

CONFIGS: list[dict[str, Any]] = [
    {"name": "model_scraped_desc"},
]

DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5-5"
# Opus 5.5 always thinks, and thinking counts against maxTokens: 8192 truncated
# the candidate list on seeds where the model thought longer.
GEN_MAX_TOKENS = 32000
DESC_MAX_TOKENS = 300
SCRAPE_TIMEOUT_SEC = 15
SCRAPE_MAX_BYTES = 800_000
SCRAPE_TEXT_CHARS = 6_000
OVERFETCH = 10  # ask for k+N so seed-removal/dedupe still leaves k
BEDROCK_RETRIES = 5
RETRYABLE_STATUSES = {429, 500, 503}
# Generating ~110 candidates is a long response; the harness default (60s)
# cuts it off under load.
MODEL_TIMEOUT_SEC = 300
_USAGE_LOCK = threading.Lock()

GEN_SYSTEM = """You are a company lookalike engine. Given a seed company, you return the companies most similar to it: the same kind of company selling to the same kind of buyer, with overlapping capabilities. Order them most similar first.

Rules:
- Only real, currently operating companies you are confident exist.
- Exclude the seed company itself, its parent, and its subsidiaries.
- No duplicates.
- "domain" is the company's bare registrable website domain, like "example.com" — no scheme, no path, no www.
- Return strict JSON, nothing else, in this shape:
  {"candidates": [{"name": "...", "domain": "..."}, ...]}"""

DESC_SYSTEM = """You write a short factual description of what a company does, based ONLY on the website text provided. 1-3 plain sentences: what the company sells, and to whom. Do not use any outside knowledge about the company; if the text does not say what the company does, describe only what it supports. Return the description text only — no preamble, no JSON."""


# --------------------------------------------------------------------------- #
# Bedrock plumbing                                                            #
# --------------------------------------------------------------------------- #


def _model_id() -> str:
    return os.environ.get("LOOKALIKE_MODEL_BASELINE_ID") or DEFAULT_MODEL_ID


def _desc_concurrency() -> int:
    raw = os.environ.get("LOOKALIKE_MODEL_DESC_CONCURRENCY")
    try:
        return max(1, int(raw)) if raw else 4
    except ValueError:
        return 4


def _converse(
    system: str, user: str, max_tokens: int, usage_totals: dict[str, int]
) -> str:
    """One Bedrock Converse call with bearer auth, retried on throttling.
    Goes through common.http_request so the literal request/response (auth
    redacted) is captured into the audit trail. Returns the text output and
    accumulates token usage into `usage_totals`."""
    env = require_env("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")
    url = (
        f"https://bedrock-runtime.{env['AWS_REGION']}.amazonaws.com"
        f"/model/{_model_id()}/converse"
    )
    body = {
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    headers = {"Authorization": f"Bearer {env['AWS_BEARER_TOKEN_BEDROCK']}"}

    last: tuple[int, Any] = (0, None)
    for attempt in range(BEDROCK_RETRIES):
        try:
            status, payload, _elapsed = http_request(
                "POST", url, headers=headers, body=body, timeout=MODEL_TIMEOUT_SEC
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            last = (0, f"{type(exc).__name__}: {exc}")
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
            continue
        if status == 200 and isinstance(payload, dict):
            usage = payload.get("usage") or {}
            with _USAGE_LOCK:  # read-modify-write from describe worker threads
                for key in ("inputTokens", "outputTokens", "totalTokens"):
                    usage_totals[key] = usage_totals.get(key, 0) + int(usage.get(key) or 0)
            blocks = ((payload.get("output") or {}).get("message") or {}).get("content") or []
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            if text.strip():
                return text
            raise RuntimeError("bedrock returned 200 with no text content")
        last = (status, payload)
        if status not in RETRYABLE_STATUSES:
            break
        time.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"bedrock converse failed (HTTP {last[0]}): {str(last[1])[:300]}")


_CANDIDATE_OBJ = re.compile(
    r'\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"domain"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
)


def _parse_candidates_json(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """Parse {"candidates": [...]} out of the model's reply, tolerating code
    fences or stray prose around the JSON object. Returns (items,
    used_fallback). Occasionally the reply is not valid JSON as a whole; the
    fallback then extracts each {"name", "domain"} object in order."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        try:
            obj = json.loads(text[start : end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        return [it for it in obj["candidates"] if isinstance(it, dict) and it.get("name")], False

    items = [
        {"name": json.loads(f'"{n}"'), "domain": json.loads(f'"{d}"')}
        for n, d in _CANDIDATE_OBJ.findall(text)
    ]
    if not items:
        raise RuntimeError(f"model reply has no parseable candidates: {raw[:200]}")
    return items, True


# --------------------------------------------------------------------------- #
# Website scraping (stdlib only, no JS rendering)                             #
# --------------------------------------------------------------------------- #


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        for name, value in attrs:
            # Meta descriptions are often the best one-line summary on a page.
            if tag == "meta" and name == "content" and value:
                self.chunks.append(value)
                break

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data)


def _html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
    except Exception:  # noqa: BLE001 — salvage whatever was parsed
        pass
    text = html.unescape(" ".join(parser.chunks))
    return re.sub(r"\s+", " ", text).strip()


def _fetch_homepage(domain: str) -> tuple[str, str]:
    """GET the homepage of `domain` (https, then http). Returns
    (text, final_url). Raises on total failure."""
    last_exc: Exception | None = None
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}/"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "gzip",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=SCRAPE_TIMEOUT_SEC) as resp:
                raw = resp.read(SCRAPE_MAX_BYTES)
                if resp.headers.get("Content-Encoding") == "gzip":
                    try:
                        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(SCRAPE_MAX_BYTES)
                    except OSError:
                        pass
                charset = resp.headers.get_content_charset() or "utf-8"
                return _html_to_text(raw.decode(charset, errors="replace")), resp.geturl()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"fetch failed: {type(last_exc).__name__}: {last_exc}")


def _describe_candidate(
    cand: Candidate, usage_totals: dict[str, int], complete: Any = None
) -> None:
    """Scrape the candidate's domain and have the model write the description
    from the scraped text. Mutates `cand` in place; never raises — a dead
    domain or a model error leaves description None with the reason in extra."""
    if complete is None:
        complete = _converse
    domain = (cand.domain or "").strip().lower()
    if not domain:
        cand.extra["description_source"] = "no_domain"
        return

    start = time.monotonic()
    try:
        text, final_url = _fetch_homepage(domain)
    except Exception as exc:  # noqa: BLE001
        record_vendor_call(
            method="GET",
            url=f"https://{domain}/",
            request_body=None,
            response_status=0,
            response_body={"error": str(exc)},
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
        cand.extra["description_source"] = "scrape_failed"
        cand.extra["scrape_error"] = str(exc)[:200]
        return

    text = text[:SCRAPE_TEXT_CHARS]
    record_vendor_call(
        method="GET",
        url=final_url,
        request_body=None,
        response_status=200,
        response_body={"text_chars": len(text), "text_preview": text[:1500]},
        elapsed_ms=int((time.monotonic() - start) * 1000),
    )
    if len(text) < 80:
        cand.extra["description_source"] = "scrape_empty"
        return

    try:
        desc = complete(
            DESC_SYSTEM,
            f"Company: {cand.name}\nDomain: {domain}\n\nWebsite text:\n{text}",
            DESC_MAX_TOKENS,
            usage_totals,
        )
    except Exception as exc:  # noqa: BLE001
        cand.extra["description_source"] = "describe_failed"
        cand.extra["describe_error"] = str(exc)[:200]
        return
    cand.description = desc.strip()[:1000]
    cand.extra["description_source"] = "scraped_homepage"


# --------------------------------------------------------------------------- #
# Runner contract                                                             #
# --------------------------------------------------------------------------- #


def _seed_query(seed: Seed) -> str:
    if seed.query:
        return seed.query
    about = seed.short_description or seed.description or ""
    domain = f" ({seed.seed_domain})" if seed.seed_domain else ""
    return f"Companies similar to {seed.seed_name}{domain}: {about}".strip().rstrip(":")


def run_pipeline(
    seed: Seed,
    k: int,
    config: dict[str, Any],
    *,
    slug: str,
    model_label: str,
    complete: Any,
    gen_max_tokens: int = GEN_MAX_TOKENS,
) -> RunResult:
    """The full model-baseline pipeline, generator-agnostic. `complete` is
    (system, user, max_tokens, usage_totals) -> text; each backend module binds
    its own transport and registers under its own slug."""
    usage_totals: dict[str, int] = {}
    started = time.monotonic()

    ask = k + OVERFETCH
    user_msg = (
        f"Seed company lookalike query:\n{_seed_query(seed)}\n\n"
        f"Return exactly {ask} candidates, most similar first."
    )
    raw = complete(GEN_SYSTEM, user_msg, gen_max_tokens, usage_totals)
    items, parse_fallback = _parse_candidates_json(raw)

    candidates = [
        Candidate(
            name=str(it["name"]).strip(),
            domain=(str(it.get("domain") or "").strip().lower() or None),
        )
        for it in items
    ]
    sent_count = len(candidates)
    candidates = _without_seed(candidates, seed)
    candidates, duplicates_removed = _without_duplicates(candidates)
    candidates = take_top(candidates, k)

    # Scrape + describe, fan-out capped per cell. copy_context() propagates the
    # active capture_http_calls() buffer into worker threads so the Bedrock and
    # scrape records still land in this cell's audit trail.
    workers = min(_desc_concurrency(), max(1, len(candidates)))
    if workers == 1:
        for cand in candidates:
            _describe_candidate(cand, usage_totals, complete)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    copy_context().run, _describe_candidate, cand, usage_totals, complete
                )
                for cand in candidates
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    described = sum(
        1 for c in candidates if c.extra.get("description_source") == "scraped_homepage"
    )
    return RunResult(
        seed_slug=seed.seed_slug,
        provider_slug=slug,
        config_name=config["name"],
        config={
            **config,
            "model_id": model_label,
            "requested_from_model": ask,
            "parse_fallback": parse_fallback,
            "described_from_homepage": described,
            "model_usage": usage_totals,
        },
        candidates=candidates,
        latency_ms=int((time.monotonic() - started) * 1000),
        cost_usd=None,  # no vendor list price; token usage is in config.bedrock_usage
        error=None,
        requested_k=k,
        duplicates_removed=duplicates_removed,
        sent_count=sent_count,
    )


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    return run_pipeline(
        seed, k, config, slug=VENDOR_SLUG, model_label=_model_id(), complete=_converse
    )
