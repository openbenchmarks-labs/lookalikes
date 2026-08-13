"""LLM-as-judge for the lookalike benchmark.

Uses the OpenAI Python SDK with either the Azure-hosted deployment or the
direct OpenAI API, depending on the selected judge model.

  env required:
    AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY
    AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL   # e.g. https://<…>.azure.com/openai/v1

Model can be overridden via `LOOKALIKE_JUDGE_MODEL` env or `Judge(model=…)`
at the call site. Default is `gpt-5.4-mini` — matches the model used
elsewhere in the OpenFunnel stack so the bench is comparable to
production calls.

`mock=True` keeps a deterministic offline path so the orchestrator
end-to-end pipeline can be exercised without burning Azure tokens.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import os
import threading
import time
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Any, Iterator

from openai import AzureOpenAI, OpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel, Field

from .common import (
    Candidate,
    JudgedCandidate,
    JudgedRun,
    JudgeVote,
    RawJudgeCall,
    RunResult,
    Seed,
    load_dotenv,
    now_iso,
)

ROOT = Path(__file__).resolve().parents[2]


# Same pattern as common._HTTP_TRACE — when a context manager is active,
# every judge call appends a RawJudgeCall to this buffer. The orchestrator
# wraps each per-candidate batch so the audit trail lines up with the
# corresponding vendor call.
_JUDGE_TRACE: ContextVar[list[RawJudgeCall] | None] = ContextVar(
    "lookalike_judge_trace", default=None
)


@contextlib.contextmanager
def capture_judge_calls() -> Iterator[list[RawJudgeCall]]:
    """Capture every Judge.score_candidate() call inside the block. Returns
    the list buffer; the caller drains it after exit."""
    buf: list[RawJudgeCall] = []
    token = _JUDGE_TRACE.set(buf)
    try:
        yield buf
    finally:
        _JUDGE_TRACE.reset(token)

# Match the model used in self-serve-backend so bench results track real prod calls.
DEFAULT_MODEL = "gpt-5.4-mini"
AZURE_KEY_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY"
AZURE_URL_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL"

# Default Azure REST api-version for `azure` transport judges (override per
# endpoint via api_version_env).
DEFAULT_AZURE_API_VERSION = "2024-10-21"

# Max concurrent judge LLM calls per cell (across judges x candidates). The
# OpenAI/Azure sync clients are thread-safe and retry 429s, so this just paces
# fan-out. Override with LOOKALIKE_JUDGE_CONCURRENCY; set 1 to force sequential.
# With cell-level parallelism (N cells at once) the live ceiling is N x this.
_DEFAULT_JUDGE_CONCURRENCY = 3


def _judge_concurrency() -> int:
    raw = os.environ.get("LOOKALIKE_JUDGE_CONCURRENCY")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_JUDGE_CONCURRENCY


# --------------------------------------------------------------------------- #
# Verdict cache — one verdict per (judge, prompt, seed, candidate)             #
# --------------------------------------------------------------------------- #
# A verdict depends only on the judge model, the prompt version, the seed, and
# the candidate company — yet the same (seed, candidate) pair is returned by
# multiple configs of one vendor and by multiple vendors (measured: ~1/3 of all
# judge calls in a full run are repeats). Caching per process deduplicates
# those calls. Deliberate side effect: the same company gets the SAME verdict
# for a seed regardless of which vendor returned it or how much metadata that
# vendor attached — removing a vendor-verbosity bias as well as cost.
# Disable with LOOKALIKE_JUDGE_NO_CACHE=1 (e.g. to measure judge variance).

# key -> (relevant, rationale, anchor_match, capabilities_matched)
_VERDICT_CACHE: dict[tuple[str, str, str, str], tuple[bool, str, bool, tuple[str, ...]]] = {}
_VERDICT_CACHE_LOCK = threading.Lock()


def _cache_enabled() -> bool:
    return os.environ.get("LOOKALIKE_JUDGE_NO_CACHE") != "1"


def _candidate_cache_key(candidate: Candidate) -> str | None:
    """Stable identity for a candidate company: domain when present, else a
    normalized name. None (uncacheable) when neither exists."""
    domain = (candidate.domain or "").strip().lower()
    if domain:
        return domain
    name = (candidate.name or "").strip().lower()
    return f"name:{name}" if name else None


def clear_verdict_cache() -> None:
    with _VERDICT_CACHE_LOCK:
        _VERDICT_CACHE.clear()


@dataclasses.dataclass(frozen=True)
class JudgeEndpoint:
    """How to reach one judge model. Four transports:

      - kind="openai": OpenAI(base_url=<url_env or url_default>, api_key=<key_env>)
        — for OpenAI-v1-compatible endpoints (the NEXTGEN .../openai/v1
        deployment, Fireworks, other gateways). The API `model` is the model id
        (or `deployment` when the serving id differs from the registry key).
      - kind="azure":  AzureOpenAI(azure_endpoint=<endpoint_env>, api_key=<key_env>,
        api_version=...) — standard Azure OpenAI deployments. The API `model` is
        the *deployment name* (`deployment`, defaulting to the model id).
      - kind="direct": OpenAI(api_key=<key_env>) — first-party OpenAI.
      - kind="anthropic": Anthropic(api_key=<key_env>) — first-party Anthropic,
        structured output via output_config.format, prompt caching via a
        cache_control breakpoint on the system block.
    """

    kind: str = "openai"  # openai-v1 endpoint | azure | direct OpenAI | anthropic
    key_env: str = AZURE_KEY_ENV
    # openai transport
    url_env: str = AZURE_URL_ENV
    url_default: str | None = None  # fallback base_url when url_env is unset
    # azure transport
    endpoint_env: str | None = None
    api_version_env: str | None = None
    api_version_default: str = DEFAULT_AZURE_API_VERSION
    deployment: str | None = None  # serving model id when it differs from the registry key

    def api_version(self) -> str:
        if self.api_version_env and os.environ.get(self.api_version_env):
            return os.environ[self.api_version_env]
        return self.api_version_default


# Per-judge transport registry. Adding a judge = one row here + its env vars.
# gpt-5.4-mini is the OpenAI-v1 NEXTGEN deployment; gpt-5.2 is a standard Azure
# deployment reached via AzureOpenAI (azure_endpoint + api_version).
JUDGE_ENDPOINTS: dict[str, JudgeEndpoint] = {
    "gpt-5.4-mini": JudgeEndpoint(kind="openai", url_env=AZURE_URL_ENV, key_env=AZURE_KEY_ENV),
    "gpt-5.2": JudgeEndpoint(
        kind="azure",
        endpoint_env="AZURE_OPENAI_GPT_5_2_DEPLOYMENT_URL",
        key_env="AZURE_OPENAI_GPT_5_2_DEPLOYMENT_KEY",
        api_version_env="AZURE_OPENAI_GPT_5_2_API_VERSION",
        api_version_default="2024-10-21",
        deployment="gpt-5.2",  # set to your Azure deployment name if it differs
    ),
    "gpt-5.1": JudgeEndpoint(
        kind="azure",
        endpoint_env="AZURE_OPENAI_GPT_5_2_DEPLOYMENT_URL",
        key_env="AZURE_OPENAI_GPT_5_2_DEPLOYMENT_KEY",
        api_version_env="AZURE_OPENAI_GPT_5_2_API_VERSION",
        api_version_default="2024-10-21",
        deployment="gpt-5.1",  # set to your Azure deployment name if it differs
    ),
    "gpt-5.6": JudgeEndpoint(kind="direct", key_env="OPENAI_API_KEY"),
    # --- 3-judge cross-lab panel (majority vote; order them cheapest-first so
    # the gated tiebreak seat is the expensive one) ---
    "gpt-5.6-terra": JudgeEndpoint(kind="direct", key_env="OPENAI_API_KEY"),
    "claude-opus-5": JudgeEndpoint(kind="anthropic", key_env="ANTHROPIC_API_KEY"),
    "kimi-k3": JudgeEndpoint(
        kind="openai",
        url_env="FIREWORKS_BASE_URL",
        url_default="https://api.fireworks.ai/inference/v1",
        key_env="FIREWORKS_API_KEY",
        deployment="accounts/fireworks/models/kimi-k3",  # verified live 2026-08-10
    ),
}


def endpoint_for(model: str) -> JudgeEndpoint:
    """Transport config for a judge model; defaults to the NEXTGEN OpenAI-v1
    deployment for unknown ids."""
    ep = JUDGE_ENDPOINTS.get(model, JudgeEndpoint())
    # Env override for the Fireworks serving path — resolved here (after
    # load_dotenv) rather than at import time so .env values apply.
    if model == "kimi-k3":
        override = os.environ.get("FIREWORKS_KIMI_K3_MODEL")
        if override and override != ep.deployment:
            ep = dataclasses.replace(ep, deployment=override)
    return ep


def _strip_v1(url: str) -> str:
    """AzureOpenAI wants the bare resource endpoint (https://<res>.openai.azure.com),
    not the OpenAI-v1 base — tolerate either by trimming a trailing /openai/v1."""
    u = url.rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    return u.rstrip("/")


# One client per distinct connection so judges sharing a deployment share it.
_CLIENT_CACHE: dict[tuple, OpenAI] = {}


def _build_client(ep: JudgeEndpoint, timeout: int) -> OpenAI:
    """Build (or reuse) the right SDK client for an endpoint's transport."""
    if ep.kind == "azure":
        endpoint = os.environ.get(ep.endpoint_env or "")
        key = os.environ.get(ep.key_env)
        if not (endpoint and key):
            raise RuntimeError(
                f"missing {ep.endpoint_env} or {ep.key_env} in env — this Azure judge "
                "needs azure_endpoint + api_key. Pass mock=True for offline runs."
            )
        endpoint = _strip_v1(endpoint)
        version = ep.api_version()
        ck = ("azure", endpoint, key, version)
        client = _CLIENT_CACHE.get(ck)
        if client is None:
            client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=key, api_version=version, timeout=timeout
            )
            _CLIENT_CACHE[ck] = client
        return client

    if ep.kind == "direct":
        key = os.environ.get(ep.key_env)
        if not key:
            raise RuntimeError(
                f"missing {ep.key_env} in env — this direct OpenAI judge needs an API key. "
                "Pass mock=True for offline runs."
            )
        ck = ("direct", key)
        client = _CLIENT_CACHE.get(ck)
        if client is None:
            client = OpenAI(api_key=key, timeout=timeout)
            _CLIENT_CACHE[ck] = client
        return client

    if ep.kind == "anthropic":
        key = os.environ.get(ep.key_env)
        if not key:
            raise RuntimeError(
                f"missing {ep.key_env} in env — this Anthropic judge needs an API key. "
                "Pass mock=True for offline runs."
            )
        import anthropic  # lazy: only needed when an Anthropic judge is used

        ck = ("anthropic", key)
        client = _CLIENT_CACHE.get(ck)
        if client is None:
            client = anthropic.Anthropic(api_key=key, timeout=timeout)
            _CLIENT_CACHE[ck] = client
        return client  # type: ignore[return-value]

    url = os.environ.get(ep.url_env) or ep.url_default
    key = os.environ.get(ep.key_env)
    if not (url and key):
        raise RuntimeError(
            f"missing {ep.url_env} or {ep.key_env} in env — this OpenAI-v1 judge needs "
            "base_url + api_key. Pass mock=True for offline runs."
        )
    ck = ("openai", url, key)
    client = _CLIENT_CACHE.get(ck)
    if client is None:
        client = OpenAI(base_url=url, api_key=key, timeout=timeout)
        _CLIENT_CACHE[ck] = client
    return client


def judge_model_label(models: list[str]) -> str:
    """Canonical `judge_model` string used by both the JSON export and (later)
    the Supabase view. N=1 -> the bare model id (byte-compatible with today);
    N>1 -> 'majority(n=K): a,b,c' with models sorted."""
    uniq = sorted(set(models))
    if len(uniq) <= 1:
        return uniq[0] if uniq else DEFAULT_MODEL
    return f"majority(n={len(uniq)}): " + ",".join(uniq)


# Pydantic response shape — strict JSON, no drift. The OpenAI SDK enforces
# the schema via `response_format` so we get back a parseable object.
class JudgeVerdict(BaseModel):
    """Gate-then-score. Field order is deliberate: the anchor gate is decided
    BEFORE the verdict so the model commits to it rather than rationalizing
    backwards from a conclusion. `relevant` is re-derived in code from
    `anchor_match` + `capabilities_matched` (see `_apply_gate`), so a model
    that answers the two structured fields honestly and then contradicts
    itself on the boolean cannot corrupt a cell."""

    anchor_match: bool = Field(
        description=(
            "HARD GATE, decide this first. True iff CANDIDATE is the same kind of "
            "company as the SEED's ANCHOR — same customer AND same fundamental "
            "category. False for a company serving a different customer, or in a "
            "different category, however many capabilities it shares."
        )
    )
    capabilities_matched: list[str] = Field(
        default_factory=list,
        description=(
            "Which of the SEED's listed CAPABILITIES this candidate also does. "
            "Copy matching strings verbatim from the SEED's list; [] if none. "
            "A candidate need not match all of them."
        ),
    )
    relevant: bool = Field(
        description=(
            "True iff anchor_match is true AND capabilities_matched is non-empty. "
            "Set it consistently with those two fields."
        )
    )
    rationale: str = Field(
        max_length=240,
        description=(
            "One line, ≤ 25 words: name the anchor verdict and the overlap. "
            "No JSON inside."
        ),
    )


# Improved verdict schema (prompt_version="v2"): reason-before-verdict ordering
# (the boolean dimensions are decided first, then `relevant`) to reduce anchoring,
# plus a structured disqualifier + confidence. `relevant`/`rationale` keep the
# same meaning so the pipeline is unchanged; the extra fields persist in the
# audit `parsed_response`. Turned on (and re-baselined) in P5.
class JudgeVerdictV2(BaseModel):
    same_business_model: bool = Field(
        description="True iff CANDIDATE sells essentially the same kind of product/service as SEED."
    )
    buyer_overlap: bool = Field(
        description="True iff a sales rep working the SEED account would also prospect CANDIDATE (overlapping buyer/customer)."
    )
    category_fit_note: str = Field(
        max_length=160,
        description="≤ 1 sentence on how CANDIDATE's category compares to SEED's.",
    )
    disqualifier: str = Field(
        description=(
            "Exactly one of: none | customer | parent | agency_serving_category | "
            "unrelated_industry | duplicate. Use the strongest that applies."
        )
    )
    confidence: str = Field(description="One of: high | medium | low.")
    relevant: bool = Field(
        description="Final verdict: a plausible lookalike of SEED for B2B prospecting."
    )
    rationale: str = Field(
        max_length=240,
        description="One-line reason, ≤ 25 words. No JSON inside.",
    )


JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring company lookalike results for a B2B benchmark.

Definition - a company lookalike API takes a seed company and returns other
companies that resemble it. Resemblance has two halves, and they are judged
DIFFERENTLY:

  ANCHOR       - who the seed sells to, and what it fundamentally is
                 (e.g. "restaurant point-of-sale platform"). This is a HARD
                 requirement. Every lookalike must be the same kind of company
                 for the same kind of customer. No amount of feature overlap
                 substitutes for it.
  CAPABILITIES - what the seed does (e.g. payments, online ordering, payroll,
                 operations). A lookalike matches ANY non-empty subset. It does
                 not need all of them, and matching more is not required to
                 qualify - only to be a closer match.

So: a true lookalike clears the ANCHOR and shares at least one CAPABILITY.

You will be given:
  • A SEED company - the company the lookalike search started from, with its
    ANCHOR and its CAPABILITIES listed separately.
  • A CANDIDATE company - a lookalike returned by a vendor's API.

Decide in this order, and do not work backwards from a conclusion:

STEP 1 - anchor_match (the gate).
  True only if the CANDIDATE is the same kind of company as the ANCHOR, for
  the same kind of customer. Ask: would someone shopping for the SEED consider
  this company to be in the same market?
    - A different customer fails, even in the same product category
      (restaurant POS vs retail-only POS: the customer differs).
    - A different category fails, even for the same customer
      (a restaurant POS vs a restaurant-supply distributor: both sell to
      restaurants, but only one is the same kind of company).
    - Serving the category rather than competing in it fails: agencies,
      consultancies, integrators, resellers.
    - Being a customer of the category fails (an actual restaurant is not a
      lookalike of a restaurant POS platform).
    - Parent/holding companies, the seed itself, and duplicate or parody
      entries fail.
  When you cannot tell what the candidate sells, anchor_match is false.

STEP 2 - capabilities_matched (the overlap).
  List the SEED capabilities this candidate also does, copying the strings
  verbatim from the SEED's list. Include a capability when the candidate
  plausibly offers it as part of its product; leave it out when you have no
  evidence. Matching a subset is normal and expected.

STEP 3 - relevant.
  relevant = anchor_match AND capabilities_matched is non-empty.
  Nothing else changes it.

Rules that apply throughout:
  - Do NOT require comparable company size, headcount, revenue, funding,
    geography, maturity, or platform breadth. A smaller, larger, or
    differently-located company clears the anchor if it is the same kind of
    company for the same kind of customer. Ignore such metadata.
  - A shared generic taxonomy label or keyword is not an anchor match. Neither
    is a similar technology stack - tech co-signals may support a judgement but
    never establish one.
  - Judge the candidate's primary business, not a side feature.
  - When in doubt on the anchor, answer false.
  - Keep the rationale ≤ 25 words: name the anchor verdict and the overlap.

Worked examples (SEED anchor -> candidate -> verdict):
  - "restaurant point-of-sale platform" [payments, online ordering, payroll]
      • a restaurant POS suite with payments and online ordering
        -> anchor true, matched [payments, online ordering], relevant true.
      • a restaurant POS with none of the listed capabilities
        -> anchor true, matched [], relevant FALSE (no overlap).
      • a retail-only POS with payments
        -> anchor FALSE (different customer), relevant false.
      • a general payments processor with no restaurant POS product
        -> anchor FALSE (different category), relevant false.
      • an actual restaurant chain
        -> anchor FALSE (customer of the category), relevant false.
      • an agency that installs restaurant POS systems
        -> anchor FALSE (serves the category), relevant false.
  - "CRM and marketing automation platform for go-to-market teams"
      • a smaller CRM with marketing automation -> anchor true, relevant true.
      • a lead-generation agency using that CRM for clients -> anchor false.
  - "digital bank for consumers"
      • a digital bank in another country -> anchor true (geography is not a
        constraint), relevant true if any capability overlaps.
  - "developer-security platform"
      • a company on a similar cloud stack selling HR software -> anchor false.
"""


JUDGE_SYSTEM_PROMPT_V2 = """You are an evaluator scoring company lookalike results for a B2B sales benchmark. You score one CANDIDATE against one SEED, deciding whether the CANDIDATE is a plausible lookalike of the SEED for B2B sales prospecting — a company a rep working the SEED account would also want to work.

Decide the structured fields first, then the final `relevant` verdict.

RUBRIC — a lookalike must clear BOTH bars:
  1. Same business model — sells essentially the same kind of product/service (set same_business_model).
  2. Overlapping customer / buyer persona — the same buyers would consider both (set buyer_overlap).

Do NOT use company size, headcount, revenue, funding, geography, maturity, or
platform breadth as a relevance requirement. The benchmark does not supply
firmographic constraints; a smaller or larger company can be a valid lookalike.
Ignore such metadata if it appears in the candidate record.

Examples: a smaller CRM/marketing-automation platform is a HubSpot lookalike;
a lead-generation agency using HubSpot is not. An online-store/e-commerce
platform is a Shopify lookalike; a payment processor without a storefront is
not. A ride-hailing marketplace is a Grab lookalike; a traditional car-rental
operator is not. A smaller HRIS/payroll platform is a Rippling lookalike;
device-security software for mobile workers is not. A corporate-card and
spend-management platform is a Brex lookalike; consumer budgeting software is
not. A foundation-model/API company is an OpenAI lookalike; an AI consultancy
is not. An endpoint-security platform is a CrowdStrike lookalike; generic IT
monitoring is not. Construction project-management software is a Procore
lookalike; a construction contractor is not. Field-service management software
is a ServiceTitan lookalike; a plumbing company using such software is not.

DISQUALIFIERS (set `disqualifier` to the strongest that applies; else "none"):
  - customer: the candidate is a customer/client of the seed's category, not a peer.
  - parent: a parent/holding company or a much larger conglomerate.
  - agency_serving_category: an agency/consultancy/integrator that SERVES the category rather than competing in it.
  - unrelated_industry: a different industry/business model.
  - duplicate: the seed itself, or a near-duplicate/parody entry.

CATEGORY GUIDANCE (the seed's category is given):
  - b2b-saas / devtools: match on product category + buyer (e.g. support tooling ≠ generic CRM); ignore horizontal giants unless truly comparable.
  - ecommerce: match DTC infra peers (SMS, subscriptions, retention), not the merchants who buy them.
  - healthtech: match care-model peers (digital MSK, value-based primary care), not hospitals/insurers they sell into.
  - home-services / trades: distinguish vertical SaaS for contractors from the contractors themselves; match like-for-like.
  - real-estate: match operator/manager peers at comparable portfolio scale.

CONSISTENCY:
  - Apply the same bar to every candidate. Do not reward verbose vendor metadata — judge on business reality.
  - Express uncertainty through `confidence` (high|medium|low), not by defaulting to false. Set `relevant` on the merits.
  - `relevant` should be true iff same_business_model AND buyer_overlap AND disqualifier == "none" AND scale is comparable.
  - Keep the rationale ≤ 25 words.
"""


@dataclasses.dataclass
class Judge:
    """One judge model. Its transport (OpenAI-v1 vs AzureOpenAI) and credentials
    are resolved from JUDGE_ENDPOINTS by model id; a panel builds one per model."""

    model: str = DEFAULT_MODEL
    mock: bool = False
    timeout_sec: int = 60
    prompt_version: str = "v1"      # "v1" = original prompt/schema (default); "v2" = improved
    apply_env_model: bool = True    # lone judges honour LOOKALIKE_JUDGE_MODEL; panel members don't
    mock_salt: str = ""             # "" => legacy mock behaviour (byte-identical single-judge)
    mock_threshold: float = 0.6
    _client: Any = dataclasses.field(default=None, init=False, repr=False)
    _send_model: str = dataclasses.field(default="", init=False, repr=False)
    _kind: str = dataclasses.field(default="openai", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mock:
            return
        load_dotenv()
        # Honour env override for the model id (lets ops swap without code).
        if self.apply_env_model:
            env_model = os.environ.get("LOOKALIKE_JUDGE_MODEL")
            if env_model:
                self.model = env_model
        ep = endpoint_for(self.model)
        if ep.kind in ("direct", "anthropic"):
            load_dotenv(ROOT / ".env.local")
        self._client = _build_client(ep, self.timeout_sec)
        self._kind = ep.kind
        # Azure routes by deployment name; gateways may serve under a different
        # model id (e.g. Fireworks account paths); default is the registry key.
        self._send_model = ep.deployment or self.model

    def label(self) -> str:
        return "mock-judge" if self.mock else self.model

    def _schema(self) -> type[BaseModel]:
        return JudgeVerdictV2 if self.prompt_version == "v2" else JudgeVerdict

    def _cache_key(self, seed: Seed, candidate: Candidate) -> tuple[str, str, str, str] | None:
        ck = _candidate_cache_key(candidate)
        if ck is None:
            return None
        return (self.model, self.prompt_version, seed.seed_slug, ck)

    def score_candidate(self, seed: Seed, candidate: Candidate) -> JudgedCandidate:
        if self.mock:
            judged = _mock_score(seed, candidate, self.mock_salt, self.mock_threshold)
            _emit_judge_trace(
                model=self.label(),
                messages=[
                    {"role": "system", "content": "<mock judge: no LLM call>"},
                    {"role": "user", "content": _build_messages(seed, candidate, self.prompt_version)[1]["content"]},
                ],
                raw_response=json.dumps(
                    {"relevant": judged.relevant, "rationale": judged.rationale}
                ),
                parsed={"relevant": judged.relevant, "rationale": judged.rationale},
                elapsed_ms=0,
                for_candidate_rank=candidate.rank,
            )
            return judged

        assert self._client is not None, "client not initialized"
        schema = self._schema()
        messages = _build_messages(seed, candidate, self.prompt_version)

        # Verdict cache — the same (judge, prompt, seed, candidate) pair is
        # returned by multiple configs and vendors; judge it once.
        key = self._cache_key(seed, candidate) if _cache_enabled() else None
        if key is not None:
            with _VERDICT_CACHE_LOCK:
                hit = _VERDICT_CACHE.get(key)
            if hit is not None:
                relevant, rationale, anchor_match, caps = hit
                payload = {
                    "anchor_match": anchor_match,
                    "capabilities_matched": caps,
                    "relevant": relevant,
                    "rationale": rationale,
                }
                _emit_judge_trace(
                    model=self.model,
                    messages=messages,
                    raw_response=json.dumps(payload),
                    parsed=payload,
                    elapsed_ms=0,
                    for_candidate_rank=candidate.rank,
                    cached=True,
                )
                return JudgedCandidate(
                    candidate=candidate, relevant=relevant, rationale=rationale,
                    anchor_match=anchor_match, capabilities_matched=list(caps),
                )

        start = time.monotonic()
        try:
            if self._kind == "anthropic":
                raw = self._anthropic_complete(messages, schema)
            else:
                request: dict[str, Any] = {
                    "model": self._send_model,
                    "messages": messages,
                    "response_format": type_to_response_format_param(schema),
                }
                if endpoint_for(self.model).kind == "direct":
                    request["reasoning_effort"] = os.environ.get(
                        "OPENAI_JUDGE_REASONING_EFFORT", "high"
                    )
                completion = self._client.chat.completions.create(**request)
                raw = completion.choices[0].message.content or ""
            elapsed_ms = int((time.monotonic() - start) * 1000)
            verdict = _validate_verdict(schema, raw)
            _emit_judge_trace(
                model=self.model,
                messages=messages,
                raw_response=raw,
                parsed=verdict.model_dump(),  # v1 -> {relevant, rationale}; v2 -> full structured verdict
                elapsed_ms=elapsed_ms,
                for_candidate_rank=candidate.rank,
            )
            rationale = verdict.rationale.strip()[:240]
            # The anchor gate is enforced here, not taken on the model's word.
            relevant, caps = _apply_gate(verdict, seed)
            anchor_match = bool(getattr(verdict, "anchor_match", True))
            if key is not None:
                with _VERDICT_CACHE_LOCK:
                    _VERDICT_CACHE[key] = (relevant, rationale, anchor_match, tuple(caps))
            return JudgedCandidate(
                candidate=candidate,
                relevant=relevant,
                rationale=rationale,
                anchor_match=anchor_match,
                capabilities_matched=caps,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.monotonic() - start) * 1000)
            _emit_judge_trace(
                model=self.model,
                messages=messages,
                raw_response="",
                parsed=None,
                elapsed_ms=elapsed_ms,
                for_candidate_rank=candidate.rank,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _anthropic_complete(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        """One structured-output verdict via the Anthropic API. The system
        block (rubric + SEED) carries a cache_control breakpoint so every
        candidate after the first for a seed reads the prefix at cache rates.
        Judged at low effort by default — a pairwise relevance call doesn't
        need deep thinking (override with ANTHROPIC_JUDGE_EFFORT)."""
        resp = self._client.messages.create(
            model=self._send_model,
            max_tokens=int(os.environ.get("ANTHROPIC_JUDGE_MAX_TOKENS", "4096")),
            system=[{
                "type": "text",
                "text": messages[0]["content"],
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": messages[1]["content"]}],
            output_config={
                "format": {"type": "json_schema", "schema": _schema_for_anthropic(schema)},
                "effort": os.environ.get("ANTHROPIC_JUDGE_EFFORT", "low"),
            },
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError(f"anthropic judge refusal: {getattr(resp, 'stop_details', None)}")
        return next(b.text for b in resp.content if b.type == "text")

    def score_run(self, seed: Seed, run: RunResult) -> JudgedRun:
        judged: list[JudgedCandidate] = []
        for c in run.candidates:
            try:
                judged.append(self.score_candidate(seed, c))
            except Exception as exc:  # noqa: BLE001
                # A judge failure on one candidate shouldn't poison the
                # whole cell — record it as "not relevant" with the
                # error so the audit trail keeps the signal.
                judged.append(
                    JudgedCandidate(
                        candidate=c,
                        relevant=False,
                        rationale=f"judge error: {exc}",
                    )
                )
        return JudgedRun(
            run=run, judged=judged, judge_model=self.label(), judged_at=now_iso()
        )


def _emit_judge_trace(
    *,
    model: str,
    messages: list[dict[str, str]],
    raw_response: str,
    parsed: dict[str, Any] | None,
    elapsed_ms: int,
    for_candidate_rank: int | None,
    error: str | None = None,
    cached: bool = False,
) -> None:
    """Append a RawJudgeCall to the active capture buffer, if any. No-op
    when no `capture_judge_calls()` context is active."""
    buf = _JUDGE_TRACE.get()
    if buf is None:
        return
    buf.append(
        RawJudgeCall(
            model=model,
            messages=messages,
            raw_response=raw_response,
            parsed_response=parsed,
            elapsed_ms=elapsed_ms,
            for_candidate_rank=for_candidate_rank,
            error=error,
            cached=cached,
        )
    )


# --------------------------------------------------------------------------- #
# Prompt helpers                                                              #
# --------------------------------------------------------------------------- #


def _schema_for_anthropic(schema: type[BaseModel]) -> dict[str, Any]:
    """Anthropic structured outputs reject string length constraints
    (minLength/maxLength); strip them — they're re-checked client-side in
    _validate_verdict — and pin additionalProperties/required as required."""
    import copy

    s = copy.deepcopy(schema.model_json_schema())

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            o.pop("maxLength", None)
            o.pop("minLength", None)
            if o.get("type") == "object" and "properties" in o:
                o["additionalProperties"] = False
                o["required"] = list(o["properties"].keys())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(s)
    return s


def _validate_verdict(schema: type[BaseModel], raw: str) -> BaseModel:
    """Parse a verdict, tolerating over-length string fields from transports
    whose schema enforcement can't carry length constraints (they're truncated
    to the field's declared max_length instead of failing the candidate)."""
    try:
        return schema.model_validate_json(raw)
    except Exception:
        data = json.loads(raw)
        for name, field in schema.model_fields.items():
            max_len = None
            for meta in field.metadata:
                max_len = getattr(meta, "max_length", None) or max_len
            if max_len and isinstance(data.get(name), str):
                data[name] = data[name][:max_len]
        return schema.model_validate(data)


def _build_messages(seed: Seed, candidate: Candidate, version: str = "v1") -> list[dict[str, str]]:
    """Build the judge chat messages. The SEED block lives in the SYSTEM
    message, not the user message: the (rubric + seed) prefix is then
    byte-identical across every candidate judged for that seed, so provider
    prompt caching (OpenAI auto-caching at a >=1024-token prefix, Anthropic
    cache_control, Moonshot cache-hit pricing) serves it at cache rates.
    Only the per-candidate block varies per request."""
    seed_block = _format_seed_block(seed)
    cand_extras = _format_extra(candidate.extra)
    cand_block = _format_company_block(
        "CANDIDATE",
        candidate.name,
        candidate.domain,
        candidate.description,
        extras=cand_extras,
    )
    base = JUDGE_SYSTEM_PROMPT_V2 if version == "v2" else JUDGE_SYSTEM_PROMPT
    if version == "v2":
        # v2 adds the seed category (the rubric is category-aware) and flags the
        # candidate extras as unverified to blunt verbosity bias.
        seed_block = f"{seed_block}\n  category: {seed.category}"
        note = (
            "\n\n(Candidate extras are unverified vendor metadata — do not "
            "over-weight them; judge on business reality.)"
            if cand_extras
            else ""
        )
        return [
            {"role": "system", "content": f"{base}\n{seed_block}"},
            {"role": "user", "content": f"{cand_block}{note}"},
        ]
    return [
        {"role": "system", "content": f"{base}\n{seed_block}"},
        {"role": "user", "content": cand_block},
    ]


def _format_seed_block(seed: Seed) -> str:
    """SEED block carrying the anchor/capabilities split the rubric judges on.
    Falls back to the flat description for seeds not yet migrated, so a
    partially-authored seed set still runs (those cells are judged with an
    empty capability list — see `_apply_gate`)."""
    lines = ["SEED:", f"  name: {seed.seed_name}"]
    if seed.seed_domain:
        lines.append(f"  domain: {seed.seed_domain}")
    anchor = (seed.anchor or "").strip()
    if anchor:
        lines.append(f"  ANCHOR (hard requirement): {anchor}")
    elif seed.description:
        lines.append(f"  ANCHOR (hard requirement): {seed.description}")
    if seed.capabilities:
        lines.append("  CAPABILITIES (match any subset):")
        lines.extend(f"    - {c}" for c in seed.capabilities)
    else:
        lines.append(
            "  CAPABILITIES (match any subset): none listed — judge on the "
            "anchor alone and return capabilities_matched: []"
        )
    return "\n".join(lines)


def _apply_gate(verdict: BaseModel, seed: Seed) -> tuple[bool, list[str]]:
    """Re-derive `relevant` from the structured fields instead of trusting the
    model's boolean, so the anchor is a real gate rather than a suggestion.

      relevant = anchor_match AND >= 1 capability matched

    Capability names are matched back to the seed's own list (case-insensitive)
    so a model that paraphrases or invents one cannot inflate the overlap.

    Seeds with no authored capabilities are judged on the anchor alone —
    otherwise every candidate for an unmigrated seed would fail the
    non-empty-overlap test and score zero."""
    anchor = bool(getattr(verdict, "anchor_match", True))
    raw = getattr(verdict, "capabilities_matched", None) or []
    by_lower = {c.strip().lower(): c for c in seed.capabilities}
    matched = []
    for m in raw:
        key = str(m).strip().lower()
        if key in by_lower and by_lower[key] not in matched:
            matched.append(by_lower[key])
    if not seed.capabilities:
        return anchor, []
    return (anchor and bool(matched)), matched


def _format_company_block(
    label: str,
    name: str,
    domain: str | None,
    description: str | None,
    extras: str | None = None,
) -> str:
    lines = [f"{label}:", f"  name: {name}"]
    if domain:
        lines.append(f"  domain: {domain}")
    if description:
        lines.append(f"  description: {description}")
    if extras:
        lines.append(f"  extras: {extras}")
    return "\n".join(lines)


def _format_extra(extra: dict[str, Any]) -> str | None:
    if not extra:
        return None
    keep: dict[str, Any] = {}
    for k, v in extra.items():
        if v in (None, "", [], {}):
            continue
        keep[k] = v if not isinstance(v, str) else v[:200]
    if not keep:
        return None
    import json

    return json.dumps(keep, ensure_ascii=False)[:4000]


# --------------------------------------------------------------------------- #
# Mock judge — deterministic, no API key required                             #
# --------------------------------------------------------------------------- #


def _mock_score(
    seed: Seed, candidate: Candidate, salt: str = "", threshold: float = 0.6
) -> JudgedCandidate:
    name = (candidate.name or "").lower().strip()
    domain = (candidate.domain or "").lower().strip()
    seed_key = seed.seed_slug.lower()
    if not name and not domain:
        return JudgedCandidate(
            candidate=candidate, relevant=False, rationale="mock: empty candidate"
        )
    base = f"{seed_key}|{name}|{domain}"
    # salt="" reproduces the original single-judge hash exactly (byte-identical);
    # distinct salts give panel members that sometimes disagree.
    keyed = f"{salt}|{base}" if salt else base
    h = hashlib.sha256(keyed.encode("utf-8")).digest()
    score = int.from_bytes(h[:2], "big") / 65535.0
    relevant = score < threshold
    return JudgedCandidate(
        candidate=candidate,
        relevant=relevant,
        rationale=(
            "mock judge: looks plausibly similar"
            if relevant
            else "mock judge: too different to count"
        ),
    )


# --------------------------------------------------------------------------- #
# Judge panel — N judges over the same candidates, majority vote               #
# --------------------------------------------------------------------------- #


def _aggregate_votes(votes: list[JudgeVote]) -> tuple[bool, str]:
    """Strict-majority aggregation. Even-N ties resolve to False (conservative,
    matching the 'when in doubt, false' bias). N=1 is exact pass-through."""
    n = len(votes)
    yes = sum(1 for v in votes if v.relevant)
    relevant = yes > n / 2
    winning = [v for v in votes if v.relevant == relevant]
    rationale = (winning[0].rationale if winning else (votes[0].rationale if votes else "")).strip()[:240]
    return relevant, rationale


def _mock_panel_params(n: int) -> list[tuple[str, float]]:
    """(salt, threshold) per mock judge. Judge 0 is the legacy default so a single
    mock judge stays byte-identical; later judges vary to create disagreement."""
    presets = [("", 0.6), ("mock-1", 0.55), ("mock-2", 0.65), ("mock-3", 0.7), ("mock-4", 0.5)]
    if n <= len(presets):
        return presets[:n]
    return presets + [(f"mock-{i}", 0.6) for i in range(len(presets), n)]


def resolve_judge_models(cli_judges: str | None = None, cli_single: str | None = None) -> list[str]:
    """Resolve the live judge model list. Priority: --judges (comma list) >
    LOOKALIKE_JUDGE_MODELS env > [--judge-model | LOOKALIKE_JUDGE_MODEL | DEFAULT_MODEL]."""
    if cli_judges:
        toks = [t.strip() for t in cli_judges.split(",") if t.strip()]
        if toks:
            return toks
    env = os.environ.get("LOOKALIKE_JUDGE_MODELS")
    if env:
        toks = [t.strip() for t in env.split(",") if t.strip()]
        if toks:
            return toks
    return [cli_single or os.environ.get("LOOKALIKE_JUDGE_MODEL") or DEFAULT_MODEL]


@dataclasses.dataclass
class JudgePanel:
    """Scores each candidate with N judges and aggregates by majority vote, keeping
    the per-judge breakdown in JudgedCandidate.votes. Same interface as Judge
    (score_run + label) so the orchestrator is agnostic to panel size.

    Gating (odd N >= 3, on by default): later judges are consulted only while
    the outcome is still undecided — e.g. with N=3 the third judge runs only
    when the first two disagree. The aggregated verdict is mathematically
    identical to always running the full panel (a vote is skipped only once a
    majority is forced); only JudgedCandidate.votes may hold fewer entries.
    Disable with gated=False or LOOKALIKE_JUDGE_FULL_PANEL=1 (e.g. to publish
    complete per-judge leaderboards)."""

    models: list[str]
    mock: bool = False
    timeout_sec: int = 60
    prompt_version: str = "v1"
    concurrency: int = 0  # max concurrent judge calls per cell; 0 => env/default
    gated: bool = True    # skip judge calls that can no longer change the majority
    judges: list[Judge] = dataclasses.field(default_factory=list, init=False)
    vote_labels: list[str] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.mock:
            n = max(1, len(self.models))
            self.judges = [
                Judge(mock=True, prompt_version=self.prompt_version, mock_salt=s, mock_threshold=t)
                for (s, t) in _mock_panel_params(n)
            ]
            self.vote_labels = ["mock-judge" if i == 0 else f"mock-judge-{i}" for i in range(n)]
            return
        for m in self.models:
            self.judges.append(
                Judge(
                    model=m,
                    timeout_sec=self.timeout_sec, prompt_version=self.prompt_version,
                    apply_env_model=False,
                )
            )
        self.vote_labels = [j.model for j in self.judges]

    def label(self) -> str:
        if self.mock:
            return "mock-judge"
        return judge_model_label([j.model for j in self.judges])

    def _score_one(self, seed: Seed, ji: int, ci: int, candidate: Candidate) -> tuple[int, int, JudgedCandidate]:
        """Score one (judge, candidate). A judge failure on one candidate is
        recorded as not-relevant with the error, never poisoning the cell."""
        try:
            return ji, ci, self.judges[ji].score_candidate(seed, candidate)
        except Exception as exc:  # noqa: BLE001
            return ji, ci, JudgedCandidate(candidate=candidate, relevant=False, rationale=f"judge error: {exc}")

    def _run_tasks(
        self,
        seed: Seed,
        candidates: list[Candidate],
        tasks: list[tuple[int, int]],
        results: list[list[JudgedCandidate | None]],
    ) -> None:
        """Execute (judge, candidate) tasks, filling results[ji][ci] — filled in
        any order; aggregation is in candidate order, so the JudgedRun is
        deterministic regardless of completion order."""
        if not tasks:
            return
        cap = self.concurrency if self.concurrency > 0 else _judge_concurrency()
        workers = 1 if self.mock else min(cap, len(tasks))
        if workers <= 1:
            # Sequential — mock path (fast, deterministic trace order) and the
            # trivial single-call case.
            for ji, ci in tasks:
                _, _, jc = self._score_one(seed, ji, ci, candidates[ci])
                results[ji][ci] = jc
        else:
            # Live fan-out. Each task runs in a COPY of the current context so the
            # active capture_judge_calls() buffer (a ContextVar) is visible in the
            # worker thread and the audit trace is still captured. The sync OpenAI/
            # AzureOpenAI clients are thread-safe; list.append is atomic under the GIL.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(copy_context().run, self._score_one, seed, ji, ci, candidates[ci])
                    for ji, ci in tasks
                ]
                for fut in concurrent.futures.as_completed(futures):
                    ji, ci, jc = fut.result()
                    results[ji][ci] = jc

    @staticmethod
    def _decided(results: list[list[JudgedCandidate | None]], ci: int, n_j: int) -> bool:
        """True when the cast votes force the majority outcome regardless of how
        every not-yet-consulted judge would vote (tie -> False, as in
        _aggregate_votes)."""
        cast = [results[ji][ci] for ji in range(n_j) if results[ji][ci] is not None]
        yes = sum(1 for r in cast if r.relevant)
        if yes > n_j / 2:
            return True  # relevant=True is locked in
        return yes + (n_j - len(cast)) <= n_j / 2  # even all-yes remainder can't flip it

    def score_run(self, seed: Seed, run: RunResult) -> JudgedRun:
        candidates = run.candidates
        n_j = len(self.judges)
        results: list[list[JudgedCandidate | None]] = [[None] * len(candidates) for _ in range(n_j)]

        gate = (
            self.gated
            and n_j >= 3
            and n_j % 2 == 1
            and os.environ.get("LOOKALIKE_JUDGE_FULL_PANEL") != "1"
        )
        if not gate:
            self._run_tasks(
                seed, candidates,
                [(ji, ci) for ji in range(n_j) for ci in range(len(candidates))],
                results,
            )
        else:
            # Round 1: the minimum majority-sized bench votes on everything.
            majority = n_j // 2 + 1
            self._run_tasks(
                seed, candidates,
                [(ji, ci) for ji in range(majority) for ci in range(len(candidates))],
                results,
            )
            # Later judges only see candidates whose outcome is still open —
            # with N=3, the third judge runs only where the first two disagree.
            for ji in range(majority, n_j):
                open_cis = [ci for ci in range(len(candidates)) if not self._decided(results, ci, n_j)]
                if not open_cis:
                    break
                self._run_tasks(seed, candidates, [(ji, ci) for ci in open_cis], results)

        judged: list[JudgedCandidate] = []
        for ci, c in enumerate(candidates):
            votes = [
                JudgeVote(
                    judge_model=self.vote_labels[ji],
                    relevant=results[ji][ci].relevant,
                    rationale=results[ji][ci].rationale,
                )
                for ji in range(n_j)
                if results[ji][ci] is not None
            ]
            relevant, rationale = _aggregate_votes(votes)
            # Gate fields follow the majority: anchor_match is itself a majority
            # vote, and the capability set is the union across judges that agreed
            # with the winning verdict (a capability any of them evidenced counts).
            cast = [results[ji][ci] for ji in range(n_j) if results[ji][ci] is not None]
            anchors = [r.anchor_match for r in cast if r.anchor_match is not None]
            anchor_match = (sum(1 for a in anchors if a) > len(anchors) / 2) if anchors else None
            caps: list[str] = []
            for r in cast:
                if r.relevant == relevant:
                    for cap in r.capabilities_matched:
                        if cap not in caps:
                            caps.append(cap)
            judged.append(
                JudgedCandidate(
                    candidate=c, relevant=relevant, rationale=rationale, votes=votes,
                    anchor_match=anchor_match, capabilities_matched=caps,
                )
            )
        return JudgedRun(run=run, judged=judged, judge_model=self.label(), judged_at=now_iso())
