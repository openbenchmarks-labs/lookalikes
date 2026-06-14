"""LLM-as-judge for the lookalike benchmark.

  env required:
    AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY
    AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL   # e.g. https://<…>.azure.com/openai/v1

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
import time
from contextvars import ContextVar, copy_context
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


@dataclasses.dataclass(frozen=True)
class JudgeEndpoint:
    """How to reach one judge model. Two transports:

      - kind="openai": OpenAI(base_url=<url_env>, api_key=<key_env>) — for
        OpenAI-v1-compatible endpoints (e.g. the NEXTGEN .../openai/v1 deployment).
        The API `model` is the model id.
      - kind="azure":  AzureOpenAI(azure_endpoint=<endpoint_env>, api_key=<key_env>,
        api_version=...) — standard Azure OpenAI deployments. The API `model` is
        the *deployment name* (`deployment`, defaulting to the model id).
    """

    kind: str = "openai"
    key_env: str = AZURE_KEY_ENV
    # openai transport
    url_env: str = AZURE_URL_ENV
    # azure transport
    endpoint_env: str | None = None
    api_version_env: str | None = None
    api_version_default: str = DEFAULT_AZURE_API_VERSION
    deployment: str | None = None  # azure deployment name; defaults to the model id

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
}


def endpoint_for(model: str) -> JudgeEndpoint:
    """Transport config for a judge model; defaults to the NEXTGEN OpenAI-v1
    deployment for unknown ids."""
    return JUDGE_ENDPOINTS.get(model, JudgeEndpoint())


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

    url = os.environ.get(ep.url_env)
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
    relevant: bool = Field(
        description="True iff CANDIDATE is a plausible lookalike of SEED for B2B sales prospecting."
    )
    rationale: str = Field(
        max_length=240,
        description="One-line reason, ≤ 25 words. No JSON inside.",
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


JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring company lookalike results for a B2B sales benchmark.

You will be given:
  • A SEED company — the company a sales operator is targeting.
  • A CANDIDATE company — a lookalike returned by a vendor's API.

Decide whether the CANDIDATE is a plausible lookalike of the SEED for
B2B sales prospecting. A plausible lookalike is a company a sales rep
working the SEED account would also want to work — same kind of
business model, comparable customer base, overlapping product category
or buyer persona.

Rules:
  - relevant = true only when the candidate is clearly the same kind of
    company (same model + comparable scale + overlapping customer).
  - relevant = false for unrelated industries, parent companies,
    customers/clients of the seed, agencies/consultancies that serve the
    category (rather than competing), and parody/duplicate entries.
  - When in doubt, label false.
  - Keep the rationale ≤ 25 words.
"""


JUDGE_SYSTEM_PROMPT_V2 = """You are an evaluator scoring company lookalike results for a B2B sales benchmark. You score one CANDIDATE against one SEED, deciding whether the CANDIDATE is a plausible lookalike of the SEED for B2B sales prospecting — a company a rep working the SEED account would also want to work.

Decide the structured fields first, then the final `relevant` verdict.

RUBRIC — a lookalike must clear ALL THREE bars:
  1. Same business model — sells essentially the same kind of product/service (set same_business_model).
  2. Comparable scale — roughly the same kind/size of company, not a giant vs. a solo shop.
  3. Overlapping customer / buyer persona — the same buyers would consider both (set buyer_overlap).

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
    _client: OpenAI | None = dataclasses.field(default=None, init=False, repr=False)
    _send_model: str = dataclasses.field(default="", init=False, repr=False)

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
        self._client = _build_client(ep, self.timeout_sec)
        # Azure routes by deployment name; OpenAI-v1 routes by model id.
        self._send_model = (ep.deployment or self.model) if ep.kind == "azure" else self.model

    def label(self) -> str:
        return "mock-judge" if self.mock else self.model

    def _system_prompt(self) -> str:
        return JUDGE_SYSTEM_PROMPT_V2 if self.prompt_version == "v2" else JUDGE_SYSTEM_PROMPT

    def _schema(self) -> type[BaseModel]:
        return JudgeVerdictV2 if self.prompt_version == "v2" else JudgeVerdict

    def score_candidate(self, seed: Seed, candidate: Candidate) -> JudgedCandidate:
        if self.mock:
            judged = _mock_score(seed, candidate, self.mock_salt, self.mock_threshold)
            _emit_judge_trace(
                model=self.label(),
                messages=[
                    {"role": "system", "content": "<mock judge: no LLM call>"},
                    {"role": "user", "content": _build_user_prompt(seed, candidate, self.prompt_version)},
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
        user_content = _build_user_prompt(seed, candidate, self.prompt_version)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_content},
        ]
        start = time.monotonic()
        try:
            completion = self._client.chat.completions.create(
                model=self._send_model,
                messages=messages,
                response_format=type_to_response_format_param(schema),
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            raw = completion.choices[0].message.content or ""
            verdict = schema.model_validate_json(raw)
            _emit_judge_trace(
                model=self.model,
                messages=messages,
                raw_response=raw,
                parsed=verdict.model_dump(),  # v1 -> {relevant, rationale}; v2 -> full structured verdict
                elapsed_ms=elapsed_ms,
                for_candidate_rank=candidate.rank,
            )
            return JudgedCandidate(
                candidate=candidate,
                relevant=verdict.relevant,
                rationale=verdict.rationale.strip()[:240],
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
        )
    )


# --------------------------------------------------------------------------- #
# Prompt helpers                                                              #
# --------------------------------------------------------------------------- #


def _build_user_prompt(seed: Seed, candidate: Candidate, version: str = "v1") -> str:
    seed_block = _format_company_block("SEED", seed.seed_name, seed.seed_domain, seed.description)
    cand_extras = _format_extra(candidate.extra)
    cand_block = _format_company_block(
        "CANDIDATE",
        candidate.name,
        candidate.domain,
        candidate.description,
        extras=cand_extras,
    )
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
        return f"{seed_block}\n\n{cand_block}{note}"
    return f"{seed_block}\n\n{cand_block}"


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

    return json.dumps(keep, ensure_ascii=False)[:400]


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
    (score_run + label) so the orchestrator is agnostic to panel size."""

    models: list[str]
    mock: bool = False
    timeout_sec: int = 60
    prompt_version: str = "v1"
    concurrency: int = 0  # max concurrent judge calls per cell; 0 => env/default
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

    def score_run(self, seed: Seed, run: RunResult) -> JudgedRun:
        candidates = run.candidates
        n_j = len(self.judges)
        # results[ji][ci] — filled in any order; aggregation below is in candidate
        # order, so the JudgedRun is deterministic regardless of completion order.
        results: list[list[JudgedCandidate | None]] = [[None] * len(candidates) for _ in range(n_j)]
        tasks = [(ji, ci) for ji in range(n_j) for ci in range(len(candidates))]

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

        judged: list[JudgedCandidate] = []
        for ci, c in enumerate(candidates):
            votes = [
                JudgeVote(
                    judge_model=self.vote_labels[ji],
                    relevant=results[ji][ci].relevant,
                    rationale=results[ji][ci].rationale,
                )
                for ji in range(n_j)
            ]
            relevant, rationale = _aggregate_votes(votes)
            judged.append(
                JudgedCandidate(candidate=c, relevant=relevant, rationale=rationale, votes=votes)
            )
        return JudgedRun(run=run, judged=judged, judge_model=self.label(), judged_at=now_iso())
