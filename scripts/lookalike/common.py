"""Shared dataclasses, env loading, snapshot I/O, and HTTP helpers for the
lookalike benchmark runners.

Why a single common module: every vendor runner has the same shape — take
a seed, hit an API, return a list of candidate companies. Keeping the
data classes and snapshot writer here means each runner stays a single,
short file focused on the vendor's quirks.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "data" / "latest-lookalike.json"
RUNS_DIR = ROOT / "data" / "lookalike-runs"
ENV_FILE = ROOT / ".env"

DEFAULT_K = 100
PRECISION_CUTOFFS = (10, 50, 100)
HTTP_TIMEOUT_SEC = 60

# Headers that carry secrets — normalized to lowercase for case-insensitive
# matching. Anything matched here is redacted before the raw artifact lands
# on disk so the open-sourced traces can't leak API credentials.
REDACTED_HEADER_NAMES = {
    "x-api-key",
    "x-api-token",
    "authorization",
    "x-auth-token",
    "x-token",
    "api-key",
    "api_key",
    "apikey",
}
REDACTED_PLACEHOLDER = "***REDACTED***"


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


class SkipConfig(Exception):
    """A hook raises this to skip the current config cleanly (no HTTP call). The
    generic runner turns it into an error RunResult so the orchestrator records
    the skip and moves to the next config. Used e.g. by firmographic-filtered
    filtered configs when a seed carries no firmographic hints."""


@dataclasses.dataclass(frozen=True)
class Seed:
    """One row from `snapshot.seeds`. The exact input every vendor sees."""

    seed_slug: str
    seed_name: str
    seed_domain: str | None
    description: str | None
    category: str
    # Optional public firmographic hints. Lets a runner use a vendor's documented firmographic filter
    # surface (e.g. OpenFunnel locations / employee range / funding stages) so
    # each API can put its best foot forward. Shape:
    #   {"locations": ["USA"], "min_employees": 10, "max_employees": 2000,
    #    "funding_stages": ["seed", "series_a", ...]}
    firmographics: dict[str, Any] | None = None


@dataclasses.dataclass
class Candidate:
    """A single lookalike returned by a vendor. We persist whatever fields
    the vendor returned so the judge has enough to score; the orchestrator
    normalizes name/domain into the canonical pair."""

    name: str
    domain: str | None = None
    description: str | None = None
    rank: int | None = None
    # Free-form vendor payload so we don't drop signal (industry, employee
    # count, funding, etc.) — useful for judge context.
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RunResult:
    """One vendor run for one (seed, config)."""

    seed_slug: str
    provider_slug: str
    config_name: str
    config: dict[str, Any]
    candidates: list[Candidate]
    latency_ms: int
    cost_usd: float | None = None
    error: str | None = None
    requested_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_slug": self.seed_slug,
            "provider_slug": self.provider_slug,
            "config_name": self.config_name,
            "config": self.config,
            "candidates": [
                {
                    "name": c.name,
                    "domain": c.domain,
                    "description": c.description,
                    "rank": c.rank,
                    "extra": c.extra,
                }
                for c in self.candidates
            ],
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "requested_k": self.requested_k,
        }


@dataclasses.dataclass
class JudgeVote:
    """One judge model's verdict on one candidate. A cell with N judges stores N
    of these per candidate; the candidate's `relevant` is the majority over them."""

    judge_model: str
    relevant: bool
    rationale: str


@dataclasses.dataclass
class JudgedCandidate:
    candidate: Candidate
    relevant: bool          # majority label across `votes` (== the lone vote when N=1)
    rationale: str
    # Per-judge breakdown. Empty for legacy single-judge constructions; populated
    # by JudgePanel. The slim/snapshot artifacts carry only the aggregate
    # `relevant`/`rationale`; full votes live in the audit trail + Supabase.
    votes: list[JudgeVote] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class JudgedRun:
    """A RunResult after the judge has scored each candidate."""

    run: RunResult
    judged: list[JudgedCandidate]
    judge_model: str
    judged_at: str  # ISO-8601 UTC

    @property
    def k(self) -> int:
        return self.run.requested_k or len(self.judged)

    @property
    def relevant_count(self) -> int:
        return sum(1 for j in self.judged if j.relevant)

    @property
    def precision_at_k(self) -> float | None:
        if self.k == 0:
            return None
        return round(100.0 * self.relevant_count_at(self.k) / self.k, 2)

    def relevant_count_at(self, cutoff: int) -> int:
        return sum(1 for j in self.judged[:cutoff] if j.relevant)

    def precision_at(self, cutoff: int) -> float | None:
        if cutoff <= 0:
            return None
        return round(100.0 * self.relevant_count_at(cutoff) / cutoff, 2)


@dataclasses.dataclass
class RawHttpCall:
    """One round-trip captured for the open-source audit trail. Auth
    headers redacted. `response_body` is the parsed JSON if the response
    decoded cleanly, else the raw text. `response_text` is always the
    decoded body string so binary-style debug is possible if needed."""

    method: str
    url: str
    request_headers: dict[str, str]
    request_body: Any
    response_status: int
    response_body: Any
    response_text_preview: str  # first 4kb of body for quick eyeballing
    elapsed_ms: int


@dataclasses.dataclass
class RawJudgeCall:
    """One judge round-trip — exactly what the LLM saw and exactly what
    it returned. `messages` is the literal prompt; `raw_response` is the
    JSON string the model returned (before pydantic parsing). For
    open-source reproducibility this is the ground truth — anyone can
    rerun the same messages against their own LLM and check drift."""

    model: str
    messages: list[dict[str, str]]
    raw_response: str
    parsed_response: dict[str, Any] | None
    elapsed_ms: int
    for_candidate_rank: int | None = None  # which candidate this judged
    error: str | None = None  # if the LLM call failed, what went wrong


# --------------------------------------------------------------------------- #
# Env loading                                                                 #
# --------------------------------------------------------------------------- #


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Minimal .env loader — keeps zero hard runtime deps. Matches the style
    used by the technography ingest scripts. Strips a leading `export ` so
    shell-style lines (`export FOO=bar`) work the same as plain `FOO=bar`."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def require_env(*keys: str) -> dict[str, str]:
    """Read each key from env; raise if any is missing. Returns the dict
    of {key: value}. Use at the top of a vendor runner so the failure
    mode is clear."""
    load_dotenv()
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"missing required env var(s): {', '.join(missing)} — set in .env"
        )
    return {k: os.environ[k] for k in keys}


# --------------------------------------------------------------------------- #
# Snapshot I/O                                                                #
# --------------------------------------------------------------------------- #


def read_snapshot() -> dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"snapshot missing at {SNAPSHOT_PATH}")
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def write_snapshot(snapshot: dict[str, Any]) -> None:
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )


def read_seeds() -> list[Seed]:
    snap = read_snapshot()
    out: list[Seed] = []
    for s in snap.get("seeds", []):
        out.append(
            Seed(
                seed_slug=s["seed_slug"],
                seed_name=s["seed_name"],
                seed_domain=s.get("seed_domain"),
                description=s.get("description"),
                category=s["category"],
            )
        )
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Persistence — per-run detail + matrix aggregate                             #
# --------------------------------------------------------------------------- #


def raw_http_call_to_dict(call: RawHttpCall) -> dict[str, Any]:
    return {
        "method": call.method,
        "url": call.url,
        "request_headers": call.request_headers,
        "request_body": call.request_body,
        "response_status": call.response_status,
        "response_body": call.response_body,
        "elapsed_ms": call.elapsed_ms,
    }


def raw_judge_call_to_dict(call: RawJudgeCall) -> dict[str, Any]:
    return {
        "model": call.model,
        "messages": call.messages,
        "raw_response": call.raw_response,
        "parsed_response": call.parsed_response,
        "elapsed_ms": call.elapsed_ms,
        "for_candidate_rank": call.for_candidate_rank,
        "error": call.error,
    }


def persist_run_raw(
    dataset_slug: str,
    *,
    seed: Seed,
    provider_slug: str,
    provider_name: str,
    k: int,
    judge_model: str,
    winning_config_name: str | None,
    attempts: list[dict[str, Any]],
) -> Path:
    """Write the full audit trail for one (seed, vendor) cell. Sibling to
    the slim `<vendor>.json`. Captures every config attempted in the sweep
    with its literal HTTP request/response + judge prompt/response.

    Schema is documented in `data/lookalike-runs/README.md`. Shape:
        {
          "dataset_slug": ..., "seed_slug": ..., "seed_input": {...},
          "provider_slug": ..., "k": ...,
          "winning_config_name": ...,
          "judge_model": ...,
          "captured_at": ...,
          "attempts": [
            {
              "config": {...},
              "vendor_calls": [<raw_http_call>, ...],
              "extracted_candidates": [<the slim candidate dicts handed to the judge>],
              "judge_calls": [<raw_judge_call>, ...],
              "result": {
                "k": ..., "relevant_count": ..., "precision_at_k": ...,
                "latency_ms": ..., "judged_at": ..., "error": ...,
              },
            },
            ...
          ],
        }
    """
    out_dir = RUNS_DIR / dataset_slug / seed.seed_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider_slug}.raw.json"
    payload = {
        "dataset_slug": dataset_slug,
        "seed_slug": seed.seed_slug,
        "seed_input": {
            "seed_name": seed.seed_name,
            "seed_domain": seed.seed_domain,
            "description": seed.description,
            "category": seed.category,
        },
        "provider_slug": provider_slug,
        "provider_name": provider_name,
        "k": k,
        "winning_config_name": winning_config_name,
        "judge_model": judge_model,
        "captured_at": now_iso(),
        "attempts": attempts,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def persist_run_detail(dataset_slug: str, judged: JudgedRun) -> Path:
    """Write the full per-candidate judge output so any cell on the
    leaderboard can be audited row-by-row. The matrix JSON only carries
    the aggregate Precision@K — detail lives here."""
    run = judged.run
    out_dir = RUNS_DIR / dataset_slug / run.seed_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run.provider_slug}.json"
    payload = {
        "dataset_slug": dataset_slug,
        "seed_slug": run.seed_slug,
        "provider_slug": run.provider_slug,
        "config_name": run.config_name,
        "config": run.config,
        "k": judged.k,
        "relevant_count": judged.relevant_count,
        "precision_at_k": judged.precision_at_k,
        "relevant_count_at_10": judged.relevant_count_at(10),
        "relevant_count_at_50": judged.relevant_count_at(50),
        "relevant_count_at_100": judged.relevant_count_at(100),
        "precision_at_10": judged.precision_at(10),
        "precision_at_50": judged.precision_at(50),
        "precision_at_100": judged.precision_at(100),
        "latency_ms": run.latency_ms,
        "cost_usd": run.cost_usd,
        "judge_model": judged.judge_model,
        "judged_at": judged.judged_at,
        "candidates": [
            {
                "name": j.candidate.name,
                "domain": j.candidate.domain,
                "description": j.candidate.description,
                "rank": j.candidate.rank,
                "extra": j.candidate.extra,
                "relevant": j.relevant,
                "rationale": j.rationale,
            }
            for j in judged.judged
        ],
        "error": run.error,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def upsert_seed_vendor_cell(
    snapshot: dict[str, Any],
    *,
    dataset_slug: str,
    seed: Seed,
    provider_slug: str,
    provider_name: str,
    k: int,
    judged: JudgedRun | None,
    error: str | None = None,
) -> None:
    """Insert / replace one (seed, provider) cell in `snapshot.seed_vendors`.

    Pass `judged=None` (and optionally an `error` string) when the vendor
    failed or returned 0 candidates — the cell will land with
    precision_at_k=null so the matrix renders it as N/A."""
    cells: list[dict[str, Any]] = snapshot.setdefault("seed_vendors", [])

    # Drop any existing cell for this (seed, vendor) pair.
    cells[:] = [
        c
        for c in cells
        if not (c["seed_slug"] == seed.seed_slug and c["provider_slug"] == provider_slug)
    ]

    if judged is None:
        cells.append(
            {
                "dataset_slug": dataset_slug,
                "seed_slug": seed.seed_slug,
                "seed_name": seed.seed_name,
                "category": seed.category,
                "provider_slug": provider_slug,
                "provider_name": provider_name,
                "k": k,
                "returned_count": 0,
                "relevant_count": None,
                "precision_at_k": None,
                "relevant_count_at_10": None,
                "relevant_count_at_50": None,
                "relevant_count_at_100": None,
                "precision_at_10": None,
                "precision_at_50": None,
                "precision_at_100": None,
                "latency_ms": None,
                "cost_usd": None,
                "judge_model": None,
                "judged_at": None,
                "config_used": None,
                "error": error,
            }
        )
        return

    run = judged.run
    cells.append(
        {
            "dataset_slug": dataset_slug,
            "seed_slug": seed.seed_slug,
            "seed_name": seed.seed_name,
            "category": seed.category,
            "provider_slug": provider_slug,
            "provider_name": provider_name,
            "k": k,
            "returned_count": len(run.candidates),
            "relevant_count": judged.relevant_count,
            "precision_at_k": judged.precision_at_k,
            "relevant_count_at_10": judged.relevant_count_at(10),
            "relevant_count_at_50": judged.relevant_count_at(50),
            "relevant_count_at_100": judged.relevant_count_at(100),
            "precision_at_10": judged.precision_at(10),
            "precision_at_50": judged.precision_at(50),
            "precision_at_100": judged.precision_at(100),
            "latency_ms": run.latency_ms,
            "cost_usd": run.cost_usd,
            "judge_model": judged.judge_model,
            "judged_at": judged.judged_at,
            "config_used": {"name": run.config_name, **run.config},
        }
    )


def recompute_leaderboard(snapshot: dict[str, Any], k: int) -> None:
    """Re-aggregate `snapshot.leaderboard` from the current `seed_vendors`
    cells. Mirrors the pattern used by the technography ingest scripts —
    one source of truth, derived rollups always consistent."""
    cells: list[dict[str, Any]] = snapshot.get("seed_vendors", [])
    rows: list[dict[str, Any]] = snapshot.get("leaderboard", [])
    total_seeds = len(snapshot.get("seeds", []))

    by_vendor: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        by_vendor.setdefault(c["provider_slug"], []).append(c)

    for row in rows:
        my_cells = by_vendor.get(row["provider_slug"], [])
        judged_cells = [c for c in my_cells if c.get("precision_at_k") is not None]
        attempted = len({c["seed_slug"] for c in my_cells})

        # avg Precision@K — mean across judged cells only
        if judged_cells:
            avg = sum(c["precision_at_k"] for c in judged_cells) / len(judged_cells)
            row["avg_precision_at_k"] = round(avg, 2)
        else:
            row["avg_precision_at_k"] = None

        for cutoff in PRECISION_CUTOFFS:
            precision_key = f"precision_at_{cutoff}"
            avg_key = f"avg_precision_at_{cutoff}"
            relevant_key = f"relevant_count_at_{cutoff}"
            total_key = f"total_relevant_at_{cutoff}"
            cells_with_cutoff = [
                c for c in my_cells if c.get(precision_key) is not None
            ]
            if cells_with_cutoff:
                row[avg_key] = round(
                    sum(float(c[precision_key]) for c in cells_with_cutoff)
                    / len(cells_with_cutoff),
                    2,
                )
            else:
                row[avg_key] = None
            row[total_key] = sum(
                int(c.get(relevant_key) or 0) for c in cells_with_cutoff
            )

        row["seeds_attempted"] = attempted
        row["seeds_judged"] = len(judged_cells)
        row["total_seeds"] = total_seeds
        row["k"] = k
        row["total_relevant"] = sum(
            int(c.get("relevant_count") or 0) for c in judged_cells
        )
        row["total_returned"] = sum(
            int(c.get("returned_count") or 0) for c in my_cells
        )

        # avg latency — only over cells that actually ran (latency != null)
        latencies = [
            c["latency_ms"] for c in my_cells if isinstance(c.get("latency_ms"), int)
        ]
        row["avg_latency_ms"] = (
            round(sum(latencies) / len(latencies), 1) if latencies else None
        )

        row["total_cost_usd"] = round(
            sum(float(c.get("cost_usd") or 0.0) for c in my_cells), 4
        )
        row["cost_per_relevant_usd"] = (
            round(row["total_cost_usd"] / row["total_relevant"], 4)
            if row["total_relevant"] > 0
            else None
        )

    # Re-rank: avg_precision_at_k DESC (nulls last), then total_relevant DESC,
    # then provider_slug ASC. This exact tie rule is mirrored in
    # lookalike_leaderboard_view so the DB and JSON never order tied vendors
    # differently. (Current data has distinct precisions, so the secondary keys
    # change nothing today.)
    rows.sort(
        key=lambda r: (
            1 if r["avg_precision_at_k"] is None else 0,
            -(r["avg_precision_at_k"] or 0.0),
            -int(r.get("total_relevant") or 0),
            r["provider_slug"],
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    snapshot["generated_at"] = now_iso()


# --------------------------------------------------------------------------- #
# HTTP helpers — kept stdlib only to avoid an extra dep                       #
# --------------------------------------------------------------------------- #


# Active capture buffer for HTTP calls. Set by `capture_http_calls()`.
# When set, every successful and failed `http_request()` call appends a
# `RawHttpCall` to this list. Used by the orchestrator to record the full
# audit trail per (seed, vendor, config) attempt.
_HTTP_TRACE: ContextVar[list[RawHttpCall] | None] = ContextVar(
    "lookalike_http_trace", default=None
)


@contextlib.contextmanager
def capture_http_calls() -> Iterator[list[RawHttpCall]]:
    """Context manager that captures every `http_request()` call made
    inside the block. Returns the list buffer so the caller can drain it
    after the block exits.

    Usage in the orchestrator:
        with capture_http_calls() as calls:
            result = runner["run"](seed, k, config)
        attempt["vendor_calls"] = [_call_to_dict(c) for c in calls]
    """
    buf: list[RawHttpCall] = []
    token = _HTTP_TRACE.set(buf)
    try:
        yield buf
    finally:
        _HTTP_TRACE.reset(token)


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Replace values of well-known auth headers with REDACTED_PLACEHOLDER
    so the open-source raw artifacts can't leak API keys."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        out[k] = REDACTED_PLACEHOLDER if k.lower() in REDACTED_HEADER_NAMES else v
    return out


def http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout: int = HTTP_TIMEOUT_SEC,
) -> tuple[int, Any, int]:
    """Returns (status, json_or_text, elapsed_ms). Raises on network errors
    so callers can decide whether to retry.

    When a `capture_http_calls()` context is active, every call here is
    also appended to the active trace buffer as a fully redacted
    `RawHttpCall`. Runners don't need to know — the capture is
    transparent."""
    data: bytes | None = None
    hdrs = dict(headers or {})
    if body is not None and method.upper() != "GET":
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    hdrs.setdefault("Accept", "application/json")
    # Several vendors (Ocean, Lusha) sit behind Cloudflare and return a 403
    # "Error 1010: Access denied" for the default Python urllib UA. A
    # mainstream desktop UA passes their bot filters reliably and is also
    # the right thing to do for any benchmark talking to public APIs.
    hdrs.setdefault(
        "User-Agent",
        "openfunnel-bench/0.1 (+https://openfunnel.dev/bench; contact=founders@openfunnel.dev)",
    )

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=hdrs)
    start = time.monotonic()
    status: int
    parsed: Any
    response_text = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            response_text = raw.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                parsed = response_text
            status = resp.status
    except urllib.error.HTTPError as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        response_text = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = response_text
        status = e.code

    trace = _HTTP_TRACE.get()
    if trace is not None:
        trace.append(
            RawHttpCall(
                method=method.upper(),
                url=url,
                request_headers=_redact_headers(hdrs),
                request_body=body,  # already JSON-ish; safe to serialize
                response_status=status,
                response_body=parsed,
                response_text_preview=response_text[:4096],
                elapsed_ms=elapsed_ms,
            )
        )

    return status, parsed, elapsed_ms


def take_top(candidates: Iterable[Candidate], k: int) -> list[Candidate]:
    out: list[Candidate] = []
    for i, c in enumerate(candidates):
        if i >= k:
            break
        if c.rank is None:
            c.rank = i + 1
        out.append(c)
    return out
