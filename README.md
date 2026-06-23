# OpenBenchmarks Lookalike Benchmark

Open head-to-head leaderboard for company-lookalike APIs.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/lookalikes

This repo is the open data + code mirror of that page — every cell on the
leaderboard is backed by a literal HTTP request/response envelope and a
literal LLM judge prompt + response, both committed under `data/lookalike-runs/`.

For each seed company, every vendor returns its top-`K = 100` lookalikes; an LLM judge (`gpt-5.4-mini`) scores each returned company for relevance against the seed's business model;
cell value is **Precision@K** — relevant returned / number judged.

Precision@K is the headline metric. nDCG@K (binary-gain over judge labels),
MAP@K, MRR, and relative pooled Recall@K are also computed per cell and exposed
via the JSON API.

## Endpoints

- **All benchmarks (home)** — https://openbenchmarks.com
- **Live benchmark UI** — https://openbenchmarks.com/lookalikes
- **JSON API** — https://openbenchmarks.com/api/benchmarks/lookalikes
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json
- **MCP server discovery** — https://openbenchmarks.com/.well-known/mcp.json

## Current leaderboard

| # | Vendor | Precision@K | Judged |
|---|---|---|---|
| 1 | OpenFunnel | 69.75% | 24/24 |
| 2 | Parallel | 56.5% | 24/24 |
| 3 | Ocean.io | 48.61% | 23/24 |
| 4 | Exa | 25.79% | 24/24 |
| 5 | PredictLeads | 19.38% | 24/24 |

24 seed companies × 5 vendors. The full per-cell breakdown
and the raw audit trail (every HTTP request + every judge call) lives under
`data/lookalike-runs/`.

## What's in this repo

| path | purpose |
|---|---|
| `data/latest-lookalike.json` | The leaderboard snapshot — seeds, per-vendor rows, per-cell aggregates. |
| `data/lookalike-runs/<dataset>/<seed>/<vendor>.json` | Slim per-cell artifact — winning config's candidates with the judge's (majority) verdict + one-line rationale. |
| `data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json` | **Full audit trail** — every config attempted, with the literal HTTP request/response (auth headers redacted) and the literal LLM prompt + raw response for every judge × candidate. |
| `data/lookalike-runs/README.md` | Schema docs for the raw artifacts. |
| `manifest.json` | Flat index of every cell with the headline numbers + file paths. Easy to ingest programmatically. |
| `scripts/run_lookalike_benchmark.py` | Orchestrator. Sweeps every config each vendor declares; keeps the highest-Precision@K winner per cell. |
| `scripts/lookalike/specs/<vendor>.yaml` | Declarative vendor adapter — base URL, auth header, request template, response paths, candidate field map, cost. |
| `scripts/lookalike/generic_runner.py` | One generic runner that executes any spec (templating, response normalization, cost). |
| `scripts/lookalike/hooks/` | Optional per-vendor Python hooks for the few quirks that can't be declarative (e.g. domain preflight, page→domain dedupe, JSON:API joins). |
| `scripts/lookalike/runners/<vendor>.py` | Thin spec-binding stub per vendor (exposes the runner contract to the orchestrator). |
| `scripts/lookalike/metrics.py` | Pluggable metric registry — Precision@K (primary) + nDCG@K / MAP@K / MRR / pooled Recall@K. |
| `scripts/lookalike/judge.py` | LLM judge + multi-judge panel — system prompt, Pydantic verdict schema, majority vote, mock mode. |
| `scripts/lookalike/common.py` | Dataclasses + HTTP helper + persistence + redaction. |

## Reproducing a cell

Pick any (seed, vendor) pair on the leaderboard. The corresponding raw file at
`data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json` contains, for every
config the orchestrator swept:

- **`vendor_calls[].request_*`** — replay the HTTP call with your own credentials.
- **`judge_calls[].messages`** — replay the literal LLM prompt (one entry per
  judge × candidate) against any OpenAI-v1 compatible model (OpenAI direct,
  Azure, Anthropic via converter, local llama, your own fine-tune).

Every Precision@K number is backed by a literal HTTP envelope you can re-run plus
the literal LLM prompts you can re-score with your own judge(s) to measure bias.

## Running the benchmark yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt          # includes PyYAML for the specs
cp .env.example .env && $EDITOR .env                # fill in vendor keys + judge endpoint

PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --mock          # offline smoke test (no keys)
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py                 # live full sweep (single judge)
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --judges gpt-5.4-mini,gpt-5.2,o4-mini   # multi-judge panel
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --only openfunnel --seeds pylon
```

## Contributing a new vendor

1. Add `scripts/lookalike/specs/<your_vendor>.yaml` — base URL, auth header + env
   var, request template, the response path to the candidate list, and the field
   map (name/domain/description/rank/extra). Most vendors need **no Python**.
2. Only if the vendor has a quirk a spec can't express (a preflight call, a
   non-standard response shape), add a hook in `scripts/lookalike/hooks/` and
   reference it by name in the spec.
3. Add a one-line spec-binding stub `scripts/lookalike/runners/<your_vendor>.py`
   and register the slug in `scripts/lookalike/runners/__init__.py::REGISTRY`.
4. Add a row to `data/latest-lookalike.json::leaderboard`, run
   `python scripts/run_lookalike_benchmark.py --only <your_vendor>`, and open a PR.

## Methodology

- **Precision@K.** For each seed, the vendor returns up to K candidates. The
  judge labels each candidate `relevant: bool` against the seed's description.
  Cell value = `relevant_count / number judged` (the K returned, or fewer if the
  vendor returned < K). Vendor row = mean across judged seeds.
- **Best-of sweep.** Each runner declares 1-4 config variants (e.g. agentic vs
  semantic, with-query vs seed-only). For every cell we run all configs and
  keep the highest-Precision@K winner. Tiebreaker: more judged candidates,
  then lower latency.
- **Judge.** Single-pass binary verdict + 1-line rationale per candidate. Same prompt and rubric across all vendors. The literal prompt and raw model response are published so judge bias is fully auditable — swap the model and re-score to measure drift.
- **Other metrics.** nDCG@K uses binary gain from the judge-majority labels over
  the vendor's ranking (not graded relevance). Recall@K is **relative pooled
  recall** (TREC-style): relevant returned / distinct relevant companies any
  vendor surfaced for that seed — not absolute recall against a ground-truth
  universe.
- **Known limitations.** (1) Judge bias: a single LLM judge has its own priors; the full audit trail lets you swap and re-score.
  (2) K-tail vs precision tradeoff: vendors that can only return small sets win
  P@K by default. (3) Recall is relative (pooled across surveyed vendors), so it
  understates misses no vendor surfaced.
