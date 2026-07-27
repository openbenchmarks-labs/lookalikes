# OpenBenchmarks Lookalike Benchmark

Open head-to-head leaderboard for company-lookalike APIs.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/lookalikes

This repo is the open data + code mirror of that page — every cell on the
leaderboard is backed by a literal HTTP request/response envelope and a
literal LLM judge prompt + response, both committed under `data/lookalike-runs/`.

For each seed company, each provider returns an ordered list of lookalikes. An LLM judge (`gpt-5.6`, high reasoning effort) scores every returned company against the seed's business model. We report **Precision@10**, **Precision@25**, and **Precision@100** where the provider returns that depth.

A score at cutoff N is relevant companies in the top N divided by N. Providers that return fewer than a cutoff are not scored at that cutoff. The full candidate-level labels, request/response envelopes, and judge calls are published below.

## Endpoints

- **Live benchmark UI** — https://openbenchmarks.com/lookalikes
- **JSON API** — https://openbenchmarks.com/api/benchmarks/lookalikes
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json
- **MCP server discovery** — https://openbenchmarks.com/.well-known/mcp.json

## Current leaderboard

| # | Vendor | Precision@K | Judged |
|---|---|---|---|
| 1 | Parallel | 67.54% | 48/48 |
| 2 | Extruct | 61.21% | 48/48 |
| 3 | Ocean.io | 56.53% | 47/48 |
| 4 | Exa | 52.4% | 48/48 |
| 5 | Discolike | 35.19% | 47/48 |
| 6 | CUFinder | N/A | 47/48 |
| 7 | PredictLeads | N/A | 48/48 |

48 seed companies × 7 vendors. The full per-cell breakdown
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
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --judge-model gpt-5.6   # reproduce the published Q3 judge
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

- **Fixed cohort and cutoffs.** The Q3 snapshot scores 48 seed companies across seven providers. We request up to 100 ranked candidates and report Precision@10, Precision@25, and Precision@100 only where a provider reaches that depth.
- **Precision@N.** Relevant companies in the top N ÷ N. A provider that returns fewer than N is not scored at that cutoff. Provider rows are means across judged seeds.
- **Judge.** `gpt-5.6` at high reasoning effort assigns a binary relevance label and one-line rationale to every returned candidate. The identical rubric is used across vendors; funding, geography, and company maturity are not relevance constraints unless they define the core product or buyer.
- **Duplicate handling.** A shared post-fetch check removes duplicate companies before scoring. Repeated companies count once; the raw audit trail retains the original calls and dedupe record.
- **NLU prompt sweep.** Exa and Parallel each run the same three fixed variants per seed and retain the list with the highest Precision@100; ties break on more judged candidates, then lower latency. The variants are: (1) full product-and-buyer framing plus buyer-evaluation instruction, (2) product-and-buyer framing, and (3) buyer-evaluation framing. Domain-first providers do not receive this sweep.
- **Other metrics.** nDCG@K uses binary gain over the ranked labels; MAP@K and MRR are also computed. Relative pooled Recall@K measures overlap with relevant companies surfaced by the surveyed providers, not absolute market coverage.
- **Known limitations.** The LLM judge has model-specific priors, and a per-seed best-of-prompt result is not a single fixed-production-prompt average. Every raw call and judge response is published so either can be audited or re-scored.
