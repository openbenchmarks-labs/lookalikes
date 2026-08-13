# OpenBenchmarks Lookalike Benchmark

Open head-to-head leaderboard for company-lookalike APIs.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/lookalikes

This repo is the open data + code mirror of that page — every cell on the
leaderboard is backed by the literal vendor call that produced it and the
literal LLM judge prompt + response, both committed under `data/lookalike-runs/`.
For seven of the eight vendors that call is an HTTP request/response envelope;
ZoomInfo has no API for this, so its cells record the `gtm` CLI invocation
instead.

For each seed company, each provider returns an ordered list of lookalikes. A panel
of three LLMs from three independent labs scores every returned company, and a
candidate counts as relevant only on a strict majority. We report **Precision@10**
and **Precision@100**, each only where the provider's endpoint reaches that depth.

A score at cutoff N is relevant companies in the top N divided by **N**, not by how
many the provider actually returned — so under-filling lowers the score rather than
being hidden by a smaller denominator. The full candidate-level labels,
request/response envelopes, and judge calls are published below.

## What counts as a lookalike

One definition, applied identically to every vendor, in two stages:

1. **Anchor — a hard gate.** The candidate must be the same kind of company selling
   to the same kind of buyer as the seed. Miss the anchor and nothing else saves it.
2. **Capabilities — matched any-of.** Past the anchor, the candidate must overlap on
   at least one of the capabilities in the seed description, not all of them.

This split is why the rubric is a gate plus a check rather than one holistic score,
and why the anchor is decided before the verdict. In practice the anchor does nearly
all the work: capability overlap flips well under 2% of verdicts.

## Endpoints

- **Live benchmark UI** — https://openbenchmarks.com/lookalikes
- **JSON API** — https://openbenchmarks.com/api/benchmarks/lookalikes
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json
- **MCP server discovery** — https://openbenchmarks.com/.well-known/mcp.json

## Current leaderboard

Two boards, because deep relevance and top-of-list relevance are different products
and the same vendor rarely wins both.

**Long list — Precision@100.** How relevant a vendor stays a hundred companies deep.

| # | Vendor | Precision@100 | Judged |
|---|---|---|---|
| 1 | Extruct | 61.1% | 48/48 |
| 2 | Ocean.io | 59.5% | 47/48 |
| 3 | Parallel | 56.9% | 47/48 |
| 4 | Exa | 48.6% | 48/48 |
| 5 | Discolike | 34.6% | 47/48 |

CUFinder, PredictLeads and ZoomInfo do not appear on this board. Their endpoints
return at most 10, 25 and 25 results respectively, so ranking them at a cutoff they
cannot reach would report the ceiling rather than the provider.

**Short list — Precision@10.** How relevant the top of the list is, which is what
matters when a person works the results by hand.

| # | Vendor | Precision@10 | Judged |
|---|---|---|---|
| 1 | PredictLeads | 89.8% | 48/48 |
| 2 | Exa | 80.8% | 48/48 |
| 3 | Extruct | 75.0% | 48/48 |
| 4 | Ocean.io | 74.3% | 47/48 |
| 5 | Parallel | 69.2% | 47/48 |
| 6 | ZoomInfo | 67.1% | 48/48 |
| 7 | Discolike | 41.9% | 47/48 |
| 8 | CUFinder | 38.9% | 47/48 |

48 seed companies × 8 vendors. The full per-cell breakdown
and the raw audit trail (every vendor call + every judge call) lives under
`data/lookalike-runs/`.

## What's in this repo

| path | purpose |
|---|---|
| `data/latest-lookalike.json` | The leaderboard snapshot — seeds (with anchor, capabilities and authored query), per-vendor rows, per-cell aggregates. |
| `data/lookalike-runs/<dataset>/<seed>/<vendor>.json` | Slim per-cell artifact — winning config's candidates with the panel's verdict, per-judge votes, anchor decision and matched capabilities. |
| `data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json` | **Full audit trail** — every config attempted, with the literal vendor call (HTTP request/response with auth headers redacted, or a CLI invocation for ZoomInfo) and the literal LLM prompt + raw response for every judge × candidate. |
| `data/lookalike-runs/README.md` | Schema docs for the raw artifacts. |
| `manifest.json` | Flat index of every cell with the headline numbers + file paths. Easy to ingest programmatically. |
| `scripts/run_lookalike_benchmark.py` | Orchestrator. Sweeps every config each vendor declares; keeps the highest-Precision@K winner per cell. |
| `scripts/lookalike/specs/<vendor>.yaml` | Declarative vendor adapter — base URL, auth header, request template, response paths, candidate field map, cost. |
| `scripts/lookalike/generic_runner.py` | One generic runner that executes any spec (templating, response normalization, cost). |
| `scripts/lookalike/hooks/` | Optional per-vendor Python hooks for the few quirks that can't be declarative (e.g. domain preflight, page→domain dedupe, JSON:API joins). |
| `scripts/lookalike/runners/<vendor>.py` | Thin spec-binding stub per vendor (exposes the runner contract to the orchestrator). |
| `scripts/lookalike/metrics.py` | Pluggable metric registry — Precision@K (primary) + nDCG@K / MAP@K / MRR / pooled Recall@K. |
| `scripts/lookalike/judge.py` | LLM judge panel — system prompt, Pydantic verdict schema, code-enforced anchor gate, majority vote, verdict cache, mock mode. |
| `scripts/lookalike/common.py` | Dataclasses + HTTP helper + persistence + redaction. |

## Reproducing a cell

Pick any (seed, vendor) pair on the leaderboard. The corresponding raw file at
`data/lookalike-runs/<dataset>/<seed>/<vendor>.raw.json` contains, for every
config the orchestrator swept:

- **`vendor_calls[].request_*`** — replay the vendor call with your own
  credentials. `method` is the HTTP verb for API-backed vendors and `"CLI"` for
  ZoomInfo, where `url` is the command and `request_body` its arguments.
- **`judge_calls[].messages`** — replay the literal LLM prompt (one entry per
  judge × candidate) against any OpenAI-v1 compatible model (OpenAI direct,
  Azure, Anthropic via converter, local llama, your own fine-tune).

Judge calls carry a `cached` flag. Within a run a verdict is memoised on
(judge, prompt version, seed, candidate), so a company returned by several vendors is
scored once and the reused entries are marked rather than silently omitted.

Every Precision@K number is backed by a literal vendor call you can re-run plus
the literal LLM prompts you can re-score with your own judge(s) to measure bias.

## Running the benchmark yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt          # includes PyYAML for the specs
cp .env.example .env && $EDITOR .env                # fill in vendor keys + all three judge keys

PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --mock          # offline smoke test (no keys)
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py                 # live full sweep
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py \
  --judges claude-opus-5,gpt-5.6-terra,kimi-k3                               # reproduce the published Q3 panel
PYTHONPATH=scripts python scripts/run_lookalike_benchmark.py --only exa --seeds toast
```

Reproducing the published numbers needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and
`FIREWORKS_API_KEY`. With fewer keys the run still works, but a single judge will not
reproduce a majority-vote result.

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

- **Fixed cohort and cutoffs.** The Q3 snapshot scores 48 seed companies across eight
  providers. We request up to 100 ranked candidates and report Precision@10 and
  Precision@100, each only where a provider's endpoint reaches that depth.
- **Precision@N.** Relevant companies in the top N ÷ N. The denominator is the cutoff,
  not the number of results returned, so a provider that returns 84 companies when
  asked for 100 is scored as though it returned 100 and 16 were wrong. A provider
  whose endpoint caps below a cutoff is left off that board entirely rather than
  shown near zero. Provider rows are means across judged seeds.
- **Judge panel.** Three models from three independent labs — `claude-opus-5`
  (Anthropic), `gpt-5.6-terra` (OpenAI), and `kimi-k3` (Moonshot AI, open weights) —
  run one fixed prompt over every candidate with the same seed context. Different
  labs and different training data means they do not share blind spots. A candidate
  is relevant by strict majority; ties resolve to not relevant.
- **The verdict is not taken from the model.** Each judge returns the anchor decision
  and the capabilities it matched as structured fields, and the `relevant` flag is
  recomputed in code from those fields. A model cannot mark a candidate relevant
  while failing the anchor.
- **Gated panel.** With an odd panel the later judges are consulted only while the
  vote is still undecided, so the third model runs only when the first two disagree.
  Set `LOOKALIKE_JUDGE_FULL_PANEL=1` to always run all three, e.g. to measure
  per-model agreement.
- **Natural-language query.** Domain-first providers receive a company domain. Exa and
  Parallel take natural language, so each seed has one hand-authored query and both
  vendors receive the identical string — that identity is what makes the two
  comparable. The query states the anchor and the capabilities separately, mirroring
  the rubric. Every seed's query is published in `data/latest-lookalike.json` and in
  the per-cell audit trail. The earlier three-variant prompt sweep has been retired,
  so a published figure is a single query's result rather than a per-seed maximum.
- **One vendor is CLI-driven.** ZoomInfo's similar-companies surface is not an HTTP
  API: the runner shells out to `gtm companies similar --name "<seed>"`, so its audit
  trail records the command, its arguments, its exit status and its raw output instead
  of a request envelope. Reproducing a ZoomInfo cell needs the GTM CLI on PATH rather
  than an API key; sign-up is self-serve at gtm.ai and there is a free tier, so the
  barrier is the install, not procurement. Its results also carry no website and no
  description, only a name plus ZoomInfo's own firmographic attributes and a
  similarity score, and the benchmark scores what the endpoint returned rather than
  enriching it in a second call.

- **Duplicate handling.** A shared post-fetch check removes duplicate companies before
  scoring. Repeated companies count once; the raw audit trail retains the original
  calls and dedupe record.
- **Other metrics.** nDCG@K uses binary gain over the ranked labels; MAP@K and MRR are
  also computed. Relative pooled Recall@K measures overlap with relevant companies
  surfaced by the surveyed providers, not absolute market coverage. The data also
  carries `precision_at_25` and `relevant_count_at_25`, which were measured but are
  no longer among the reported cutoffs.
- **Known limitations.** The panel has its own calibration, held constant across
  vendors, so vendor-to-vendor rank comparisons are valid while absolute Precision@K
  carries a panel-specific offset — do not port an absolute number into a business
  case. Latency is measured on a cold, unthrottled path rather than under production
  concurrency. Every raw call and judge response is published so any of this can be
  audited or re-scored.
