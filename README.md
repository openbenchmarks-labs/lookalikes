# Lookalike Benchmark

Open head-to-head leaderboard for company-lookalike APIs.

**Live leaderboard:** http://openbenchmarks.com/lookalikes

This repo is the open data + code mirror of that page — every cell on the
leaderboard is backed by a literal HTTP request/response envelope and a
literal LLM judge prompt + response, both committed under `data/lookalike-runs/`.

For each seed company, every vendor returns its top-`K = 10` lookalikes; a panel of LLM judges (`majority(n=3): gpt-5.1,gpt-5.2,gpt-5.4-mini`) scores each returned company and a candidate is marked relevant by **majority vote**;
cell value is **Precision@K** — relevant returned / number judged.

Precision@K is the headline metric. nDCG@K (binary-gain over judge-majority
labels), MAP@K, MRR, and relative pooled Recall@K are also computed per cell and
exposed via the JSON API. A separate, **judge-free TAM-recall benchmark** (see
[TAM Recall](#tam-recall--second-benchmark-coverage-judge-free) below) measures
market *coverage* against a frozen, vendor-independent reference set.

## Endpoints

- **Live leaderboard UI** — http://openbenchmarks.com/lookalikes
- **JSON API** — http://openbenchmarks.com/api/lookalikes
- **Markdown agent docs** — http://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — http://openbenchmarks.com/openapi.json
- **MCP server discovery** — http://openbenchmarks.com/.well-known/mcp.json

## Current leaderboard

| # | Vendor | Precision@K | Judged |
|---|---|---|---|
| 1 | OpenFunnel | 89.11% | 14/14 |
| 2 | PredictLeads | 73.57% | 14/14 |
| 3 | Ocean.io | 70.71% | 14/14 |
| 4 | Parallel | 64.62% | 13/14 |
| 5 | Exa | 31.82% | 14/14 |

14 seed companies × 5 vendors. The full per-cell breakdown
and the raw audit trail (every HTTP request + every judge call) lives under
`data/lookalike-runs/`.

## TAM Recall — second benchmark (coverage, judge-free)

A complementary benchmark answering the opposite question to Precision@K: not
"are the few you returned good?" but **"build me my whole TAM."** Each vendor
returns its deepest list (fetch depth 100); we match it against a
**frozen, vendor-independent reference set** (G2 category rosters resolved to
canonical domains) and report **Recall@K** — the fraction of that reference set
the vendor surfaced in its top K. A "hit" is deterministic reference-set
membership — **no LLM judge**. Recall is relative to the reference set, **not
absolute TAM**.

| # | Vendor | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| 1 | Ocean.io | 2.15% | 6.49% | 8.3% |
| 2 | OpenFunnel | 1.89% | 4.83% | 8.21% |
| 3 | Parallel | 1.19% | 4.18% | 5.67% |
| 4 | PredictLeads | 2.93% | 5.56% | 5.56% |
| 5 | Exa | 0.54% | 1.15% | 2.04% |

Reference ("gold") set sizes per seed: Postscript (431), Pylon (107), Recharge (135), ServiceTitan (499). Each is built from public,
vendor-independent sources (never a benchmarked vendor) and content-hash-frozen
*before* any vendor runs — see `scripts/lookalike/RECALL_METHODOLOGY.md` and
`data/lookalike-tam/gold/`. The fairness audit
in `data/latest-lookalike-tam.json` shows most surfaced gold companies are found
by **≥2 independent vendors**, evidence the reference set isn't biased toward
any single vendor.

Run it (judge-free): `PYTHONPATH=scripts python scripts/run_tam_recall_benchmark.py`
(`--mock` for offline). Per-cell recall audits — configs swept + the matched gold
companies (rank + match method) — live under `data/lookalike-tam-runs/`.

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
| **TAM Recall** | |
| `data/latest-lookalike-tam.json` | Recall snapshot — per-vendor recall leaderboard, per-(seed,vendor) cells, and the per-vendor fairness audit. |
| `data/lookalike-tam/gold/<seed>.gold.json` | Frozen, content-hashed reference ("gold") set per seed — the recall denominator. `<seed>.g2.json` is the raw G2 harvest layer (provenance). |
| `data/lookalike-tam/seeds.json` | Recall seed registry — seed + G2 category slug + public firmographic hints. |
| `data/lookalike-tam-runs/<seed>/<vendor>.json` | Per-cell recall audit — every config swept, matched gold companies (rank + match method: domain/alias/name), and the recall metrics. |
| `scripts/run_tam_recall_benchmark.py` | Judge-free recall orchestrator — fetches depth N per vendor, scores by gold-set overlap, writes the snapshot. |
| `scripts/lookalike/recall.py` | Recall core — canonical-domain matching, Recall@K / R-Precision / Hit@K, fairness audit, gold freeze + hash verification. |
| `scripts/lookalike/goldset/` | Gold-set harvest pipeline (G2 roster → domain resolution → Wikidata/Wikipedia corroboration → inclusion rule → freeze). |
| `scripts/lookalike/RECALL_METHODOLOGY.md` | Full recall methodology — gold construction (incl. the NAICS spec for non-software verticals), matching, fairness, versioning. |

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
- **Judge (multi-judge panel).** Several LLM judges score each candidate independently; the `relevant` label is the **majority vote** (even-N ties resolve to not-relevant). Every judge's verdict is stored and the literal prompt + raw response for *each* judge is published, so you can audit disagreement and swap judges to measure bias.
- **Other metrics.** nDCG@K uses binary gain from the judge-majority labels over
  the vendor's ranking (not graded relevance). Recall@K is **relative pooled
  recall** (TREC-style): relevant returned / distinct relevant companies any
  vendor surfaced for that seed — not absolute recall against a ground-truth
  universe.
- **Known limitations.** (1) Judge bias: a panel reduces but doesn't eliminate shared model priors; per-judge votes + prompts are published so you can re-score with your own jury.
  (2) K-tail vs precision tradeoff: vendors that can only return small sets win
  P@K by default. (3) Recall is relative (pooled across surveyed vendors), so it
  understates misses no vendor surfaced.
