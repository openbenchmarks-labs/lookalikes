# TAM Recall Benchmark — Methodology

**Status:** frozen spec v1. Changes are versioned (see [Versioning](#9-versioning--freeze)).

This document is the contract for the TAM-recall leaderboard. It exists so the
benchmark can be **audited and reproduced by a third party (human or agent)
without trusting our numbers** — only our published process and frozen inputs.
If any result here can be questioned, the gap is a bug in this spec, not in the
data.

---

## 1. The question this benchmark answers

The existing lookalike benchmark answers *"are the top results clean?"*
(Precision@K). This one answers the question a GTM operator actually asks when
building a Total Addressable Market list:

> **"Build me my TAM of `{category}` companies."**
> Input: one representative seed company (+ optional firmographic filters).
> Output: as many real, similar companies as exist in the market.

TAM-building is fundamentally a **coverage / recall** problem: a vendor that
returns 10 perfect companies but misses the other 2,000 in the market is
useless for sizing or working a TAM. So the headline metric is **recall against
an independent reference set**, not precision.

### 1.1 Honest naming

We never claim to measure *absolute* recall — nobody knows the literal complete
set of every company in a market. We measure **recall against a frozen,
independently-sourced reference set of known-similar companies.** Every surface
(leaderboard, llms.txt, tooltips) uses that phrasing. Claiming unqualified
"recall" would be the one thing a skeptic could legitimately attack.

---

## 2. Task definition (identical for every vendor)

Each benchmark **case** is:

```
seed:     { name, domain }              # one representative company
filters:  { locations?, funding_stages?, min_employees?, max_employees? }   # optional
ask:      "companies similar to {seed} that fit {filters}"
```

Rules that make cases comparable:

1. **Same seed, same filters, same fetch depth → every vendor.** No per-vendor
   input tailoring beyond the documented config sweep (inherited from the
   precision benchmark).
2. **Filters are passed identically** to every vendor that supports them.
   Vendors that don't support a given filter are recorded in a **capability
   matrix** (not silently penalized); their cell is annotated `filter:
   unsupported`.
3. **Filters constrain the reference set too** (see §5.4). If a case applies a
   filter, the gold denominator is the reference set *intersected with the same
   filter predicate*, so recall stays apples-to-apples. For the first cohort we
   recommend filters be **empty or location-only** to keep gold sets clean;
   funding/headcount filters are opt-in per case and require the reference set
   to carry those attributes.

---

## 3. Fetch depth & efficiency

- **Fetch depth `N = 100`**, identical for all vendors. Recall@K is
  mathematically capped at `K / |gold|`, so `N` must be ≥ the largest gold set
  for recall to be able to reach 100%.
- **Fetch once, score many.** We pull the top `N` from each vendor a single
  time, then compute Recall@10, @50, @100 by **truncating the same ranked
  list**. One expensive call per (case, vendor) yields the whole recall curve.
- **A vendor's max-return ceiling is a published result.** If a vendor caps
  output below `N` (e.g. only ever returns 25), we record `returned_count` and
  surface the ceiling — a vendor that can't return >25 cannot build a
  1,000–5,000-company TAM regardless of top-of-list quality.

---

## 4. Metrics

All metrics reuse binary relevance labels from the existing LLM judge **and/or**
membership in the reference set (see §6 for how they combine). For a case with
reference set `G` (size `R = |G|`) and a vendor's ranked candidates
`c_1 … c_N`:

| Metric | Definition | Why it matters for TAM |
|---|---|---|
| **Recall@K** | `|G ∩ {c_1…c_K}| / R` | Market coverage at depth K (headline, K∈{10,50,100}) |
| **R-Precision** | precision at `K = R` | Size-normalized single number, fair across seeds with different `R` |
| **Hit@K** | `1` if `G ∩ {c_1…c_K} ≠ ∅` else `0` | Floor metric — did the vendor whiff entirely on this seed? |
| **Precision@K** | `|relevant ∩ {c_1…c_K}| / K` | Retained from the existing benchmark — top-of-list cleanliness |
| **Coverage** | fraction of the *union* gold universe a vendor surfaces across all seeds | Vendor-level breadth |

- **Aggregation is macro** (mean of per-seed values), matching the existing
  `avg_precision_at_k`. Every seed counts equally regardless of `R`. Micro
  (pooled) is reported as a secondary column.
- A cell where a vendor errored or returned 0 candidates is `null` (rendered
  N/A), never `0` — same convention as the precision benchmark.

---

## 5. The reference (gold) set — construction rules

This is the heart of "airtight." The governing principle:

> **The reference set is built by a process that never queries any vendor under
> test — including OpenFunnel.** Provenance is external, public, and citable.

### 5.1 Vendor independence (hard rule)

The harvest toolchain MUST exclude every benchmarked vendor and its API:
OpenFunnel, Exa, Ocean, Parallel, PredictLeads, Lusha, ZoomInfo. Using e.g. Exa
search to harvest competitors would reintroduce the exact vendor-dependence this
design avoids. This is enforced in code (`goldset/sources.py` has no vendor
client) and stated on the methodology page.

### 5.2 Per-category source registry (pre-registered)

There is no single universal source. We pre-register, **per category, before
running**, which third-party rosters are sanctioned. Sources are tiered:

- **Tier-A (authoritative roster):** a single neutral analyst/trade body whose
  *whole published list* defines a market. Qualifies a company for inclusion
  **on its own**.
- **Tier-B (corroborating):** credible but partial; requires **≥2 independent
  Tier-B sources** to include a company.

| Category | Tier-A sources | Tier-B sources |
|---|---|---|
| `b2b-saas`, `devtools`, `ecommerce` (software) | Gartner Peer Insights "market" rosters | G2 / Capterra category pages, Crunchbase competitors, Wikidata `industry` |
| `home-services` (software/chains) | Gartner FSM market (where applicable) | G2 category, Crunchbase, franchise directories |
| `trades` (HVAC, plumbing, electrical) | Trade-press league tables (e.g. RER 100 for rental); IBISWorld market-share leaders | Trade-association member directories (ACCA, PHCC, ARA), franchise directories (Franchise 500), NAICS-enumerated peers |
| `real-estate` | Trade league tables (e.g. NMHC Top 50 operators) | Crunchbase, NAICS-enumerated peers, association directories |

**Inclusion rule:** a company enters `G` iff it appears in **≥1 Tier-A roster**
**OR** **≥2 independent Tier-B sources**. Every inclusion stores its sources.

### 5.3 Anti-cherry-pick rules (pre-registered)

1. **Take whole lists.** When a source publishes a roster (a Gartner market, a
   G2 category, the RER 100), ingest the *entire* list. Selecting a subset
   *within* a source is forbidden — that's where bias re-enters.
2. **Reviewer can drop, never add.** A human QA pass may remove obvious junk
   (the seed itself, parents/subsidiaries, dead domains, duplicates) but may
   **not** add a company that isn't in a sanctioned source.
3. **Self-exclusion.** `G` excludes the seed, its parent, and its subsidiaries.
4. **ToS / copyright.** Factual membership ("X is in market Y") is not
   copyrightable; the curated prose is. We store and redistribute only the
   derived domains + a citation to the source, never mirrored source text.
   Gated sources (Gartner, RER 100, association directories) are captured
   manually as facts, not scraped wholesale.

### 5.4 Filters and the denominator

If a case carries a filter, `G` is intersected with the same predicate using
attributes already on the gold companies (region, employee band, funding). A
gold company missing the attribute needed to evaluate a filter is **excluded
from that filtered case's denominator** (we can't fairly require a vendor to
return it under a filter we can't verify). Filtered and unfiltered runs are
separate cases.

### 5.5 How the current G2-rostered gold sets were built (software seeds)

The four shipped seeds (`pylon`, `postscript`, `recharge`, `servicetitan`) are
software/B2B categories where **G2's category roster is the Tier-A source**.
The build is a two-stage harvest (`goldset/build_seeds.py`), then a human QA
pass, then a freeze:

1. **Harvest the whole G2 category roster** — `goldset/g2_rapidapi.py --seed
   <slug> --category-slug <g2_category_slug>` pulls the *entire* published G2
   category roster via the G2 Data API (RapidAPI). The full list is ingested
   (anti-cherry-pick §5.3.1); each product becomes the Tier-A source
   `g2cat:<slug>`, and G2 product "alternatives" are captured as the Tier-B
   signal `g2alt:<slug>`. Responses cache under `gold/_g2cache/` so the sweep is
   re-runnable.
2. **Resolve each product to a company domain** (`goldset/namedomain.py`): first
   the product's G2 `company_website`; when absent (G2 returns it for only a
   minority), fall back to Clearbit autocomplete (`resolution=clearbit-name`)
   guarded by a **name-token-overlap check** so a fuzzy match can't bind the
   wrong domain. Clearbit (HubSpot) is **not a benchmarked vendor** — §5.1 holds.
3. **Corroborate from open sources** (`goldset/wikidata.py`, optional): Wikidata
   `industry`/`instance-of` (`wikidata`) and an English Wikipedia article
   (`wikipedia`) are independent Tier-B signals; a company with both clears the
   ≥2-Tier-B bar on open data alone.
4. **Human QA, then inclusion rule** (`goldset/build.py`): `build.py harvest`
   emits a `<seed>.raw.json` working file; a reviewer edits it (and/or
   `<seed>.manual.json`) to **drop junk only** (the seed itself, parents/subs,
   dead domains, dupes) — **never to add** an unsourced company (§5.3.2).
   `build.py freeze` then merges the layers (`<seed>.g2.json` + manual/raw),
   keeps only companies passing `sources.qualifies` (**≥1 Tier-A OR ≥2 Tier-B**),
   and records the rejections with reasons.
5. **Freeze + verify**: `freeze` computes the content hash
   (`recall.GoldSet.compute_hash` over sorted name + canonical domains) and
   writes the dated `<seed>.gold.json`; `build.py verify` and
   `recall.load_goldset` (at run time) recompute the hash and refuse to load on
   mismatch.

**Adding a software seed:** append an entry to `data/lookalike-tam/seeds.json`
(`seed_slug`, `seed_name`, `seed_domain`, `description`, `category`,
`g2_category_slug`, optional `firmographics`), then run
`python3 scripts/lookalike/goldset/build_seeds.py --seeds <slug>`. The frozen
`<seed>.gold.json` **and** the raw `<seed>.g2.json` provenance are committed (and
mirrored to the public OSS repo); the `_g2cache/` / `_raw/` API caches stay
local. Per-company provenance is queryable (`lookalike_gold_provenance_view`) and
the freeze is independently re-checkable (`lookalike_gold_integrity_view`).

### 5.6 Non-G2 / traditional-industry gold via NAICS (spec — not yet built)

G2 rosters only cover software markets. For **trades, real-estate, home-services
and other local/SMB verticals** there is no G2 category, so the `naics` Tier-B
source (already registered in `sources.py` for `trades`/`real-estate`) is built
from public, vendor-independent US-government data. This is a **specification**
for future seeds — no traditional-industry seed ships yet.

1. **Anchor the vertical to 6-digit NAICS codes.** NAICS is the OMB-mandated
   federal classification (hierarchy: 2-digit sector → 3 subsector → 4 industry
   group → 5 NAICS industry → 6 national industry). Pin each vertical to specific
   codes and **freeze to a NAICS vintage** (2022 is in force; a 2027 revision is
   underway) — retain the Census SIC↔NAICS + inter-vintage concordances so
   numbers survive a revision. Examples:

   | Vertical | 6-digit NAICS |
   |---|---|
   | HVAC / plumbing contractors | 238220 |
   | Electrical contractors | 238210 |
   | Residential property managers | 531311 |
   | Nonresidential property managers | 531312 |
   | Physicians' offices | 621111 |

   Source: Census NAICS Manual + search (`census.gov/naics`).
2. **Named gold set → SAM.gov public entity extract.** The gold *set* (named
   companies) comes from the SAM.gov public entity-registration extract: a
   frozen, date-stamped, pipe-delimited bulk ZIP carrying legal business name,
   DBA, structured physical address, and the entity's primary + full NAICS code
   string. Filter by NAICS to the vertical; record the exact `YYYYMMDD` snapshot.
   **Caveat:** SAM only contains entities registered for federal awards — a
   *partial/pooled* universe, so it is a reference **subset**, never a census of
   the vertical. (Source: open.gsa.gov SAM entity-extract V2 public layout.)
3. **Denominator context → Census County Business Patterns (CBP).** CBP publishes
   sample-free establishment **counts** by NAICS × geography (national/state/
   county/MSA/ZIP/congressional-district) from the Business Register, via the
   Census API (free key) + bulk download + data.gov; pair with **Nonemployer
   Statistics** for the sole-proprietor SMB tail CBP excludes. Use CBP only to
   *contextualize* coverage ("the reference set covers N of ~M establishments in
   NAICS X"), **not** as a universe to score absolute recall against. **Caveats:**
   CBP excludes nonemployers and applies disclosure noise/suppression, so fine
   cells are perturbed estimates. (Source: Census CBP methodology + `cbp-api`.)
4. **Recall framing stays honest.** As everywhere here, recall for these seeds is
   **recall against the frozen reference *subset*, not absolute recall** — the
   SAM-derived list is a pooled directory, not the whole market.
   > **Open question (NOT adopted).** Estimating recall against the *full* NAICS
   > universe from a sampled gold (TREC-style pooling; infAP / xinfAP / bpref /
   > statAP sampling estimators) is unresolved — a literature pass could not
   > confirm an appropriate estimator for our setting. Until it does, we publish
   > only recall-vs-reference-subset and label it as such. Flagged as future
   > research, not method.
5. **Entity resolution for domainless SMB/local/franchise.** Many local firms
   have no clean canonical domain (DBAs, franchises, multi-location chains,
   shared corporate domains). SAM's legal name + DBA + structured address are the
   match keys; franchise / multi-location / shared-domain false matches are a
   known risk, and the matcher's `method ∈ {domain, alias, name}` audit trail
   (§6) records how each hit was made. The matching methodology for these is not
   yet validated → future work.

**Future work.** Add trades/real-estate seeds; build a `goldset/naics.py`
harvester (SAM.gov NAICS-filtered named entities + CBP counts) analogous to
`g2_rapidapi.py`; until then the manual `.manual.json` path is the route. Public
sources named but not yet evaluated: SBA, state Secretary-of-State registries,
OpenCorporates, USAspending.

**References:** NAICS — `census.gov/naics` + 2022 NAICS Manual; CBP — Census CBP
methodology + `cbp-api`; SAM — open.gsa.gov SAM entity-extract V2 public layout.

---

## 6. Matching: did a vendor "find" a gold company?

Matching is the mechanical core; ambiguity here is an attack surface, so it is
fully specified and implemented once in `recall.py`.

1. **Primary key = canonical registrable domain.** Both gold domains and vendor
   result domains are normalized: lowercase, strip scheme/`www.`/path/port,
   reduce to the registrable domain (eTLD+1, with a known multi-part suffix
   list for `co.uk` etc.).
2. **Aliases.** A gold company may carry **multiple known domains**
   (`brex.com`, `getbrex.com`). A vendor result counts as a hit if its
   canonical domain matches **any** alias.
3. **Name fallback (flagged).** Only when a vendor returns **no domain** for a
   result do we fall back to normalized-name exact match (lowercased,
   punctuation/suffix-stripped: "Inc", "LLC", "Ltd"). Name-matched hits are
   counted but **flagged** in the audit trail, because name matching is
   collision-prone. Vendors that return domains are never name-matched.
4. **One gold company is credited once.** Duplicate vendor rows resolving to the
   same gold company count as a single hit (and the duplication is recorded as a
   list-hygiene signal).

The match function is pure and deterministic; the same inputs always yield the
same hits, and the matched (rank → gold company) mapping is persisted per cell
so any reviewer can re-derive every metric by hand.

---

## 7. Fairness audit (published alongside results)

We don't *assert* the reference set is unbiased — we **measure and publish**
evidence:

1. **Findability distribution.** For each gold company, how many distinct
   vendors found it. A reference set secretly biased toward OpenFunnel would
   show competitors systematically failing to find gold companies. Publishing
   "X% of gold companies were found by ≥2 vendors" is direct evidence of
   fairness.
2. **audit-vendor-only gold companies.** Count of gold companies found *only* by
   OpenFunnel. If the set were an OpenFunnel index dump, this would be high; we
   expect it near zero and surface it transparently (a few are legitimate, a
   spike is a red flag we own).
3. **Per-vendor unique contribution.** For each vendor, gold companies it found
   that no other vendor did — context for interpreting recall leads.
4. **Source provenance per company.** Every gold company links to its sources so
   any reader can verify it wasn't hand-inserted.

These are computed from the same matched-hit data as the metrics — no extra
labeling.

---

## 8. Reproducibility artifacts

Inherited from the existing harness and extended:

- **Raw vendor calls** (`*.raw.json`) — every HTTP request/response, auth
  redacted.
- **Raw judge calls** — exact prompt + model output, so anyone can replay the
  judge against their own LLM.
- **Matched-hit map per cell** — rank → gold company + match method
  (domain/alias/name), so every recall number is hand-checkable.
- **Frozen gold set** — content-hashed and dated (§9).

An agent reading our `llms.txt` should be able to **re-derive** the leaderboard,
not just read it.

---

## 9. Versioning & freeze

- The gold set is committed to git as data with a **content hash** and an
  **`as_of` date** *before* any vendor run. The git commit timestamp is the
  proof the truth set wasn't retrofitted after seeing results.
- Each benchmark snapshot records the **gold-set version/hash it ran against**.
- **Refresh cadence:** gold sets are re-derived on a published schedule
  (firmographic rosters drift as companies die / are acquired). Each refresh is
  a new version with a changelog; old versions remain in git for longitudinal
  comparison.
- Because the denominator is vendor-independent, **adding a new vendor does not
  change any gold set or any historical recall number** — the new vendor simply
  runs against the existing frozen yardstick.

---

## 10. Known limitations (stated, not hidden)

1. Reference-set recall is **not absolute recall**; it measures coverage of a
   curated yardstick (§1.1).
2. Trade league tables rank by **size**, not similarity, so `trades`/
   `real-estate` gold sets skew toward larger operators; NAICS enumeration is
   used to add the long tail where feasible.
3. Source coverage is **uneven across categories** (software is well-covered by
   Gartner/G2; some service verticals less so). Per-category gold-set sizes are
   published so readers can weight accordingly.
4. The benchmark measures **list quality as a leading proxy** for the outcomes
   operators ultimately care about (win-rate-by-cohort, CAC, retention/LTV),
   which require CRM feedback loops outside an API benchmark's scope.
