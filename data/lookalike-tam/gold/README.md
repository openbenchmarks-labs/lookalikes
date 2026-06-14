# TAM-recall gold sets

Frozen, **vendor-independent** reference sets — one JSON file per seed,
`<seed_slug>.gold.json`. These are the denominator for the recall metrics.
See `scripts/lookalike/RECALL_METHODOLOGY.md` for the full construction rules.

These files are **data, committed to git**. Their git history + `version_hash`
are the proof the truth set was frozen *before* any vendor ran (methodology §9).

## Schema

```jsonc
{
  "seed_slug": "ramp",            // matches a seed in latest-lookalike.json
  "seed_name": "Ramp",
  "category": "b2b-saas",         // LookalikeCategory
  "as_of": "2026-06-08",          // when the rosters were captured
  "filters": {},                  // optional firmographic predicate this set is scoped to
  "version_hash": "sha256:...",   // content hash, set by the freeze step; verified on load
  "sources_used": [               // provenance of the rosters harvested (for the page)
    { "id": "g2:spend-management", "tier": "B", "url": "https://www.g2.com/categories/spend-management", "captured_at": "2026-06-08" },
    { "id": "crunchbase:ramp/competitors", "tier": "B", "url": "https://www.crunchbase.com/organization/ramp/...", "captured_at": "2026-06-08" }
  ],
  "companies": [
    {
      "name": "Brex",
      "domains": ["brex.com", "getbrex.com"],   // canonical aliases; a hit on ANY counts
      "sources": ["g2:spend-management", "crunchbase:ramp/competitors"],  // >=1 Tier-A OR >=2 Tier-B
      "attrs": { "region": "AMER", "employee_band": "1000-5000" }         // optional; used for filtered cases
    }
  ]
}
```

### Field rules (enforced by the build/freeze step)

- `companies[].sources` must satisfy the inclusion rule: **≥1 Tier-A source OR
  ≥2 distinct Tier-B sources** (methodology §5.2). The builder rejects entries
  that don't.
- `domains` are stored as published; canonicalization (eTLD+1, strip `www`) is
  applied at match time by `recall.py` — don't pre-mangle them.
- The seed itself, its parent, and subsidiaries must **not** appear (§5.3).
- `version_hash` is computed by `goldset/build.py`; `recall.load_goldset()`
  re-verifies it and refuses to load an edited-but-unfrozen file.

## Workflow

```bash
# 1. harvest rosters into a raw working file (per-category adapters)
python3 scripts/lookalike/goldset/build.py harvest --seed ramp --category b2b-saas

# 2. (human) QA pass — drop junk only, never add outside sourced rosters

# 3. freeze: apply inclusion rule, compute hash, write <seed>.gold.json
python3 scripts/lookalike/goldset/build.py freeze --seed ramp

# 4. verify everything loads + hashes match
python3 scripts/lookalike/goldset/build.py verify
```

## How the shipped seeds were built

The four current seeds (`pylon`, `postscript`, `recharge`, `servicetitan`) are
**software/B2B categories**, so the **G2 category roster is the Tier-A source**
and the build is automated end-to-end:

```bash
# G2 roster (RapidAPI) -> domain resolution (G2 website, Clearbit fallback)
# -> Wikidata/Wikipedia corroboration -> human drop-only QA -> freeze + hash
python3 scripts/lookalike/goldset/build_seeds.py --seeds pylon
```

Each `<seed>.g2.json` is the raw G2 harvest layer (committed here as provenance +
mirrored to the public OSS repo); `<seed>.gold.json` is the frozen, hash-verified
result. The `_g2cache/` / `_raw/` API caches are local-only (gitignored). Full
step-by-step procedure: **`scripts/lookalike/RECALL_METHODOLOGY.md` §5.5**.

**Traditional / non-software verticals** (trades, real-estate, local services)
have no G2 category. The `naics` Tier-B source for those is **specified** (Census
NAICS codes + SAM.gov named entities + Census CBP denominator) in
**RECALL_METHODOLOGY.md §5.6** but not yet built — no traditional-industry seed
ships today.
