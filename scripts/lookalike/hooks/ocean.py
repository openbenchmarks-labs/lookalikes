"""Ocean.io quirk hook (single build_request, two independent concerns).

Both concerns map the seed's public input into Ocean's `companiesFilters` and
stash the result in template vars the spec body references and prunes when null.
A config may opt into either, both, or neither:

1. `use_seed_firmo` — maps the seed's public firmographic hints (employee range
   + locations) to `companySizes` bands + `primaryLocations.includeCountries`
   (vars `company_sizes` / `primary_locations`). Parity with OpenFunnel's
   *_filtered configs. Raises `SkipConfig` when the seed carries no hints.
2. `use_seed_keywords` — derives a `keywords.anyOf` filter from the seed's
   1-line description (var `keywords`). This is the lever the original adapter
   left on the table: Ocean's lookalike endpoint accepts free-text keywords
   alongside `lookalikeDomains`, which disambiguates seeds whose name/domain is
   ambiguous (OpenAI vs agricultural "seed" companies, conglomerates, etc.) —
   the same disambiguation OpenFunnel/Parallel/Exa get from their query string.
   Raises `SkipConfig` when no usable keywords can be extracted.

Ocean's static configs (seed_only / mid_market / us_uk_de_funded / broad_match)
layer their filters via the spec's `merge` directive and scalar vars, so this
hook is a no-op for them. Employee-band mapping is ported verbatim from the
pre-refactor runners/ocean.py.
"""
from __future__ import annotations

import re
from typing import Any

from ..common import Seed, SkipConfig

# Ocean's discrete employee-count bands. A seed's [min,max] employee hint maps
# to every overlapping band.
_OCEAN_BANDS: list[tuple[int, int, str]] = [
    (0, 1, "0-1"),
    (2, 10, "2-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1000"),
    (1001, 5000, "1001-5000"),
    (5001, 10000, "5001-10000"),
    (10001, 50000, "10001-50000"),
    (50001, 100000, "50001-100000"),
    (100001, 500000, "100001-500000"),
    (500001, 10**9, "500000+"),
]


def _size_bands(min_emp: int | None, max_emp: int | None) -> list[str]:
    lo = min_emp if min_emp is not None else 0
    hi = max_emp if max_emp is not None else 10**9
    return [label for blo, bhi, label in _OCEAN_BANDS if blo <= hi and bhi >= lo]


# Description → keyword extraction. ponytail: naive split on common delimiters +
# leading-filler trim, capped at 8 phrases. Ceiling: no POS tagging / phrase
# ranking, so a long clause can land verbatim as one keyword. The best-of sweep
# keeps the higher-Precision@K config, so a noisy keyword set can only tie
# seed_only, never lower the cell. Upgrade path: a proper noun-phrase chunker.
_KW_SPLIT = re.compile(r",|/|;|\band\b|&", re.IGNORECASE)
_KW_STRIP_LEAD = re.compile(
    r"^(a|an|the|including|providing|provider of|for)\s+", re.IGNORECASE
)
_KW_DROP = {"etc", "api", "apis", "tooling", "services", "solutions"}


def _keywords_from_description(desc: str | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seg in _KW_SPLIT.split(desc or ""):
        s = _KW_STRIP_LEAD.sub("", seg.strip().strip(".").lower()).strip()
        if len(s) < 3 or s in _KW_DROP or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 8:
            break
    return out


def firmographic_filters(ctx: dict[str, Any]) -> None:
    """build_request: derive Ocean filter vars from the seed's public input for
    the configs that opt in (`use_seed_keywords` and/or `use_seed_firmo`); a
    no-op for static configs. Skips cleanly when the requested signal is absent."""
    config = ctx["config"]
    seed: Seed = ctx["seed"]
    vars = ctx["vars"]

    if config.get("use_seed_keywords"):
        kws = _keywords_from_description(seed.description)
        if not kws:
            raise SkipConfig("ocean: keyword config skipped (no keywords for seed)")
        vars["keywords"] = {"anyOf": kws}

    if config.get("use_seed_firmo"):
        firmo = seed.firmographics
        if not firmo:
            raise SkipConfig("ocean: firmographic config skipped (no hints for seed)")
        bands = _size_bands(firmo.get("min_employees"), firmo.get("max_employees"))
        vars["company_sizes"] = bands or None
        locs = firmo.get("locations") or []
        vars["primary_locations"] = (
            {"includeCountries": [str(c).lower() for c in locs]} if locs else None
        )
