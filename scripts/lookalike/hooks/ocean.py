"""Ocean.io quirk hook.

`firmographic_filters` (build_request) handles the `seed_firmographic` config.
It maps the seed's public firmographic hints (employee range + locations) to Ocean's `companiesFilters`
shape (`companySizes` bands + `primaryLocations.includeCountries`) and stashes
them in template vars (`company_sizes` / `primary_locations`) that the spec body
references and prunes when null. It raises `SkipConfig` when a `use_seed_firmo`
config lands on a seed with no hints.

Ocean's static configs (seed_only / mid_market / us_uk_de_funded) layer their
filters via the spec's `merge` directive (`config.filters → companiesFilters`),
so this hook is a no-op for them. Employee-band mapping is ported verbatim from
the pre-refactor runners/ocean.py.
"""
from __future__ import annotations

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


def firmographic_filters(ctx: dict[str, Any]) -> None:
    """build_request: for `use_seed_firmo` configs, derive Ocean filter vars from
    the seed's firmographic hints (skip cleanly when the seed has none)."""
    config = ctx["config"]
    if not config.get("use_seed_firmo"):
        return  # static configs layer filters via the spec `merge` directive
    seed: Seed = ctx["seed"]
    firmo = seed.firmographics
    if not firmo:
        raise SkipConfig("ocean: firmographic config skipped (no hints for seed)")
    vars = ctx["vars"]
    bands = _size_bands(firmo.get("min_employees"), firmo.get("max_employees"))
    vars["company_sizes"] = bands or None
    locs = firmo.get("locations") or []
    vars["primary_locations"] = (
        {"includeCountries": [str(c).lower() for c in locs]} if locs else None
    )
