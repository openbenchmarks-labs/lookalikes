"""CompanyEnrich quirk hooks.

`request` (build_request) caps pageSize at the documented 100 and, for the
`seed_firmographic` config, maps the seed's public, TAM-derived hints to
CompanyEnrich's filter shape (`countries` as ISO alpha-2 codes, `employees` as
its discrete bands). It raises `SkipConfig` when a `use_seed_firmo` config lands
on a seed with no hints.

`scored_items` (transform_candidates) maps each company to a Candidate and
attaches its similarity score, which the API returns out of band in `scores`
keyed by company id (v1 nested it under `metadata.scores`).
"""
from __future__ import annotations

from typing import Any

from ..common import Candidate, Seed, SkipConfig

MAX_PAGE_SIZE = 100

# CompanyEnrich's `employees` enum. A seed's [min,max] employee hint maps to
# every overlapping band.
_BANDS: list[tuple[int, int, str]] = [
    (1, 10, "1-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1K"),
    (1001, 5000, "1K-5K"),
    (5001, 10000, "5K-10K"),
    (10001, 10**9, "over-10K"),
]

# Hints are free-form ("us", "USA"); the API wants ISO 3166-1 alpha-2.
_COUNTRY_ALIASES = {"usa": "US", "uk": "GB"}


def _size_bands(min_emp: int | None, max_emp: int | None) -> list[str]:
    lo = min_emp if min_emp is not None else 0
    hi = max_emp if max_emp is not None else 10**9
    return [label for blo, bhi, label in _BANDS if blo <= hi and bhi >= lo]


def _country_codes(locations: list[Any]) -> list[str]:
    codes: list[str] = []
    for loc in locations:
        raw = str(loc).strip()
        code = _COUNTRY_ALIASES.get(raw.casefold(), raw.upper())
        if len(code) == 2 and code.isalpha() and code not in codes:
            codes.append(code)
    return codes


def request(ctx: dict[str, Any]) -> None:
    """build_request: page size, plus firmographic filter vars for the
    `use_seed_firmo` config (skip cleanly when the seed has none)."""
    vars = ctx["vars"]
    vars["page_size"] = max(1, min(ctx["k"], MAX_PAGE_SIZE))
    vars["countries"] = None
    vars["employees"] = None
    if not ctx["config"].get("use_seed_firmo"):
        return
    seed: Seed = ctx["seed"]
    firmo = seed.firmographics
    if not firmo:
        raise SkipConfig("companyenrich: firmographic config skipped (no hints for seed)")
    vars["employees"] = _size_bands(firmo.get("min_employees"), firmo.get("max_employees")) or None
    vars["countries"] = _country_codes(firmo.get("locations") or []) or None


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def scored_items(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: company -> Candidate, in API order, with the
    similarity score from `scores[id]`."""
    payload = _obj(ctx.get("payload"))
    scores = payload.get("scores") or _obj(payload.get("metadata")).get("scores") or {}
    out: list[Candidate] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        location = _obj(item.get("location"))
        socials = _obj(item.get("socials"))
        out.append(
            Candidate(
                name=str(item.get("name") or "").strip(),
                domain=item.get("domain") or item.get("website"),
                description=item.get("description"),
                extra={
                    "similarity_score": scores.get(item.get("id")),
                    "companyenrich_id": item.get("id"),
                    "country": _obj(location.get("country")).get("code"),
                    "industry": item.get("industry"),
                    "industries": item.get("industries"),
                    "categories": item.get("categories"),
                    "employees": item.get("employees"),
                    "revenue": item.get("revenue"),
                    "founded_year": item.get("foundedYear"),
                    "linkedin_url": socials.get("linkedinUrl"),
                },
            )
        )
    return out
