"""Parallel quirk hooks.

`objective` builds the natural-language entity-search objective from the seed +
mode (broad/strict); for `use_filters` configs it folds the seed's public
firmographic hints (location / employee range / funding stage) into
the objective text (Parallel has no structured firmographic filter), and raises
`SkipConfig` when the seed carries no hints. `clean_aggregators` strips `www.`,
routes aggregator hosts (LinkedIn, Crunchbase, …) out of the `domain` field, and
handles the optional `entity` sub-object wrapper. Ported verbatim from
runners/parallel.py.
"""
from __future__ import annotations

from typing import Any

from ..common import Candidate, Seed, SkipConfig

AGGREGATOR_HOSTS = {
    "linkedin.com",
    "tracxn.com",
    "platform.tracxn.com",
    "crunchbase.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "bloomberg.com",
    "pitchbook.com",
}


# Human-readable country labels for the firmographic objective clause.
_COUNTRY_NAMES = {"us": "the United States", "gb": "the United Kingdom", "ca": "Canada"}


def _firmographic_clause(firmo: dict[str, Any]) -> str:
    parts: list[str] = []
    locs = firmo.get("locations") or []
    if locs:
        named = ", ".join(_COUNTRY_NAMES.get(str(c).lower(), str(c).upper()) for c in locs)
        parts.append(f"headquartered in {named}")
    lo, hi = firmo.get("min_employees"), firmo.get("max_employees")
    if lo is not None and hi is not None:
        parts.append(f"with roughly {lo}–{hi} employees")
    elif hi is not None:
        parts.append(f"with up to {hi} employees")
    stages = firmo.get("funding_stages") or []
    if stages:
        parts.append(f"at funding stage {', '.join(str(s) for s in stages)}")
    if not parts:
        return ""
    return " Restrict to companies " + "; ".join(parts) + "."


def _build_objective(seed: Seed, k: int, mode: str, firmo: dict[str, Any] | None = None) -> str:
    seed_id = (
        f"{seed.seed_name} ({seed.seed_domain})"
        if seed.seed_domain
        else seed.seed_name
    )
    desc_blob = f" — {seed.description}" if seed.description else ""
    firmo_blob = _firmographic_clause(firmo) if firmo else ""
    if mode == "strict":
        return (
            f"Find the top {k} direct competitors of {seed_id}{desc_blob}. "
            "Direct competitors only: same product category, overlapping "
            "buyer persona, comparable scale. Exclude parent companies, "
            "customers, agencies, consultancies, integrators, and "
            "downstream resellers." + firmo_blob + " Return primary company "
            "name and domain for each match."
        )
    return (
        f"Find the top {k} B2B companies most similar to {seed_id}{desc_blob}. "
        "Similar = same business model, overlapping target customer, "
        "comparable scale. Prefer product / SaaS companies over agencies "
        "or services firms." + firmo_blob + " Return primary company name and "
        "domain for each match."
    )


def objective(ctx: dict[str, Any]) -> None:
    """build_request: set the `query` (objective) var before templating. For
    `use_filters` configs, fold the seed's firmographic hints into the objective
    text (skip cleanly when the seed has none)."""
    seed: Seed = ctx["seed"]
    config = ctx["config"]
    firmo = seed.firmographics
    if config.get("use_filters") and not firmo:
        raise SkipConfig("parallel: filtered config skipped (no firmographic hints for seed)")
    ctx["vars"]["query"] = _build_objective(
        seed, ctx["k"], config.get("mode", "broad"), firmo if config.get("use_filters") else None
    )


def clean_aggregators(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: map entities to candidates, keeping aggregator
    pages out of the company `domain` field."""
    candidates: list[Candidate] = []
    for r in raw_items:
        if not isinstance(r, dict):
            continue
        ent = r.get("entity") if isinstance(r.get("entity"), dict) else r
        name = ent.get("name") or ent.get("entity_name") or ent.get("title") or ""
        raw_url = ent.get("url") or ent.get("website") or ent.get("domain")
        domain: str | None = None
        linkedin_url: str | None = None
        if isinstance(raw_url, str) and raw_url:
            stripped = (
                raw_url.replace("https://", "")
                .replace("http://", "")
                .split("/")[0]
                .lower()
            )
            stripped = stripped[4:] if stripped.startswith("www.") else stripped
            host = stripped
            if any(host == h or host.endswith("." + h) for h in AGGREGATOR_HOSTS):
                if "linkedin.com" in host:
                    linkedin_url = raw_url
                # else: keep raw_url in extra.source_url below
            else:
                domain = stripped
        candidates.append(
            Candidate(
                name=str(name).strip(),
                domain=domain,
                description=(
                    ent.get("description")
                    or ent.get("snippet")
                    or ent.get("summary")
                    or r.get("description")
                ),
                extra={
                    "linkedin_url": linkedin_url,
                    "source_url": raw_url if not domain and not linkedin_url else None,
                    "source_urls": r.get("source_urls") or r.get("citations"),
                    "confidence": r.get("confidence") or r.get("score"),
                },
            )
        )
    return candidates
