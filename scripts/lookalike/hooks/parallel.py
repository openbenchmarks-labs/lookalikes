"""Parallel quirk hooks."""
from __future__ import annotations

import re
from typing import Any

from ..common import Candidate, Seed

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


def _embedded_website_domain(description: str | None) -> str | None:
    """Extract a vendor-supplied canonical website from its text payload."""
    if not description:
        return None
    match = re.search(r"\bWebsite\s+Url:\s*(https?://[^\s|]+)", description, re.IGNORECASE)
    if not match:
        return None
    host = match.group(1).rstrip(".,;:)]}").replace("https://", "").replace("http://", "").split("/")[0].lower()
    host = host[4:] if host.startswith("www.") else host
    if any(host == item or host.endswith("." + item) for item in AGGREGATOR_HOSTS):
        return None
    return host or None


def objective(ctx: dict[str, Any]) -> None:
    """Frame lookalikes as companies a shared buyer would evaluate together."""
    seed: Seed = ctx["seed"]
    if ctx["config"].get("query_variant") == "concise_fallback":
        description = f" — {seed.description}" if seed.description else ""
        ctx["vars"]["query"] = f"companies like {seed.seed_name}{description}"
        return
    description = (seed.description or f"{seed.seed_name} is a company.").strip()
    if description[-1:] not in ".!?":
        description += "."
    variant = ctx["config"].get("prompt_variant", "full")
    prefix = f"Find companies that are lookalikes of {seed.seed_name}: "
    if variant == "no_return":
        ctx["vars"]["query"] = f"{prefix}companies with a similar core product and buyer. {description}"
    elif variant == "no_similarity_clause":
        ctx["vars"]["query"] = (
            f"{prefix}{description} Return companies a buyer would realistically evaluate alongside {seed.seed_name}."
        )
    else:
        ctx["vars"]["query"] = (
            f"{prefix}companies with a similar core product and buyer. {description} "
            f"Return companies a buyer would realistically evaluate alongside {seed.seed_name}."
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
        is_aggregator = False
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
                is_aggregator = True
                if "linkedin.com" in host:
                    linkedin_url = raw_url
                # else: keep raw_url in extra.source_url below
            else:
                domain = stripped
        description = (
            ent.get("description")
            or ent.get("snippet")
            or ent.get("summary")
            or r.get("description")
        )
        if not domain:
            domain = _embedded_website_domain(description)
        candidates.append(
            Candidate(
                name=str(name).strip(),
                domain=domain,
                description=description,
                extra={
                    "linkedin_url": linkedin_url,
                    "source_url": raw_url if is_aggregator and not linkedin_url else None,
                    "source_urls": r.get("source_urls") or r.get("citations"),
                    "confidence": r.get("confidence") or r.get("score"),
                },
            )
        )
    return candidates
