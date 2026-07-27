"""Exa quirk hooks."""
from __future__ import annotations

from typing import Any

from ..common import Candidate

AGGREGATOR_HOSTS = {
    "linkedin.com", "tracxn.com", "platform.tracxn.com", "crunchbase.com",
    "wikipedia.org", "en.wikipedia.org", "bloomberg.com", "pitchbook.com",
}


def company_query(ctx: dict[str, Any]) -> None:
    """build_request: form one of the benchmark's lookalike query variants."""
    k = ctx["k"]
    seed = ctx["seed"]
    config = ctx["config"]
    ctx["vars"]["num_results"] = min(k * int(config.get("over_fetch") or 1), 100)
    if config.get("query_variant") == "concise_fallback":
        description = f" — {seed.description}" if seed.description else ""
        ctx["vars"]["query"] = f"companies like {seed.seed_name}{description}"
        return
    description = (seed.description or f"{seed.seed_name} is a company.").strip()
    if description[-1:] not in ".!?":
        description += "."
    prefix = f"Find companies that are lookalikes of {seed.seed_name}: "
    variant = config.get("prompt_variant", "full")
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


def _company_entity(item: dict[str, Any]) -> dict[str, Any] | None:
    for entity in item.get("entities") or []:
        if isinstance(entity, dict) and entity.get("type") == "company":
            props = entity.get("properties")
            return props if isinstance(props, dict) else {}
    return None


def dedupe_by_domain(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: collapse page hits to one candidate per domain
    (highest score), then sort by score DESC."""
    by_domain: dict[str, tuple[dict[str, Any], str | None]] = {}
    for r in raw_items:
        if not isinstance(r, dict):
            continue
        u = r.get("url") or ""
        if not isinstance(u, str) or not u:
            continue
        host = u.replace("https://", "").replace("http://", "").split("/")[0].lower()
        host = host[4:] if host.startswith("www.") else host
        if not host:
            continue
        is_aggregator = any(host == item or host.endswith("." + item) for item in AGGREGATOR_HOSTS)
        domain = None if is_aggregator else host
        key = domain or f"url:{u.lower()}"
        existing = by_domain.get(key)
        if existing is None or float(r.get("score") or 0) > float(existing[0].get("score") or 0):
            by_domain[key] = (r, domain)

    candidates: list[Candidate] = []
    for r, domain in by_domain.values():
        company = _company_entity(r) or {}
        title = company.get("name") or r.get("title") or r.get("author") or domain or r.get("url")
        text = (
            company.get("description")
            or r.get("text")
            or ((r.get("highlights") or [None])[0] if r.get("highlights") else None)
        )
        candidates.append(
            Candidate(
                name=str(title).strip(),
                domain=domain,
                description=(text or "")[:400] if isinstance(text, str) else None,
                extra={
                    "score": r.get("score"),
                    "url": r.get("url"),
                    "entity": company or None,
                },
            )
        )
    candidates.sort(key=lambda c: -float(c.extra.get("score") or 0))
    return candidates
