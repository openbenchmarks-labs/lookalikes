"""Exa quirk hooks."""
from __future__ import annotations

from typing import Any

from ..common import Candidate


def company_query(ctx: dict[str, Any]) -> None:
    """build_request: use Exa's company vertical natural-language pattern."""
    k = ctx["k"]
    seed = ctx["seed"]
    config = ctx["config"]
    ctx["vars"]["num_results"] = min(k * int(config.get("over_fetch") or 1), 100)
    ctx["vars"]["query"] = f"companies like {seed.seed_name}"


def _company_entity(item: dict[str, Any]) -> dict[str, Any] | None:
    for entity in item.get("entities") or []:
        if isinstance(entity, dict) and entity.get("type") == "company":
            props = entity.get("properties")
            return props if isinstance(props, dict) else {}
    return None


def dedupe_by_domain(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: collapse page hits to one candidate per domain
    (highest score), then sort by score DESC."""
    by_domain: dict[str, dict[str, Any]] = {}
    for r in raw_items:
        if not isinstance(r, dict):
            continue
        u = r.get("url") or ""
        if not isinstance(u, str) or not u:
            continue
        dom = u.replace("https://", "").replace("http://", "").split("/")[0].lower()
        if not dom:
            continue
        existing = by_domain.get(dom)
        if existing is None or float(r.get("score") or 0) > float(existing.get("score") or 0):
            by_domain[dom] = r

    candidates: list[Candidate] = []
    for dom, r in by_domain.items():
        company = _company_entity(r) or {}
        title = company.get("name") or r.get("title") or r.get("author") or dom
        text = (
            company.get("description")
            or r.get("text")
            or ((r.get("highlights") or [None])[0] if r.get("highlights") else None)
        )
        candidates.append(
            Candidate(
                name=str(title).strip(),
                domain=dom,
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
