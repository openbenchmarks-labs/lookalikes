"""Exa quirk hooks.

Exa is a web-page index, not a company index, so a single domain returns
multiple page hits. `num_results` over-fetches; `dedupe_by_domain` collapses
to one candidate per domain (highest score), extracts the company domain from
the page URL, and sorts by score. Ported verbatim from runners/exa.py.
"""
from __future__ import annotations

from typing import Any

from ..common import Candidate


def num_results(ctx: dict[str, Any]) -> None:
    """build_request: compute the over-fetched numResults before templating."""
    k = ctx["k"]
    config = ctx["config"]
    ctx["vars"]["num_results"] = min(k * int(config.get("over_fetch") or 1), 100)


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
        title = r.get("title") or r.get("author") or dom
        text = r.get("text") or (r.get("highlights") or [None])[0] if r.get("highlights") else r.get("text")
        candidates.append(
            Candidate(
                name=str(title).strip(),
                domain=dom,
                description=(text or "")[:400] if isinstance(text, str) else None,
                extra={"score": r.get("score"), "url": r.get("url")},
            )
        )
    candidates.sort(key=lambda c: -float(c.extra.get("score") or 0))
    return candidates
