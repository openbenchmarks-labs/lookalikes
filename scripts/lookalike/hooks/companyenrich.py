"""CompanyEnrich quirk hook.

`POST /v2/companies/search` with a `lookalike.domains` seed returns a flat
`items` list whose fields map declaratively (see specs/companyenrich.yaml), but
the similarity score is not on the item: it lives in a top-level `scores` map
keyed by company id. This hook runs the declarative item map unchanged and then
joins that score in as `extra.similarity`, so the audit trail carries the
vendor's own ranking signal the way Discolike's `similarity` and Ocean's
`relevance_score` do. Order is preserved — items arrive in similarity rank and
the generic runner assigns rank by position after seed-removal and dedupe.
"""
from __future__ import annotations

from typing import Any

from ..common import Candidate


def attach_scores(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: declarative item map + `scores[id]` join."""
    # Local import: generic_runner imports the hook registry at module load, so
    # a top-level import here would be circular.
    from ..generic_runner import _map_item

    payload = ctx.get("payload")
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        scores = {}
    item_map = ctx["spec"].response.item

    candidates: list[Candidate] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        candidate = _map_item(item, item_map)
        candidate.extra["similarity"] = scores.get(item.get("id"))
        candidates.append(candidate)
    return candidates
