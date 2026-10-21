"""Discolike quirk hooks."""
from __future__ import annotations

from typing import Any

from ..common import SkipConfig


def build_request(ctx: dict[str, Any]) -> None:
    """build_request: route the seed's inputs the way Discolike recommends.

    Domain + authored query -> domain lookalike steered by icp_prompt (the
    query is the identical string Exa and Parallel receive). Query only ->
    icp_prompt alone. Domain only -> domain lookalike, with and without
    auto_icp_text. Over-fetch so the cross-vendor dedupe and seed removal
    still leave K rows."""
    config = ctx["config"]
    vars = ctx["vars"]
    seed = ctx["seed"]
    query = (seed.query or "").strip() or None
    if config.get("auto_icp_text") and (query or not seed.seed_domain):
        raise SkipConfig("auto_icp_text only applies to domain-only seeds")
    vars["max_records"] = int(ctx["k"] * float(config.get("over_fetch") or 1))
    vars["icp_prompt"] = query
    vars["auto_icp_text"] = "true" if config.get("auto_icp_text") else None
