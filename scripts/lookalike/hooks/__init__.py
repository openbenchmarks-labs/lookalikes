"""Vendor quirk hooks for the generic spec runner.

A spec references hooks by name (string); YAML never imports Python. `HOOKS`
maps those names to callables. Unknown names fail loudly at spec load.

Hook stages (all optional per vendor):
  - preflight(ctx) -> None        runs first, before templating; may make HTTP
                                  calls (captured); mutates ctx["vars"] and
                                  ctx["vars"]["_audit"].
  - build_request(ctx) -> None    runs before templating; pure var computation
                                  (objective, num_results, conditional query).
  - transform_candidates(raw_items, ctx) -> list[Candidate]
                                  replaces the declarative field-map when the
                                  vendor needs cross-referencing / dedupe / sort.
                                  ctx["payload"] holds the full parsed response.
"""
from __future__ import annotations

from typing import Any, Callable

from . import exa, ocean, openfunnel, parallel, predictleads

HOOKS: dict[str, Callable[..., Any]] = {
    # OpenFunnel
    "openfunnel_canonicalize_domain": openfunnel.canonicalize_domain,
    "openfunnel_query_for_config": openfunnel.query_for_config,
    # Exa
    "exa_num_results": exa.num_results,
    "exa_dedupe_by_domain": exa.dedupe_by_domain,
    # Ocean
    "ocean_firmographic_filters": ocean.firmographic_filters,
    # Parallel
    "parallel_objective": parallel.objective,
    "parallel_clean_aggregators": parallel.clean_aggregators,
    # PredictLeads
    "predictleads_paging": predictleads.paging,
    "predictleads_jsonapi_resolve": predictleads.jsonapi_resolve,
}
