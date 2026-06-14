"""PredictLeads quirk hooks.

`paging` sets the optional page/per_page query vars. `jsonapi_resolve` joins the
JSON:API `data[]` similarity rows to the `included[]` company entities, drops the
seed self-echo, and carries the vendor's explicit `position` as the rank — and,
for the `paginated` config, follows additional pages to reach the requested k.

Pagination (recall depth): PredictLeads serves a fixed ~20 rows/page and its
similar-companies graph is small (`meta.count` ~25-30). The generic runner makes
the page-1 call; for `use_paging` configs this hook continues from page 2 until
we have k candidates, a page comes back empty, `meta.count` is drained, or a
10-page safety bound is hit — the exact stop conditions of the pre-refactor
runner. (Stop conditions match page-1-satisfies-k for every recorded precision
fixture, so the replay regression stays byte-identical. Additional-page latency
is not folded into latency_ms; recall is a coverage benchmark, not a latency one.)

Note: ranks are intentionally NOT renumbered here. Rank deduplication/renumbering
for the `unique(run_id, rank)` DB constraint happens in the P4 persist layer.
Ported from runners/predictleads.py.
"""
from __future__ import annotations

import urllib.parse
from typing import Any

from ..common import Candidate, Seed, http_request

_MAX_PAGES = 10  # safety bound; meta.count is ~25-30 in practice


def paging(ctx: dict[str, Any]) -> None:
    """build_request: set page/per_page vars only when the config opts in."""
    config = ctx["config"]
    if config.get("use_paging"):
        ctx["vars"]["page"] = 1
        ctx["vars"]["per_page"] = ctx["k"]
    else:
        ctx["vars"]["page"] = None
        ctx["vars"]["per_page"] = None


def _rows_to_candidates(payload: dict[str, Any], seed_domain_lower: str) -> list[Candidate]:
    """Resolve one page's `data[]` similarity rows against its `included[]`."""
    included_by_id: dict[str, dict[str, Any]] = {}
    for inc in payload.get("included") or []:
        if isinstance(inc, dict) and inc.get("type") == "company" and inc.get("id"):
            included_by_id[inc["id"]] = inc.get("attributes") or {}

    candidates: list[Candidate] = []
    for entry in payload.get("data") or []:
        if not isinstance(entry, dict):
            continue
        attrs = entry.get("attributes") or {}
        rels = entry.get("relationships") or {}
        sim = (rels.get("similar_company") or {}).get("data") or {}
        sim_id = sim.get("id")
        co = included_by_id.get(sim_id, {}) if sim_id else {}
        domain = co.get("domain") or co.get("website")
        # Drop the seed itself if echoed back as self-similar.
        if domain and domain.lower() == seed_domain_lower:
            continue
        candidates.append(
            Candidate(
                name=str(co.get("company_name") or co.get("name") or "").strip(),
                domain=domain,
                description=attrs.get("reason"),
                rank=attrs.get("position"),
                extra={
                    "score": attrs.get("score"),
                    "ticker": co.get("ticker"),
                    "refreshed_at": attrs.get("refreshed_at"),
                    "predict_leads_id": sim_id,
                },
            )
        )
    return candidates


def _drained(candidates: list[Candidate], k: int, payload: dict[str, Any]) -> bool:
    """Branch stop conditions (excluding the page bound): reached k, empty page,
    or drained the vendor's reported graph size."""
    total_available = (payload.get("meta") or {}).get("count")
    return (
        not (payload.get("data") or [])
        or len(candidates) >= k
        or (isinstance(total_available, int) and len(candidates) >= total_available)
    )


def jsonapi_resolve(raw_items: list[Any], ctx: dict[str, Any]) -> list[Candidate]:
    """transform_candidates: resolve page 1, then (for `paginated`) follow pages
    2..N until k candidates / empty page / graph drained / 10-page bound."""
    payload: dict[str, Any] = ctx["payload"]
    seed: Seed = ctx["seed"]
    config: dict[str, Any] = ctx["config"]
    seed_domain_lower = (seed.seed_domain or "").lower()

    candidates = _rows_to_candidates(payload, seed_domain_lower)
    if not config.get("use_paging") or _drained(candidates, ctx["k"], payload):
        return candidates

    # Page 1 was the generic runner's call; continue from page 2 with the same
    # headers/path it used (auth from spec, eTLD-encoded seed domain).
    k = ctx["k"]
    spec = ctx["spec"]
    headers = {a.header: ctx["env"][a.env] for a in spec.auth}
    dom_enc = urllib.parse.quote(seed.seed_domain or "", safe="")
    base = f"{spec.base_url}/api/v3/companies/{dom_enc}/similar_companies"
    page = 2
    while page <= _MAX_PAGES:
        status, pl, _ = http_request("GET", f"{base}?page={page}&per_page={k}", headers=headers)
        if status >= 300 or not isinstance(pl, dict):
            break  # keep what we already paginated
        candidates.extend(_rows_to_candidates(pl, seed_domain_lower))
        if _drained(candidates, k, pl):
            break
        page += 1
    return candidates
