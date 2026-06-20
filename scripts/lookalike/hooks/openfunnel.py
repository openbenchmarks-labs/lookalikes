"""OpenFunnel quirk hooks for the generic spec runner.

Two quirks that can't be declarative:
  - `canonicalize_domain` (preflight): the seed's marketing domain may not be
    OpenFunnel's canonical index domain (e.g. recharge.com → getrecharge.com).
    We resolve it via the free `lookup-companies` helper, memoized per process
    so the config sweep only pays for one lookup per seed. Also raises
    `SkipConfig` for a `use_filters` config on a seed with no firmographic hints
    (so the recall sweep doesn't duplicate the unfiltered variant).
  - `query_for_config` (build_request): the `query` param is conditional on the
    config's `use_query` flag, with a name-based fallback string; it also
    promotes the seed's public firmographic hints into querystring vars for
    `use_filters` configs (locations / employee range / funding stage).

Ported verbatim from the pre-refactor runners/openfunnel.py so behaviour is
byte-identical.
"""
from __future__ import annotations

import re
from typing import Any

from ..common import Seed, SkipConfig, http_request

BASE_URL = "https://api.openfunnel.dev"
LOOKUP_ENDPOINT = "/api/v1/account/lookup-companies"

# Per-process cache of seed → canonical OpenFunnel domain. Keyed by the
# (seed_domain, seed_name) pair; one lookup per seed across the whole sweep.
_DOMAIN_CACHE: dict[tuple[str, str], str | None] = {}


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _employee_score(match: dict[str, Any]) -> int:
    hi = match.get("employee_count_max")
    lo = match.get("employee_count_min")
    for val in (hi, lo):
        if isinstance(val, int):
            return val
    return 0


def _match_score(seed: Seed, match: dict[str, Any], input_idx: int) -> tuple[int, int, int, int]:
    seed_name = _norm(seed.seed_name)
    match_name = _norm(match.get("name"))
    seed_domain = (seed.seed_domain or "").lower()
    match_domain = (match.get("domain") or "").lower()
    matched_on = match.get("matched_on")

    exact_name = 1 if seed_name and match_name == seed_name else 0
    contains_name = 1 if seed_name and (seed_name in match_name or match_name in seed_name) else 0
    domain_match = 1 if seed_domain and match_domain == seed_domain else 0
    # Prefer fuzzy/name input over broad domain aliases when both are available:
    # domain-only lookup can return many tiny regional pages sharing the same
    # website; the name path is usually closer to the intended corporate seed.
    name_input = 1 if matched_on == "name" or input_idx > 0 else 0
    return (
        exact_name,
        contains_name,
        name_input,
        _employee_score(match),
        domain_match,
    )


def _select_lookup_match(seed: Seed, payload: dict[str, Any]) -> dict[str, Any] | None:
    scored: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    for input_idx, result in enumerate(payload.get("results", [])):
        for match in result.get("matches") or []:
            if not isinstance(match, dict):
                continue
            if (match.get("domain") or "").strip():
                scored.append((_match_score(seed, match, input_idx), match))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def _resolve_canonical_domain(seed: Seed, api_key: str) -> str | None:
    key = (seed.seed_domain or "", seed.seed_name or "")
    if key in _DOMAIN_CACHE:
        return _DOMAIN_CACHE[key]

    items: list[dict[str, Any]] = []
    if seed.seed_domain:
        items.append({"domain": seed.seed_domain})
    if seed.seed_name:
        items.append({"name": seed.seed_name})
    if not items:
        _DOMAIN_CACHE[key] = None
        return None

    status, payload, _ = http_request(
        "POST",
        f"{BASE_URL}{LOOKUP_ENDPOINT}",
        headers={"X-API-Key": api_key},
        body={"companies": items},
    )
    canonical: str | None = None
    if status < 300 and isinstance(payload, dict):
        match = _select_lookup_match(seed, payload)
        canonical = (match.get("domain") or "").strip() if match else None

    _DOMAIN_CACHE[key] = canonical
    return canonical


def canonicalize_domain(ctx: dict[str, Any]) -> None:
    """preflight: resolve the canonical domain and stash the audit value.

    `seed_domain` var becomes the *effective* domain (canonical or original);
    the raw canonical (which may be None) is recorded for `config.resolved_domain`
    on both success and error paths.

    Skips early (before any lookup) when a `use_filters` config lands on a seed
    with no firmographic hints — mirrors the pre-refactor runner so the recall
    sweep records a clean skip instead of duplicating the unfiltered variant."""
    seed: Seed = ctx["seed"]
    if ctx["config"].get("use_filters") and not seed.firmographics:
        raise SkipConfig("openfunnel: filtered config skipped (no firmographic hints for seed)")
    if ctx["config"].get("query_only"):
        ctx["vars"]["seed_domain"] = None
        ctx["vars"]["_audit"]["resolved_domain"] = None
        return
    api_key = ctx["env"]["OPENFUNNEL_API_KEY"]
    canonical = _resolve_canonical_domain(seed, api_key)
    ctx["vars"]["seed_domain"] = canonical or seed.seed_domain
    ctx["vars"]["_audit"]["resolved_domain"] = canonical


def query_for_config(ctx: dict[str, Any]) -> None:
    """build_request: set the conditional `query` var before templating so the
    querystring param order matches the original runner. For `use_filters`
    configs, also promote the seed's public firmographic hints into the
    `locations` / `min_employees` / `max_employees` / `funding_stages` vars
    (array vars repeat per param via doseq; absent vars are pruned)."""
    seed: Seed = ctx["seed"]
    config = ctx["config"]
    vars = ctx["vars"]
    if config.get("query_only"):
        desc = (seed.description or "").strip()
        base = f"Companies most similar to {seed.seed_name}"
        if desc:
            query = f"{base}: {desc}"
        else:
            query = base
        vars["query"] = query[:200]
    elif config.get("use_query") and seed.description:
        vars["query"] = seed.description
    elif config.get("use_query"):
        vars["query"] = f"Companies similar to {seed.seed_name}"
    else:
        vars["query"] = None

    if config.get("use_filters"):
        firmo = seed.firmographics or {}
        locations = [str(x) for x in (firmo.get("locations") or [])]
        funding_stages = [str(x) for x in (firmo.get("funding_stages") or [])]
        vars["locations"] = locations or None
        vars["funding_stages"] = funding_stages or None
        mn, mx = firmo.get("min_employees"), firmo.get("max_employees")
        vars["min_employees"] = str(mn) if mn is not None else None
        vars["max_employees"] = str(mx) if mx is not None else None
