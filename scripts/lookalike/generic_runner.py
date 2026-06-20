"""One generic runner that turns a declarative `VendorSpec` + a config into a
`RunResult`. Replaces the six per-vendor `run()` functions; vendor quirks live
in named hooks (`hooks/`). All HTTP goes through `common.http_request` so the
capture + redaction audit trail is unchanged.

Flow:
  1. require_env for the spec's auth keys.
  2. preflight hook (may HTTP; mutates vars) → build_request hook (computes vars).
  3. guard: requires_seed_domain.
  4. template path/query/body from vars; shallow-merge config sub-dicts; prune None.
  5. http_request; on >=300 return an error RunResult (config may carry preflight audit).
  6. resolve candidate list (path fallbacks) → transform hook OR declarative item map.
  7. merge audit values into the returned config; compute cost; take_top.
"""
from __future__ import annotations

import urllib.parse
from typing import Any

from .common import Candidate, RunResult, Seed, SkipConfig, http_request, require_env, take_top
from .hooks import HOOKS
from .spec_loader import ItemMap, VendorSpec


# --------------------------------------------------------------------------- #
# Templating + path resolution                                                #
# --------------------------------------------------------------------------- #


def _base_vars(seed: Seed, k: int, config: dict[str, Any]) -> dict[str, Any]:
    dom = seed.seed_domain
    vars: dict[str, Any] = {
        "seed_domain": dom,
        "seed_domain_enc": urllib.parse.quote(dom, safe="") if dom else None,
        "seed_url": f"https://{dom}" if dom else None,
        "seed_name": seed.seed_name,
        "seed_description": seed.description,
        "k": k,
        "k_min5": max(k, 5),
        "query": None,
    }
    # Expose scalar config knobs as $<key>, without clobbering canonical vars.
    for key, val in config.items():
        if key == "name" or key in vars:
            continue
        vars[key] = val
    vars["_audit"] = {}
    return vars


_PH_FULL = "$"  # marker prefix


def _template(value: Any, vars: dict[str, Any]) -> Any:
    """Recursively substitute $placeholders. A leaf that is exactly "$name"
    returns the raw var value (type-preserving, e.g. int/None/list). A leaf that
    merely contains $name interpolates textually."""
    if isinstance(value, str):
        if value.startswith("$") and _is_ident(value[1:]):
            return vars.get(value[1:])
        return _interpolate(value, vars)
    if isinstance(value, dict):
        return {k: _template(v, vars) for k, v in value.items()}
    if isinstance(value, list):
        return [_template(v, vars) for v in value]
    return value


def _is_ident(s: str) -> bool:
    return bool(s) and (s[0].isalpha() or s[0] == "_") and all(c.isalnum() or c == "_" for c in s)


def _interpolate(s: str, vars: dict[str, Any]) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "$":
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            ident = s[i + 1 : j]
            if ident:
                v = vars.get(ident)
                out.append("" if v is None else str(v))
                i = j
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path (with optional [i] index) against obj. None if any
    segment is missing or the shape is wrong."""
    cur = obj
    for seg in path.split("."):
        key = seg
        idx: int | None = None
        if seg.endswith("]") and "[" in seg:
            key, rest = seg.split("[", 1)
            try:
                idx = int(rest[:-1])
            except ValueError:
                return None
        if key:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
        if idx is not None:
            if isinstance(cur, list) and -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        if cur is None:
            return None
    return cur


def _resolve_ref(item: Any, ref: Any) -> Any:
    """A ref is None, a single path string (plain .get — value as-is), or a list
    of fallback paths (first truthy wins, like the original `a or b` chains)."""
    if ref is None:
        return None
    if isinstance(ref, list):
        for p in ref:
            v = _resolve_path(item, p)
            if v:
                return v
        return None
    return _resolve_path(item, ref)


def _map_item(item: dict[str, Any], item_map: ItemMap) -> Candidate:
    return Candidate(
        name=str(_resolve_ref(item, item_map.name) or "").strip(),
        domain=_resolve_ref(item, item_map.domain),
        description=_resolve_ref(item, item_map.description),
        rank=_resolve_path(item, item_map.rank) if item_map.rank else None,
        extra={key: _resolve_ref(item, ref) for key, ref in item_map.extra.items()},
    )


def _resolve_candidate_list(payload: Any, paths: list[str]) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    for p in paths:  # first non-empty list wins (matches `a or b or []`)
        v = _resolve_path(payload, p)
        if isinstance(v, list) and v:
            return v
    for p in paths:  # else first list at all (possibly empty)
        v = _resolve_path(payload, p)
        if isinstance(v, list):
            return v
    return []


def _prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _prune_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune_none(v) for v in value]
    return value


def _build_out_config(
    config: dict[str, Any],
    spec: VendorSpec,
    audit_vars: dict[str, Any],
    payload: Any,
    *,
    success: bool,
) -> dict[str, Any]:
    out = dict(config)
    for key, src in spec.audit.items():
        if src.startswith("$audit."):
            out[key] = audit_vars.get(src[len("$audit.") :])
        elif success:
            out[key] = _resolve_path(payload, src)
    return out


def _compute_cost(spec: VendorSpec, candidates: list[Candidate]) -> float | None:
    if spec.cost.mode == "none":
        return None
    if spec.cost.mode == "per_call":
        return spec.cost.usd
    return round((spec.cost.usd or 0.0) * len(candidates), 6)  # per_result (reserved)


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


def _err(spec: VendorSpec, seed: Seed, config: dict[str, Any], latency_ms: int,
         message: str, out_config: dict[str, Any] | None = None, requested_k: int | None = None) -> RunResult:
    return RunResult(
        seed_slug=seed.seed_slug,
        provider_slug=spec.slug,
        config_name=config["name"],
        config=out_config if out_config is not None else config,
        candidates=[],
        latency_ms=latency_ms,
        error=message,
        requested_k=requested_k,
    )


def run_from_spec(spec: VendorSpec, seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    env = require_env(*spec.auth_env_keys()) if spec.auth_env_keys() else {}
    vars = _base_vars(seed, k, config)
    ctx: dict[str, Any] = {
        "seed": seed, "k": k, "config": config, "spec": spec, "env": env, "vars": vars,
    }

    try:
        if spec.hooks.preflight:
            HOOKS[spec.hooks.preflight](ctx)
        if spec.hooks.build_request:
            HOOKS[spec.hooks.build_request](ctx)
    except SkipConfig as skip:
        # A firmographic-filter config asked to be skipped for this seed (e.g. no
        # firmographic hints). Record it as a non-fatal skip, no HTTP call made.
        return _err(spec, seed, config, 0, f"skipped: {skip}" if str(skip) else "skipped", requested_k=k)

    if spec.requires_seed_domain and not vars.get("seed_domain"):
        return _err(spec, seed, config, 0,
                    spec.no_domain_error or f"{spec.slug}: seed has no domain", requested_k=k)

    method = spec.request.method.upper()
    url = spec.base_url + str(_template(spec.request.path, vars))
    if spec.request.query:
        q = {kk: vv for kk, vv in _template(spec.request.query, vars).items() if vv is not None}
        if q:
            url = f"{url}?{urllib.parse.urlencode(q, doseq=True)}"

    body: Any = None
    if spec.request.body is not None:
        body = _template(spec.request.body, vars)
        for m in spec.request.merge:
            sub = config.get(m.from_config) or {}
            target = body.setdefault(m.into_body, {})
            if isinstance(target, dict) and isinstance(sub, dict):
                target.update(sub)  # shallow merge, matches the original runner
        body = _prune_none(body)

    headers = {a.header: env[a.env] for a in spec.auth}
    status, payload, elapsed_ms = http_request(method, url, headers=headers, body=body)

    if status >= 300:
        return _err(
            spec, seed, config, elapsed_ms, f"HTTP {status}: {str(payload)[:200]}",
            out_config=_build_out_config(config, spec, vars["_audit"], None, success=False),
            requested_k=k,
        )

    raw_items = _resolve_candidate_list(payload, spec.response.candidates_path)
    if spec.hooks.transform_candidates:
        ctx["payload"] = payload
        candidates = HOOKS[spec.hooks.transform_candidates](raw_items, ctx)
    else:
        candidates = [_map_item(it, spec.response.item) for it in raw_items if isinstance(it, dict)]

    return RunResult(
        seed_slug=seed.seed_slug,
        provider_slug=spec.slug,
        config_name=config["name"],
        config=_build_out_config(config, spec, vars["_audit"], payload, success=True),
        candidates=take_top(candidates, k),
        latency_ms=elapsed_ms,
        cost_usd=_compute_cost(spec, candidates),
        error=None,
        requested_k=k,
    )
