"""Declarative vendor-spec loader.

A vendor is described by a YAML file under `specs/<slug>.yaml`. The generic
runner (`generic_runner.py`) consumes the loaded `VendorSpec` to issue the
request and normalize the response into `Candidate`s, calling named hooks for
the irreducible per-vendor quirks. Adding a simple vendor needs only a YAML
file + a one-line allowlist entry — no Python.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import REDACTED_HEADER_NAMES
from .hooks import HOOKS

SPECS_DIR = Path(__file__).resolve().parent / "specs"

# Path values in the item map / audit map: a dotted path string, or a list of
# fallback paths (first truthy wins).
PathRef = Any  # str | list[str] | None


class AuthHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")
    header: str
    env: str
    value_prefix: str = ""


class CostSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "none"  # none | per_call | per_result
    usd: Optional[float] = None

    @model_validator(mode="after")
    def _check(self) -> "CostSpec":
        if self.mode not in ("none", "per_call", "per_result"):
            raise ValueError(f"cost.mode must be none|per_call|per_result, got {self.mode!r}")
        if self.mode != "none" and self.usd is None:
            raise ValueError("cost.usd is required unless cost.mode == none")
        return self


class MergeDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_config: str  # key in the config dict holding a sub-dict to merge
    into_body: str     # body key to shallow-merge that sub-dict into


class RequestSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str
    path: str
    query: Optional[dict[str, Any]] = None
    body: Any = None
    merge: list[MergeDirective] = Field(default_factory=list)


class ItemMap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: PathRef = None
    domain: PathRef = None
    description: PathRef = None
    rank: Optional[str] = None
    extra: dict[str, PathRef] = Field(default_factory=dict)


class ResponseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates_path: list[str]
    item: Optional[ItemMap] = None


class HooksSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preflight: Optional[str] = None
    build_request: Optional[str] = None
    transform_candidates: Optional[str] = None


class VendorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    name: str
    base_url: str
    requires_seed_domain: bool = False
    no_domain_error: Optional[str] = None  # custom message for the guard
    cost: CostSpec = Field(default_factory=CostSpec)
    auth: list[AuthHeader] = Field(default_factory=list)
    request: RequestSpec
    response: ResponseSpec
    hooks: HooksSpec = Field(default_factory=HooksSpec)
    # audit: ordered map of config-key -> source, merged into the returned
    # config on success. source "$audit.X" reads ctx["vars"]["_audit"][X]
    # (also applied on the error path); any other value is a response path.
    audit: dict[str, str] = Field(default_factory=dict)
    configs: list[dict[str, Any]]

    @model_validator(mode="after")
    def _validate(self) -> "VendorSpec":
        for stage, name in (
            ("preflight", self.hooks.preflight),
            ("build_request", self.hooks.build_request),
            ("transform_candidates", self.hooks.transform_candidates),
        ):
            if name and name not in HOOKS:
                raise ValueError(f"spec {self.slug}: unknown {stage} hook {name!r}")
        if self.request.body is not None and self.request.method.upper() == "GET":
            raise ValueError(f"spec {self.slug}: GET request must not declare a body")
        if not self.hooks.transform_candidates and self.response.item is None:
            raise ValueError(
                f"spec {self.slug}: response.item is required when no transform_candidates hook is set"
            )
        for a in self.auth:
            if a.header.lower() not in REDACTED_HEADER_NAMES:
                raise ValueError(
                    f"spec {self.slug}: auth header {a.header!r} is not in REDACTED_HEADER_NAMES "
                    "— add it to common.REDACTED_HEADER_NAMES so it is redacted in audit artifacts"
                )
        for c in self.configs:
            if "name" not in c:
                raise ValueError(f"spec {self.slug}: every config needs a 'name'")
        return self

    def auth_env_keys(self) -> list[str]:
        return [a.env for a in self.auth]


_CACHE: dict[str, VendorSpec] = {}


def load_spec(slug: str) -> VendorSpec:
    if slug in _CACHE:
        return _CACHE[slug]
    path = SPECS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"vendor spec missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = VendorSpec(**data)
    if spec.slug != slug:
        raise ValueError(f"spec slug {spec.slug!r} != filename {slug!r} ({path})")
    _CACHE[slug] = spec
    return spec
