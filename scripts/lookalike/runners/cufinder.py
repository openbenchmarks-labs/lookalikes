"""CUFinder Company Lookalikes Finder adapter (a hard top-10 endpoint)."""
from __future__ import annotations

from typing import Any

from ..common import Candidate, RunResult, Seed, http_request, require_env, take_top

VENDOR_SLUG = "cufinder"
VENDOR_NAME = "CUFinder"
CONFIGS: list[dict[str, Any]] = [{"name": "domain_lookalikes"}]
MAX_RESULTS = 10


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    env = require_env("CUFINDER_API_KEY")
    status, payload, elapsed_ms = http_request(
        "POST",
        "https://api.cufinder.io/v2/fcl",
        headers={"x-api-key": env["CUFINDER_API_KEY"]},
        body={"query": seed.seed_domain or seed.seed_name},
        body_encoding="form",
    )
    if status >= 300:
        return RunResult(seed.seed_slug, VENDOR_SLUG, config["name"], config, [], elapsed_ms,
                         error=f"HTTP {status}: {str(payload)[:200]}", requested_k=k, max_results=MAX_RESULTS)
    rows = payload.get("data", {}).get("companies", []) if isinstance(payload, dict) else []
    candidates = [
        Candidate(
            name=str(row.get("name") or "").strip(),
            domain=row.get("domain") or row.get("website"),
            description=row.get("description"),
            extra={
                "linkedin_url": row.get("linkedin_url"), "industry": row.get("industry"),
                "employee_count": row.get("employee_count"), "size": row.get("size"),
                "country": row.get("country"), "confidence_level": payload.get("data", {}).get("confidence_level"),
            },
        )
        for row in rows if isinstance(row, dict)
    ]
    return RunResult(seed.seed_slug, VENDOR_SLUG, config["name"], config, take_top(candidates, min(k, MAX_RESULTS)),
                     elapsed_ms, requested_k=k, max_results=MAX_RESULTS)
