"""ZoomInfo GTM CLI company-similarity runner.

The native endpoint returns a fixed, non-paginated top 25.  Its results have
no company website or description, so the benchmark records only its own
attributes; it does not enrich candidates in a second call.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from ..common import Candidate, RunResult, Seed, record_vendor_call, take_top

# Slug is deliberately not plain "zoominfo": the enrichment and funding boards
# already register that slug for a different ZoomInfo surface. The lookalike
# board keys on this one, and so do the persisted artifacts and the DB.
VENDOR_SLUG = "zoominfo-lookalike"
VENDOR_NAME = "ZoomInfo"
CONFIGS: list[dict[str, Any]] = [{"name": "native_similar_by_name"}]
MAX_RESULTS = 25


def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
    company_id = config.get("zoominfo_company_id")
    reference = ["--id", str(company_id)] if company_id else ["--name", seed.seed_name]
    command = ["gtm", "companies", "similar", *reference, "--format", "json"]
    started = time.monotonic()
    if not shutil.which("gtm"):
        return RunResult(seed.seed_slug, VENDOR_SLUG, config["name"], config, [], 0,
                         error="gtm CLI is not installed", requested_k=k, max_results=MAX_RESULTS)
    completed = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
    elapsed_ms = round((time.monotonic() - started) * 1000)
    try:
        payload: Any = json.loads(completed.stdout) if completed.stdout.strip() else {"stderr": completed.stderr.strip()}
    except json.JSONDecodeError:
        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
    record_vendor_call(
        method="CLI", url="gtm companies similar", request_body={"zoominfo_company_id": company_id, "name": None if company_id else seed.seed_name, "format": "json"},
        response_status=0 if completed.returncode == 0 else completed.returncode,
        response_body=payload, elapsed_ms=elapsed_ms,
    )
    if completed.returncode != 0:
        return RunResult(seed.seed_slug, VENDOR_SLUG, config["name"], config, [], elapsed_ms,
                         error=f"gtm exited {completed.returncode}: {completed.stderr.strip()[:200]}",
                         requested_k=k, max_results=MAX_RESULTS)
    rows = payload.get("lookalikes", []) if isinstance(payload, dict) else []
    candidates: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        name = attrs.get("companyName") or row.get("companyName")
        if not isinstance(name, str) or not name.strip():
            continue
        candidates.append(Candidate(
            name=name.strip(),
            rank=attrs.get("rank") or row.get("rank"),
            extra={
                "zoominfo_company_id": row.get("zoominfoCompanyId") or row.get("id"),
                "similarity_score": attrs.get("score") or row.get("score"),
                "industry": attrs.get("industry"),
                "revenue_range": attrs.get("revenueRange"),
                "employee_range": attrs.get("employeeRange"),
                "country": attrs.get("country"),
            },
        ))
    candidates.sort(key=lambda c: c.rank if isinstance(c.rank, int) else 10**9)
    return RunResult(seed.seed_slug, VENDOR_SLUG, config["name"], config,
                     take_top(candidates, min(k, MAX_RESULTS)), elapsed_ms,
                     requested_k=k, max_results=MAX_RESULTS)
