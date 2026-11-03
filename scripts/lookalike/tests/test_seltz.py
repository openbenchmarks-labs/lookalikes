"""Offline tests for the Seltz lookalike runner.

  1. Markdown parse: H1 name, canonical Website domain (never LinkedIn), About
     description, firmographic extras.
  2. Request build: query is "<anchor> for <caps>" (PR #8), not the
     authored seed-first lookalike sentence.
  3. Spec load: registered in REGISTRY and the generic runner.

No network. Run:
  PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_seltz.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SELTZ_API_KEY", "test-key")

from lookalike import generic_runner  # noqa: E402
from lookalike.common import Seed  # noqa: E402
from lookalike.hooks.seltz import parse_companies  # noqa: E402
from lookalike.runners import REGISTRY  # noqa: E402
from lookalike.spec_loader import load_spec  # noqa: E402

MARKDOWN = """# Square

**Payments · Public company · San Francisco**

## About
Square (now Block) provides payment processing and point-of-sale
software for restaurants and retailers.

## Company Details
* Website: https://www.squareup.com
* LinkedIn: https://www.linkedin.com/company/join-square
* Industry: Payments
* Type: Public company
* Size: 5,001-10,000
* Headquarters: San Francisco, CA
* Business model: SaaS + payments
"""

SEED = Seed(
    seed_slug="toast",
    seed_name="Toast",
    seed_domain="toasttab.com",
    description="Restaurant point-of-sale, payments, online ordering, payroll, and operations platform",
    category="hospitality",
    short_description="the restaurant point-of-sale and operations platform",
    query="Find companies similar to Toast (toasttab.com): restaurant point-of-sale platforms.",
    anchor="restaurant point-of-sale platform",
    capabilities=["payments", "online ordering", "payroll", "inventory management"],
)

EXPECTED_QUERY = (
    "restaurant point-of-sale platform for payments, online ordering, payroll, "
    "and inventory management"
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        failures.append(msg)


class Capture:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=60):
        self.calls.append({"method": method, "url": url, "body": body, "timeout": timeout, "headers": headers})
        return 200, {"documents": []}, 5


def _run(config_name: str):
    spec = load_spec("seltz")
    config = next(c for c in spec.configs if c["name"] == config_name)
    cap = Capture()
    generic_runner.http_request = cap  # type: ignore[assignment]
    try:
        result = generic_runner.run_from_spec(spec, SEED, 100, config)
    finally:
        import lookalike.common as _c
        generic_runner.http_request = _c.http_request  # type: ignore[assignment]
    return result, cap


def main() -> int:
    print("parse_companies")
    docs = [{
        "url": "https://www.linkedin.com/company/join-square",
        "content": MARKDOWN,
    }]
    cands = parse_companies(docs, {})
    check(len(cands) == 1, f"one candidate (got {len(cands)})")
    if cands:
        c = cands[0]
        check(c.name == "Square", f"name from H1 (got {c.name!r})")
        check(c.domain == "squareup.com", f"canonical website domain, not LinkedIn (got {c.domain!r})")
        check(c.description is not None and c.description.startswith("Square (now Block)"), "About section as description")
        check(c.extra.get("linkedin_url") == "https://www.linkedin.com/company/join-square", "linkedin from details")
        check(c.extra.get("industry") == "Payments", "industry extra")
        check(c.extra.get("headquarters") == "San Francisco, CA", "headquarters extra")

    skipped = parse_companies([{"url": "https://linkedin.com/company/x", "content": "no headings"}], {})
    check(skipped == [], "skip documents with neither name nor domain")

    print("\nrequest build")
    spec = load_spec("seltz")
    check(spec.slug == "seltz", "spec slug")
    check("seltz" in REGISTRY, "registered")
    check(spec.timeout_sec == 180, f"timeout_sec=180 (got {spec.timeout_sec})")

    result, cap = _run("anchor_capabilities_query")
    check(result.error is None, f"anchor_capabilities_query run ok (got {result.error})")
    check(cap.calls and cap.calls[0]["url"] == "https://api.seltz.ai/v1/search", "POST /search")
    check(cap.calls[0]["timeout"] == 180, f"timeout forwarded (got {cap.calls[0].get('timeout')})")
    body = cap.calls[0]["body"]
    check(body.get("scope") == "companies", "scope=companies")
    check(body.get("max_results") == 100, "max_results=$k")
    check(
        body.get("query") == EXPECTED_QUERY,
        f"query is anchor+capabilities (got {body.get('query')!r})",
    )
    check(body.get("query") != SEED.query, "must not send the authored lookalike query")
    check(body.get("query") != SEED.description, "must not send the seed description")
    check(cap.calls[0]["headers"].get("x-api-key") == "test-key", "x-api-key auth")

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
