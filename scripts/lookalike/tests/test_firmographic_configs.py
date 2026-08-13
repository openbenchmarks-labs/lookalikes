"""Offline request-build tests for the recall-cohort firmographic configs.

No network: `http_request` is monkeypatched to capture the outgoing request and
return a canned empty response. Asserts that
  - filtered/firmographic configs fold the seed's public hints into the request
    (Parallel objective text, Ocean companiesFilters),
  - they SkipConfig (→ "skipped: …" RunResult) on a seed with no hints,
  - non-filtered configs are unchanged (no firmographic params leak in),
  - Ocean's static configs still layer their filters via the spec `merge`.

Run:  PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_firmographic_configs.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse

# Dummy creds so require_env() passes; http_request is mocked so they're unused.
for _k in ("OCEAN_API_KEY", "PARALLEL_API_KEY", "EXA_API_KEY", "EXTRUCT_API_TOKEN", "DISCOLIKE_API_KEY"):
    os.environ.setdefault(_k, "test-key")

from lookalike import generic_runner  # noqa: E402
from lookalike.common import Candidate, Seed  # noqa: E402
from lookalike.hooks import parallel as parallel_hook  # noqa: E402
from lookalike.hooks import predictleads as predictleads_hook  # noqa: E402
from lookalike.spec_loader import load_spec  # noqa: E402

FIRMO = {
    "locations": ["us"],
    "min_employees": 11,
    "max_employees": 1000,
    "funding_stages": ["series_a"],
}
SEED_FIRMO = Seed("acme", "Acme", "acme.com", "Acme makes widgets", "saas", firmographics=FIRMO)
SEED_BARE = Seed("bare", "Bare Co", "bare.com", "Bare desc", "saas", firmographics=None)


class Capture:
    """Stands in for common.http_request: records calls, returns empty results."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=60):
        self.calls.append({"method": method, "url": url, "body": body})
        return 200, {"results": [], "companies": [], "entities": [], "data": []}, 5

    @property
    def last_search(self) -> dict:
        return self.calls[-1]


def _run(slug: str, config_name: str, seed: Seed):
    spec = load_spec(slug)
    config = next(c for c in spec.configs if c["name"] == config_name)
    cap = Capture()
    generic_runner.http_request = cap          # type: ignore[assignment]
    try:
        result = generic_runner.run_from_spec(spec, seed, 100, config)
    finally:
        # restore the real function so other tests/imports are unaffected
        import lookalike.common as _c
        generic_runner.http_request = _c.http_request      # type: ignore[assignment]
    return result, cap


def _qs(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def test_parallel() -> None:
    print("\n[parallel]")
    _, cap = _run("parallel", "seed_first", SEED_FIRMO)
    objective = cap.last_search["body"]["objective"]
    check(
        objective == (
            "Find companies similar to Acme (acme.com), Acme makes widgets. "
            "Return companies a buyer would realistically evaluate alongside Acme."
        ),
        "objective is seed-first: name + domain anchor, identity clause, buyer return framing",
    )
    check(cap.last_search["body"]["match_limit"] == 100, "match_limit=100 (deep fetch)")
    ctx = {"seed": SEED_FIRMO, "config": {"query_variant": "concise_fallback"}, "vars": {}}
    parallel_hook.objective(ctx)
    check(
        ctx["vars"]["query"] == "companies like Acme (acme.com) — Acme makes widgets",
        "fallback restores the concise name-plus-identity query",
    )


def test_seed_exclusion() -> None:
    print("\n[seed exclusion]")
    candidates = [
        Candidate("Acme", "acme.com", rank=1),
        Candidate("Acme mirror", "www.acme.com", rank=2),
        Candidate("Peer", "peer.com", rank=3),
    ]
    got = generic_runner._without_seed(candidates, SEED_FIRMO)
    check([c.name for c in got] == ["Peer"], "seed and www-domain duplicate are excluded")
    check(got[0].rank is None, "remaining candidates are re-ranked before scoring")


def test_post_fetch_deduplication() -> None:
    print("\n[post-fetch deduplication]")
    candidates = [
        Candidate("Ramp", None, rank=1, extra={"linkedin_url": "https://www.linkedin.com/company/ramp"}),
        Candidate("Ramp", None, rank=2, extra={"linkedin_url": "https://tw.linkedin.com/company/ramp?trk=foo"}),
        Candidate("Ramp", "ramp.com", rank=3),
        Candidate("Ramp Payments", "ramp.com", rank=4),
        Candidate("Brex", "brex.com", rank=5),
    ]
    got, removed = generic_runner._without_duplicates(candidates)
    check([c.name for c in got] == ["Ramp", "Brex"], "dedupes by LinkedIn slug, domain, and name")
    check(removed == 3, "records every removed duplicate")
    check(all(c.rank is None for c in got), "deduplicated results are re-ranked before scoring")


def test_parallel_embedded_domain() -> None:
    print("\n[parallel domain extraction]")
    check(
        parallel_hook._embedded_website_domain(
            "Website Url: https://www.autolease.com.tw/index.html | Founded Year: 2007"
        ) == "autolease.com.tw",
        "embedded Website Url becomes the canonical company domain",
    )


def test_exa() -> None:
    print("\n[exa]")
    _, cap = _run("exa", "seed_first", SEED_FIRMO)
    check(
        cap.last_search["body"]["query"] == (
            "Find companies similar to Acme (acme.com), Acme makes widgets. "
            "Return companies a buyer would realistically evaluate alongside Acme."
        ),
        "query is seed-first: name + domain anchor, identity clause, buyer return framing",
    )

    seed_short = Seed(
        "short", "Shorty", "shorty.com",
        "Long feature enumeration that should not reach the query", "saas",
        short_description="the widget platform",
    )
    _, cap = _run("exa", "seed_first", seed_short)
    check(
        cap.last_search["body"]["query"] == (
            "Find companies similar to Shorty (shorty.com), the widget platform. "
            "Return companies a buyer would realistically evaluate alongside Shorty."
        ),
        "short_description wins over the long description in the query",
    )

    seed_no_desc = Seed("no-desc", "Name Only", "name-only.com", None, "saas")
    _, cap = _run("exa", "seed_first", seed_no_desc)
    check(
        cap.last_search["body"]["query"] == (
            "Find companies similar to Name Only (name-only.com). "
            "Return companies a buyer would realistically evaluate alongside Name Only."
        ),
        "query stays seed-only when no description exists",
    )


def test_extruct() -> None:
    print("\n[extruct]")
    _, cap = _run("extruct", "domain_similarity", SEED_FIRMO)
    check(
        cap.last_search["url"] == "https://api.extruct.ai/v1/companies/acme.com/similar",
        "uses the seed domain in Extruct's lookalike path",
    )
    check(cap.last_search["body"] == {"pagination": {"offset": 0, "limit": 100}}, "requests 100 results")


def test_discolike() -> None:
    print("\n[discolike]")
    _, cap = _run("discolike", "domain_similarity", SEED_FIRMO)
    check(
        _qs(cap.last_search["url"]) == {"domain": ["acme.com"], "max_records": ["100"]},
        "uses the seed domain and requests 100 results",
    )


def test_ocean() -> None:
    print("\n[ocean]")
    _, cap = _run("ocean", "seed_firmographic", SEED_FIRMO)
    cf = cap.last_search["body"]["companiesFilters"]
    # 11..1000 overlaps bands 11-50, 51-200, 201-500, 501-1000
    check(cf.get("companySizes") == ["11-50", "51-200", "201-500", "501-1000"],
          f"companySizes bands from employee range (got {cf.get('companySizes')})")
    check(cf.get("primaryLocations") == {"includeCountries": ["us"]},
          f"primaryLocations from hint (got {cf.get('primaryLocations')})")
    check("acme.com" in cf.get("lookalikeDomains", []), "lookalikeDomains still set")

    res, _ = _run("ocean", "seed_firmographic", SEED_BARE)
    check((res.error or "").startswith("skipped"), f"bare seed skips firmo config (err={res.error!r})")

    # static config still layers filters via merge; no firmo bands injected
    _, cap = _run("ocean", "mid_market", SEED_FIRMO)
    cf = cap.last_search["body"]["companiesFilters"]
    check(cf.get("companySizes") == ["51-200", "201-500", "501-1000"],
          f"mid_market companySizes from merge (got {cf.get('companySizes')})")
    check("primaryLocations" not in cf, "mid_market has no primaryLocations")

    _, cap = _run("ocean", "seed_only", SEED_FIRMO)
    cf = cap.last_search["body"]["companiesFilters"]
    check("companySizes" not in cf and "primaryLocations" not in cf,
          "seed_only carries only lookalike/exclude domains")


class PagedPredictLeads:
    """Mock PredictLeads: serves `per_page_rows` similarity rows per page until
    `total` is reached, advertising meta.count=total. Lets us assert the hook
    follows pages to k and stops on the right condition."""

    def __init__(self, total: int, per_page_rows: int = 20) -> None:
        self.total, self.per_page_rows = total, per_page_rows
        self.pages_fetched: list[int] = []

    def __call__(self, method, url, *, headers=None, body=None, timeout=60):
        q = _qs(url)
        page = int(q.get("page", ["1"])[0])
        self.pages_fetched.append(page)
        start = (page - 1) * self.per_page_rows
        rows, included = [], []
        for i in range(start, min(start + self.per_page_rows, self.total)):
            cid = f"c{i}"
            rows.append({
                "attributes": {"position": i + 1, "reason": "similar"},
                "relationships": {"similar_company": {"data": {"id": cid}}},
            })
            included.append({"type": "company", "id": cid,
                             "attributes": {"company_name": f"Co {i}", "domain": f"co{i}.com"}})
        return 200, {"data": rows, "included": included, "meta": {"count": self.total}}, 5


def _run_predictleads(spec, seed, config, mock):
    """Patch BOTH the generic runner (page 1) and the hook (pages 2+)."""
    import lookalike.common as _c
    generic_runner.http_request = mock          # type: ignore[assignment]
    predictleads_hook.http_request = mock        # type: ignore[assignment]
    try:
        return generic_runner.run_from_spec(spec, seed, 100, config)
    finally:
        generic_runner.http_request = _c.http_request       # type: ignore[assignment]
        predictleads_hook.http_request = _c.http_request     # type: ignore[assignment]


def test_predictleads_pagination() -> None:
    print("\n[predictleads pagination]")
    spec = load_spec("predictleads")
    paginated = next(c for c in spec.configs if c["name"] == "paginated")
    default = next(c for c in spec.configs if c["name"] == "default")
    seed = Seed("pl", "PL", "pl.com", "desc", "saas")

    # paginated, graph of 60 → follow pages, stop when meta.count (60) drained.
    mock = PagedPredictLeads(total=60, per_page_rows=20)
    res = _run_predictleads(spec, seed, paginated, mock)
    check(len(res.candidates) == 60, f"paginated drains 60-company graph (got {len(res.candidates)})")
    check(mock.pages_fetched == [1, 2, 3], f"followed pages 1→3 (got {mock.pages_fetched})")

    # paginated, large graph → stop at k=100 (5 pages of 20 + take_top).
    mock = PagedPredictLeads(total=500, per_page_rows=20)
    res = _run_predictleads(spec, seed, paginated, mock)
    check(len(res.candidates) == 100, f"paginated caps at k=100 (got {len(res.candidates)})")
    check(max(mock.pages_fetched) == 5, f"stopped once k reached, ≤5 pages (got {mock.pages_fetched})")

    # default config never paginates — one page only, even at k=100.
    mock = PagedPredictLeads(total=500, per_page_rows=20)
    res = _run_predictleads(spec, seed, default, mock)
    check(mock.pages_fetched == [1], f"default config fetches one page only (got {mock.pages_fetched})")


def main() -> int:
    print("=== firmographic config request-build tests (offline) ===")
    test_parallel()
    test_seed_exclusion()
    test_post_fetch_deduplication()
    test_parallel_embedded_domain()
    test_exa()
    test_extruct()
    test_discolike()
    test_ocean()
    test_predictleads_pagination()
    print()
    if failures:
        print(f"FAILED ({len(failures)} checks)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
