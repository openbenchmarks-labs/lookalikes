"""G2 Data API (via RapidAPI) adapter — authorized third-party roster source.

This is a sanctioned commercial reseller of G2 catalog data, so unlike scraping
g2.com directly (DataDome-walled, ToS-prohibited) it returns the FULL published
category roster programmatically. Two endpoints:

  GET /g2-categories?category=<slug>   → exhaustive roster: name + G2 product slug
                                          + industries + market_segments  (1 call)
  GET /g2-products?product=<slug>       → product detail: company_website (domain!),
                                          location, employees, founded, categories,
                                          alternatives                     (1 call/product)

Roster membership → Tier-A source `g2cat:<slug>` (exhaustive market roster).
Product 'alternatives' → Tier-B source `g2alt:<slug>`.

Detail calls cost quota, so they're cached to gold/_g2cache/<slug>.json.
Set G2_RAPIDAPI_KEY (+ optional G2_RAPIDAPI_HOST) in .env.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lookalike import recall as R  # noqa: E402
from lookalike.goldset import namedomain  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = ROOT / "data" / "lookalike-tam" / "gold"
CACHE_DIR = GOLD_DIR / "_g2cache"
DEFAULT_HOST = "g2-data-api.p.rapidapi.com"


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return None


def _creds() -> tuple[str, str]:
    key = _env("G2_RAPIDAPI_KEY")
    if not key:
        raise RuntimeError("G2_RAPIDAPI_KEY not set (env or .env)")
    return key, (_env("G2_RAPIDAPI_HOST") or DEFAULT_HOST)


def _call(path: str) -> Any:
    key, host = _creds()
    req = urllib.request.Request(
        f"https://{host}{path}",
        headers={"x-rapidapi-key": key, "x-rapidapi-host": host, "Content-Type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            if exc.code == 429:
                time.sleep(3 * (attempt + 1))
                last = exc
                continue
            raise RuntimeError(f"G2 RapidAPI {exc.code} on {path}: {body}") from exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"G2 RapidAPI failed on {path}: {last}")


def _slug_from_link(link: str | None) -> str | None:
    if not link:
        return None
    m = link.rstrip("/").split("/products/", 1)
    if len(m) == 2:
        return m[1].split("/", 1)[0]
    return None


def category_roster(category_slug: str) -> list[dict]:
    """Full published roster for a G2 category slug. One API call."""
    data = _call(f"/g2-categories?category={urllib.parse.quote(category_slug)}")
    rows = data.get("results") or []
    out = []
    for r in rows:
        out.append({
            "name": (r.get("name") or "").strip(),
            "g2_slug": _slug_from_link(r.get("link")),
            "industries": r.get("industries") or [],
            "market_segments": r.get("market_segments") or [],
        })
    return [r for r in out if r["name"] and r["g2_slug"]]


def product_detail(product_slug: str, *, use_cache: bool = True) -> dict | None:
    """Product detail (domain, firmographics, alternatives). Cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{product_slug}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        d = _call(f"/g2-products?product={urllib.parse.quote(product_slug)}")
    except RuntimeError as exc:
        print(f"    detail failed for {product_slug}: {str(exc)[:80]}", file=sys.stderr)
        return None
    cache.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return d


def _domain_of(detail: dict) -> str | None:
    for key in ("company_website", "product_website"):
        cd = R.canonical_domain(detail.get(key))
        if cd:
            return cd
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", required=True)
    p.add_argument("--seed-name", default=None)
    p.add_argument("--seed-domain", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--category", required=True, help="lookalike category (e.g. ecommerce) for source tiering")
    p.add_argument("--category-slug", required=True, help="G2 category slug for the roster (e.g. sms-marketing)")
    p.add_argument("--max-details", type=int, default=120, help="cap on product-detail calls (quota guard)")
    p.add_argument("--no-resolve", action="store_true", help="skip neutral name->domain fallback")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--list-only", action="store_true", help="print roster names+size only, no detail calls")
    args = p.parse_args()

    seed_name = args.seed_name or args.seed
    roster = category_roster(args.category_slug)
    print(f"G2 category '{args.category_slug}': {len(roster)} products in roster")
    if args.list_only:
        for r in roster[:40]:
            print(f"  {r['name']}  ({r['g2_slug']})")
        print(f"  ... ({len(roster)} total)")
        return 0

    src_cat = f"g2cat:{args.category_slug}"
    n_detail = min(len(roster), args.max_details)
    print(f"resolving domains for {n_detail}/{len(roster)} products (cached to _g2cache/)…")

    companies: list[dict] = []
    unresolved: list[dict] = []
    seed_dom = R.canonical_domain(args.seed_domain)
    n_g2 = n_resolved = 0
    for i, r in enumerate(roster[:n_detail]):
        d = product_detail(r["g2_slug"], use_cache=not args.no_cache) or {}
        domain = _domain_of(d)
        resolution = "g2-website"
        if not domain and not args.no_resolve:
            domain = namedomain.resolve(r["name"], use_cache=not args.no_cache)
            resolution = "clearbit-name"
        if not domain:
            unresolved.append({"name": r["name"], "g2_slug": r["g2_slug"]})
            continue
        if resolution == "g2-website":
            n_g2 += 1
        else:
            n_resolved += 1
        # self-exclusion
        if R.normalize_name(r["name"]) == R.normalize_name(seed_name) or (seed_dom and domain == seed_dom):
            continue
        companies.append({
            "name": r["name"],
            "domains": [domain],
            "sources": [src_cat],
            "attrs": {
                "origin": "g2-rapidapi", "resolution": resolution, "g2_slug": r["g2_slug"],
                "location": d.get("company_location"),
                "employees_linkedin": d.get("number_of_employees_on_linkedin"),
                "founded_year": d.get("company_founded_year"),
                "annual_revenue": d.get("company_annual_revenue"),
                "g2_categories": [c.get("name") for c in (d.get("categories") or []) if c.get("name")],
                "market_segments": r["market_segments"], "industries": r["industries"],
            },
        })
        if (i + 1) % 25 == 0:
            print(f"    …{i + 1}/{n_detail}  kept={len(companies)}")

    companies.sort(key=lambda c: c["name"])
    today = datetime.date.today().isoformat()
    layer = {
        "seed_slug": args.seed, "seed_name": seed_name, "seed_domain": args.seed_domain,
        "description": args.description,
        "category": args.category, "as_of": today,
        "_origin": "g2-rapidapi (G2 Data API category roster + product detail)",
        "sources_used": [{
            "id": src_cat, "tier": "A",
            "name": f"G2 category roster — {args.category_slug} ({len(roster)} products)",
            "url": f"https://www.g2.com/categories/{args.category_slug}", "captured_at": today,
        }],
        "domain_coverage": {
            "roster_size": len(roster), "attempted": n_detail,
            "resolved": len(companies), "g2_website": n_g2, "name_resolved": n_resolved,
            "unresolved": len(unresolved),
        },
        "unresolved": sorted(unresolved, key=lambda x: x["name"].lower()),
        "companies": companies,
    }
    out = GOLD_DIR / f"{args.seed}.g2.json"
    out.write_text(json.dumps(layer, indent=2) + "\n", encoding="utf-8")
    cov = round(100 * len(companies) / max(1, n_detail), 1)
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(companies)} companies with domains; "
          f"g2-website={n_g2}, name-resolved={n_resolved}, unresolved={len(unresolved)}, coverage={cov}%)")
    print(f"  next: python3 scripts/lookalike/goldset/build.py freeze --seed {args.seed} --allow-rejects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
