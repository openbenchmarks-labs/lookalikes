"""G2 V2 API adapter — an authorized, license-clean programmatic source.

Unlike scraping g2.com (DataDome-walled, ToS-prohibited), this hits G2's
official V2 REST API (https://data.g2.com) with a self-serve Bearer token.
Two complementary rosters, both returning name + company domain + website:

  • category roster   GET /api/v2/products?filter[category_id][]=<uuid>
                      (the WHOLE published market roster — needs products.read)
  • seed competitors  GET /api/v2/products/{slug}/competitors
                      (G2's competitor set for the seed — no scope required)

Set G2_API_TOKEN in .env. CLI writes a `<seed>.g2.json` layer that
`build.py freeze` merges alongside the manual + auto layers.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lookalike import recall as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = ROOT / "data" / "lookalike-tam" / "gold"
BASE = "https://data.g2.com"
PRODUCT_FIELDS = "name,domain,public_detail_url,slug,star_rating,review_count"


def _load_token() -> str:
    tok = os.environ.get("G2_API_TOKEN")
    if not tok:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("G2_API_TOKEN=") and not line.startswith("#"):
                    tok = line.split("=", 1)[1].strip()
                    break
    if not tok:
        raise RuntimeError("G2_API_TOKEN not set (env or .env)")
    return tok


def _get(path: str, params: dict[str, Any] | None = None, *, token: str | None = None) -> dict:
    token = token or _load_token()
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.api+json",
            "User-Agent": "benchmark-runner/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"G2 API {exc.code} on {path}: {body}") from exc


def _domains_from(attrs: dict) -> list[str]:
    out: list[str] = []
    for key in ("domain", "public_detail_url"):
        cd = R.canonical_domain(attrs.get(key))
        if cd and cd not in out:
            out.append(cd)
    return out


def resolve_category(slug: str, *, token: str | None = None) -> dict | None:
    """Resolve a category slug (e.g. 'sales-compensation') to {id, name, slug}."""
    data = _get("/api/v2/categories", {"filter[slug_eq]": slug}, token=token)
    rows = data.get("data") or []
    if not rows:
        return None
    r = rows[0]
    return {"id": r.get("id"), "name": (r.get("attributes") or {}).get("name"),
            "slug": (r.get("attributes") or {}).get("slug", slug)}


def products_in_category(category_id: str, *, max_items: int = 500, token: str | None = None) -> list[dict]:
    """Full product roster for a category (paginated). Needs products.read scope."""
    out: list[dict] = []
    params = {
        "filter[category_id][]": category_id,
        "fields[products]": PRODUCT_FIELDS,
        "page[size]": 100,
    }
    after: str | None = None
    while len(out) < max_items:
        if after:
            params["page[after]"] = after
        data = _get("/api/v2/products", params, token=token)
        for row in data.get("data") or []:
            attrs = row.get("attributes") or {}
            out.append({"name": attrs.get("name", "").strip(), "slug": attrs.get("slug"),
                        "domains": _domains_from(attrs), "g2_url": attrs.get("g2_url"),
                        "star_rating": attrs.get("star_rating"), "review_count": attrs.get("review_count")})
        after = _next_cursor(data)
        if not after:
            break
    return [c for c in out if c["name"]]


def competitors_of(product: str, *, per: int = 50, token: str | None = None) -> list[dict]:
    """G2's competitor roster for a seed product (UUID or slug). No scope required."""
    data = _get(
        f"/api/v2/products/{urllib.parse.quote(product, safe='')}/competitors",
        {"per": per, "fields[products]": PRODUCT_FIELDS},
        token=token,
    )
    out: list[dict] = []
    for row in data.get("data") or []:
        attrs = row.get("attributes") or {}
        out.append({"name": attrs.get("name", "").strip(), "slug": attrs.get("slug"),
                    "domains": _domains_from(attrs), "g2_url": attrs.get("g2_url")})
    return [c for c in out if c["name"]]


def _next_cursor(data: dict) -> str | None:
    """Pull the forward cursor from a JSON:API response (links or meta)."""
    for container in (data.get("links"), data.get("meta")):
        if isinstance(container, dict):
            nxt = container.get("next")
            if nxt:
                # could be a full URL or a bare cursor token
                if "page[after]=" in nxt:
                    return urllib.parse.parse_qs(urllib.parse.urlparse(nxt).query).get("page[after]", [None])[0]
                return nxt
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", required=True, help="seed slug for the output layer file")
    p.add_argument("--seed-name", default=None)
    p.add_argument("--seed-domain", default=None)
    p.add_argument("--category-slug", default=None, help="G2 category slug for the full roster (needs products.read)")
    p.add_argument("--competitors-of", default=None, help="G2 product slug/UUID to pull competitors for (no scope)")
    p.add_argument("--max", type=int, default=500)
    p.add_argument("--probe", action="store_true", help="just print connectivity + sample, don't write")
    args = p.parse_args()

    token = _load_token()
    by_key: dict[str, dict] = {}
    sources_used: list[dict] = []
    today = datetime.date.today().isoformat()

    def add(name: str, domains: list[str], source: str, extra: dict | None = None) -> None:
        if not domains:
            return  # need a domain to match
        key = domains[0]
        rec = by_key.setdefault(key, {"name": name, "domains": [], "sources": [], "attrs": {"origin": "g2-api"}})
        for d in domains:
            if d not in rec["domains"]:
                rec["domains"].append(d)
        if source not in rec["sources"]:
            rec["sources"].append(source)
        if extra:
            rec["attrs"].update(extra)

    if args.category_slug:
        cat = resolve_category(args.category_slug, token=token)
        if not cat:
            print(f"category '{args.category_slug}' not found on G2", file=sys.stderr)
        else:
            print(f"category {cat['name']} → {cat['id']}")
            src_id = f"g2:{cat['slug']}"
            sources_used.append({"id": src_id, "tier": "B", "name": f"G2 category roster — {cat['name']}",
                                 "url": f"https://www.g2.com/categories/{cat['slug']}", "captured_at": today})
            roster = products_in_category(cat["id"], max_items=args.max, token=token)
            print(f"  roster: {len(roster)} products")
            for c in roster:
                add(c["name"], c["domains"], src_id, {"g2_url": c.get("g2_url"), "g2_slug": c.get("slug")})

    if args.competitors_of:
        comp = competitors_of(args.competitors_of, token=token)
        print(f"competitors of {args.competitors_of}: {len(comp)} products")
        src_id = "g2:competitors"
        sources_used.append({"id": src_id, "tier": "B", "name": f"G2 competitors of {args.competitors_of}",
                             "url": f"https://www.g2.com/products/{args.competitors_of}/competitors",
                             "captured_at": today})
        for c in comp:
            add(c["name"], c["domains"], src_id, {"g2_url": c.get("g2_url"), "g2_slug": c.get("slug")})

    companies = sorted(by_key.values(), key=lambda c: (-len(c["sources"]), c["name"]))
    if args.probe:
        print(f"\nprobe ok — {len(companies)} unique companies:")
        for c in companies[:15]:
            print(f"  {c['name']:<26} {c['sources']}  {c['domains']}")
        return 0

    layer = {
        "seed_slug": args.seed, "seed_name": args.seed_name or args.seed,
        "seed_domain": args.seed_domain, "category": "", "as_of": today,
        "_origin": "g2-api (authorized V2 REST)", "sources_used": sources_used,
        "companies": companies,
    }
    out = GOLD_DIR / f"{args.seed}.g2.json"
    out.write_text(json.dumps(layer, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(companies)} companies)")
    print(f"  next: python3 scripts/lookalike/goldset/build.py freeze --seed {args.seed} --allow-rejects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
