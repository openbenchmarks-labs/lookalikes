"""Wikidata adapter — a license-clean, programmatic source.

Wikidata is CC0 and exposes a public SPARQL + REST API, so unlike Gartner/G2/
trade rosters it can be queried directly without ToS friction or bot walls.

Two signals come out of one industry query (see `companies_in_industry`):
  1. membership in the industry roster (P452) — the "wikidata" source
  2. whether the company has its own English Wikipedia article — an
     INDEPENDENT editorial process, counted as the "wikipedia" source

That lets the corroboration rule (>=2 sources) be satisfied from open data for
any company notable enough to have a Wikipedia article — which doubles as a
notability/reliability filter.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# Wikidata's own WDQS is frequently rate-limited / in outage. QLever mirrors the
# full Wikidata dump over SPARQL and is fast + unthrottled, so we try it first
# and fall back to WDQS (with 429-aware backoff).
SPARQL_ENDPOINTS = [
    "https://qlever.dev/api/wikidata",
    "https://query.wikidata.org/sparql",
]
API_URL = "https://www.wikidata.org/w/api.php"
UA = "benchmark-runner/0.1 (+http://openbenchmarks.com/; contact=founders@openbenchmarks.com)"


def _get_json(url: str, *, accept: str = "application/json") -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"wikidata fetch failed: {url[:120]} — {last_exc}")


def _sparql(query: str) -> dict[str, Any]:
    """Run a SPARQL query, trying QLever then WDQS. Handles 429 with long backoff."""
    last_exc: Exception | None = None
    for endpoint in SPARQL_ENDPOINTS:
        url = endpoint + "?" + urllib.parse.urlencode({"query": query})
        for attempt in range(3):
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429:
                    wait = 65  # WDQS aggressive limit is ~1 req/min
                    print(f"    {endpoint.split('/')[2]} 429 → waiting {wait}s")
                    time.sleep(wait)
                else:
                    break  # try next endpoint
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"all SPARQL endpoints failed — {last_exc}")


def search_entity(query: str, *, kind: str = "item") -> dict[str, str] | None:
    """Resolve a free-text query to the top Wikidata entity. Returns
    {qid, label, description} or None."""
    params = {
        "action": "wbsearchentities", "search": query, "language": "en",
        "format": "json", "type": kind, "limit": "1",
    }
    data = _get_json(API_URL + "?" + urllib.parse.urlencode(params))
    hits = data.get("search") or []
    if not hits:
        return None
    h = hits[0]
    return {"qid": h.get("id", ""), "label": h.get("label", ""), "description": h.get("description", "")}


def companies_in_industry(industry_qid: str, limit: int = 300) -> list[dict[str, Any]]:
    """Return companies whose `industry` (P452) is `industry_qid` (including
    industry subclasses one level down via P452/P279*), each with:
        {name, qid, domains:[...], has_enwiki: bool, enwiki_url}
    """
    # Label-service-free (works on both QLever and WDQS): fetch English rdfs:label
    # directly. Industry match walks subclasses (P452 then P279*).
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
    SELECT ?company ?companyLabel ?website ?article WHERE {{
      ?company wdt:P452/wdt:P279* wd:{industry_qid} .
      ?company rdfs:label ?companyLabel . FILTER(LANG(?companyLabel) = "en")
      OPTIONAL {{ ?company wdt:P856 ?website . }}
      OPTIONAL {{
        ?article schema:about ?company ;
                 schema:isPartOf <https://en.wikipedia.org/> .
      }}
    }} LIMIT {limit}
    """
    data = _sparql(query)

    by_qid: dict[str, dict[str, Any]] = {}
    for row in data.get("results", {}).get("bindings", []):
        comp_uri = row.get("company", {}).get("value", "")
        qid = comp_uri.rsplit("/", 1)[-1] if comp_uri else ""
        name = row.get("companyLabel", {}).get("value", "").strip()
        if not name or not qid or name == qid:
            continue
        entry = by_qid.setdefault(
            qid, {"name": name, "qid": qid, "domains": [], "has_enwiki": False, "enwiki_url": None}
        )
        website = row.get("website", {}).get("value")
        if website:
            host = website.replace("https://", "").replace("http://", "").split("/")[0]
            if host and host not in entry["domains"]:
                entry["domains"].append(host)
        article = row.get("article", {}).get("value")
        if article:
            entry["has_enwiki"] = True
            entry["enwiki_url"] = article
    return list(by_qid.values())


def industries_of(company_qid: str) -> list[str]:
    """Industry (P452) QIDs declared on a company entity."""
    params = {"action": "wbgetentities", "ids": company_qid, "props": "claims", "format": "json"}
    data = _get_json(API_URL + "?" + urllib.parse.urlencode(params))
    claims = (data.get("entities", {}).get(company_qid, {}) or {}).get("claims", {})
    out: list[str] = []
    for c in claims.get("P452", []) or []:
        try:
            out.append(c["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            continue
    return out
