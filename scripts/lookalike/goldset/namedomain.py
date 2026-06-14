"""Neutral name -> domain resolver for gold-set construction.

The G2 Data API only returns a website for ~12% of catalog products, so we
resolve the rest from company name. Resolution MUST stay vendor-independent
(methodology §5.1), so we use Clearbit's free keyless autocomplete — Clearbit
(HubSpot) is NOT one of the benchmarked vendors (openfunnel/ocean/exa/parallel/
predictleads). Results are cached and guarded by a name-match check to avoid
fuzzy mis-resolution.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lookalike import recall as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "lookalike-tam" / "gold" / "_namecache"
CLEARBIT = "https://autocomplete.clearbit.com/v1/companies/suggest?query="

# Generic tokens that shouldn't drive a name match.
_STOP = {
    "marketing", "platform", "software", "the", "all", "in", "one", "suite", "inc",
    "llc", "co", "app", "cloud", "hub", "io", "crm", "automation", "engagement",
    "and", "for", "by", "group", "solutions", "systems", "technologies", "labs",
    "ai", "com", "engage", "plus", "pro", "online", "services",
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in _STOP and len(t) > 1}


def _clearbit(query: str) -> list[dict]:
    req = urllib.request.Request(CLEARBIT + urllib.parse.quote(query), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8")) or []
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return []
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return []


_DOMAIN_IN_NAME = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*\.(?:com|io|ai|net|co|app|org|us|dev|gr|me))\b",
    re.I,
)


def _domain_in_name(name: str) -> str | None:
    """Some G2 product names ARE the domain (e.g. '800.com', 'ChaChing.ai',
    'https://fieldcomplete.com'). Recover those for free, no network call."""
    m = _DOMAIN_IN_NAME.search(name)
    return R.canonical_domain(m.group(1)) if m else None


def _accept(query: str, cand_name: str, domain: str) -> bool:
    """Guard: the candidate must share a meaningful token with the query name,
    or the query's core token must appear in the domain's registrable label."""
    qt, ct = _tokens(query), _tokens(cand_name)
    if qt & ct:
        return True
    sld = (R.canonical_domain(domain) or "").split(".")[0]
    return any(t in sld or sld in t for t in qt) if sld else False


def resolve(name: str, *, use_cache: bool = True) -> str | None:
    """Resolve a company name to a canonical domain (eTLD+1), or None."""
    if not name:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80] or "x"
    cf = CACHE / f"{key}.json"

    # Deterministic, free recovery: the name itself is a domain. This also
    # rescues prior null-cached misses, so it runs before the cache read.
    embedded = _domain_in_name(name)
    if embedded:
        cf.write_text(json.dumps({"name": name, "domain": embedded, "via": "name-is-domain"}) + "\n", encoding="utf-8")
        return embedded

    if use_cache and cf.exists():
        return json.loads(cf.read_text(encoding="utf-8")).get("domain")

    domain = None
    # try full name, then progressively shorter prefixes for suffix-heavy names
    queries = [name]
    words = name.split()
    if len(words) > 2:
        queries.append(" ".join(words[:2]))
    queries.append(words[0])
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        for cand in _clearbit(q):
            dom = R.canonical_domain(cand.get("domain"))
            if dom and _accept(name, cand.get("name", ""), dom):
                domain = dom
                break
        if domain:
            break

    cf.write_text(json.dumps({"name": name, "domain": domain}) + "\n", encoding="utf-8")
    return domain


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(f"{n} -> {resolve(n)}")
