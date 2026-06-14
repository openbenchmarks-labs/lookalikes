"""Pre-registered, per-category source registry + the inclusion rule.

This is the frozen list of sanctioned third-party rosters the gold set may be
built from (methodology §5.2), plus the function that enforces the inclusion
rule (§5.2): a company qualifies iff it appears in **>=1 Tier-A source** OR
**>=2 distinct Tier-B sources**.

Hard rule (§5.1): none of these sources is a benchmarked vendor. Adding a
vendor API here would invalidate the benchmark.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

Tier = Literal["A", "B"]


@dataclasses.dataclass(frozen=True)
class Source:
    """A sanctioned third-party roster.

    `access` documents how facts are obtained:
      • "open"   — programmatic, license-clean (e.g. Wikidata SPARQL)
      • "manual" — captured by hand as facts (ToS/paywalled rosters: Gartner,
                   G2, Crunchbase, trade league tables, association directories)
    We store only derived domains + a citation, never mirrored source prose.
    """

    id: str
    tier: Tier
    name: str
    access: Literal["open", "manual"]
    note: str = ""


# --------------------------------------------------------------------------- #
# Registry — keyed by LookalikeCategory. Mirror of RECALL_METHODOLOGY.md §5.2. #
# --------------------------------------------------------------------------- #

# Source ids are bare provider prefixes. A concrete citation namespaces the
# prefix with specifics, e.g. `g2:sms-marketing`, `crunchbase:postscript`,
# `gartner:field-service-management`. `tier_of` matches on the prefix.
SOURCES: dict[str, list[Source]] = {
    "b2b-saas": [
        Source("gartner", "A", "Gartner Peer Insights market roster", "manual"),
        Source("g2cat", "A", "G2 category roster (exhaustive, via G2 Data API)", "open"),
        Source("g2", "B", "G2 category page", "manual"),
        Source("g2alt", "B", "G2 product 'alternatives' list", "open"),
        Source("capterra", "B", "Capterra category page", "manual"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
        Source("wikidata", "B", "Wikidata industry / instance-of", "open"),
        Source("wikipedia", "B", "English Wikipedia article (independent editorial)", "open"),
        Source("sec", "B", "SEC EDGAR 10-K competition section", "open"),
    ],
    "devtools": [
        Source("gartner", "A", "Gartner Peer Insights market roster", "manual"),
        Source("g2cat", "A", "G2 category roster (exhaustive, via G2 Data API)", "open"),
        Source("g2", "B", "G2 category page", "manual"),
        Source("g2alt", "B", "G2 product 'alternatives' list", "open"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
        Source("wikidata", "B", "Wikidata industry / instance-of", "open"),
        Source("wikipedia", "B", "English Wikipedia article (independent editorial)", "open"),
        Source("sec", "B", "SEC EDGAR 10-K competition section", "open"),
    ],
    "ecommerce": [
        Source("gartner", "A", "Gartner Peer Insights market roster", "manual"),
        Source("g2cat", "A", "G2 category roster (exhaustive, via G2 Data API)", "open"),
        Source("g2", "B", "G2 category page", "manual"),
        Source("g2alt", "B", "G2 product 'alternatives' list", "open"),
        Source("capterra", "B", "Capterra category page", "manual"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
        Source("wikidata", "B", "Wikidata industry / instance-of", "open"),
        Source("wikipedia", "B", "English Wikipedia article (independent editorial)", "open"),
        Source("sec", "B", "SEC EDGAR 10-K competition section", "open"),
    ],
    "healthtech": [
        Source("gartner", "A", "Gartner Peer Insights market roster", "manual"),
        Source("g2cat", "A", "G2 category roster (exhaustive, via G2 Data API)", "open"),
        Source("g2", "B", "G2 category page", "manual"),
        Source("g2alt", "B", "G2 product 'alternatives' list", "open"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
        Source("wikidata", "B", "Wikidata industry / instance-of", "open"),
        Source("wikipedia", "B", "English Wikipedia article (independent editorial)", "open"),
        Source("sec", "B", "SEC EDGAR 10-K competition section", "open"),
    ],
    "home-services": [
        Source("gartner", "A", "Gartner Field Service Management market", "manual"),
        Source("g2cat", "A", "G2 category roster (exhaustive, via G2 Data API)", "open"),
        Source("g2", "B", "G2 category page", "manual"),
        Source("g2alt", "B", "G2 product 'alternatives' list", "open"),
        Source("franchise", "B", "Franchise 500 / Franchise Times Top 400", "manual"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
    ],
    "trades": [
        Source("trade", "A", "Trade-press league table (RER 100, etc.)", "manual"),
        Source("ibisworld", "A", "IBISWorld market-share leaders", "manual"),
        Source("association", "B", "Trade association directory (ACCA, PHCC, ARA)", "manual"),
        Source("franchise", "B", "Franchise 500 / Franchise Times Top 400", "manual"),
        Source("naics", "B", "NAICS-code-enumerated peers", "manual"),
    ],
    "real-estate": [
        Source("trade", "A", "Trade league table (NMHC Top 50, etc.)", "manual"),
        Source("association", "B", "Industry association directory", "manual"),
        Source("crunchbase", "B", "Crunchbase competitors", "manual"),
        Source("naics", "B", "NAICS-code-enumerated peers", "manual"),
    ],
}


def tier_of(category: str, source_id: str) -> Tier | None:
    """Tier of a source id within a category. A source id may be namespaced
    with a suffix (e.g. `g2:category:sms-marketing`); we match on the prefix
    registered for the category."""
    for src in SOURCES.get(category, []):
        if source_id == src.id or source_id.startswith(src.id + ":"):
            return src.tier
    return None


def qualifies(category: str, source_ids: list[str]) -> tuple[bool, str]:
    """Apply the inclusion rule. Returns (ok, reason).

    >=1 Tier-A OR >=2 distinct Tier-B (counting distinct source ids).
    Unknown source ids (not in the registry for this category) are ignored and
    flagged in the reason so the QA pass can catch typos / off-registry sources.
    """
    distinct = list(dict.fromkeys(source_ids))  # de-dupe, keep order
    tier_a = [s for s in distinct if tier_of(category, s) == "A"]
    tier_b = [s for s in distinct if tier_of(category, s) == "B"]
    unknown = [s for s in distinct if tier_of(category, s) is None]

    if tier_a:
        ok, reason = True, f"Tier-A: {tier_a[0]}"
    elif len(tier_b) >= 2:
        ok, reason = True, f"2x Tier-B: {', '.join(tier_b[:2])}"
    else:
        ok = False
        reason = (
            f"insufficient: tierA={tier_a} tierB={tier_b} "
            f"(need >=1 A or >=2 B)"
        )
    if unknown:
        reason += f" | WARNING off-registry sources ignored: {unknown}"
    return ok, reason


def category_template_sources(category: str) -> list[dict]:
    """Sanctioned sources for a category, as dicts ready to drop into a raw
    working file's `sources_used` for a human to fill in URLs."""
    return [
        {"id": s.id, "tier": s.tier, "name": s.name, "access": s.access,
         "url": "", "captured_at": ""}
        for s in SOURCES.get(category, [])
    ]
