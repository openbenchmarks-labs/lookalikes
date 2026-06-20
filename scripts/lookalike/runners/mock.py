"""Mock runner — deterministic, offline, no API key required.

Generates K plausible-looking candidate companies per seed by walking a
small per-vertical pool. Combined with the mock judge in `judge.py`,
this lets us populate the snapshot end-to-end without burning vendor
credits or OpenAI tokens.

Not registered in `runners/__init__.REGISTRY`. The orchestrator wires
it in directly when `--mock` is passed.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..common import Candidate, RunResult, Seed, take_top

CONFIGS: list[dict[str, Any]] = [{"name": "mock", "filters": {}}]

# Small per-vertical candidate pools. The mock runner deterministically
# shuffles into K candidates per seed so the matrix has actual companies
# (not Latin placeholders) but is obviously seeded — judged_at + the
# vendor-shared "mock" prefix tag the rows as non-live in tooltips.
POOLS: dict[str, list[tuple[str, str]]] = {
    # narrow B2B SaaS — customer support / RevOps / inbound plumbing.
    # Tuned for Pylon + Default; intentionally smaller and more vertical
    # than horizontal "Notion / Asana" giants.
    "b2b-saas": [
        ("Plain", "plain.com"),
        ("Help Scout", "helpscout.com"),
        ("Front", "front.com"),
        ("Intercom", "intercom.com"),
        ("Zendesk", "zendesk.com"),
        ("Crisp", "crisp.chat"),
        ("Chatwoot", "chatwoot.com"),
        ("Chili Piper", "chilipiper.com"),
        ("Calendly", "calendly.com"),
        ("RevenueHero", "revenuehero.io"),
        ("LeanData", "leandatainc.com"),
        ("Distribute", "distribute.so"),
        ("Lemkin", "lemkinhq.com"),
        ("Forwrd.ai", "forwrd.ai"),
        ("Pocus", "pocus.com"),
    ],
    # devtools — realtime collab + background-job adjacencies for
    # Liveblocks + Trigger.dev.
    "devtools": [
        ("Inngest", "inngest.com"),
        ("Hatchet", "hatchet.run"),
        ("Defer", "defer.run"),
        ("Mergent", "mergent.co"),
        ("Temporal", "temporal.io"),
        ("Hookdeck", "hookdeck.com"),
        ("River", "riverqueue.com"),
        ("Ably", "ably.com"),
        ("Pusher", "pusher.com"),
        ("PubNub", "pubnub.com"),
        ("PartyKit", "partykit.io"),
        ("Yjs", "yjs.dev"),
        ("Replicache", "replicache.dev"),
        ("Convex", "convex.dev"),
        ("Hocuspocus", "tiptap.dev/hocuspocus"),
    ],
    # ecommerce — DTC infra (SMS, subscriptions, retention) for
    # Postscript + Recharge.
    "ecommerce": [
        ("Attentive", "attentive.com"),
        ("Klaviyo", "klaviyo.com"),
        ("Yotpo", "yotpo.com"),
        ("Omnisend", "omnisend.com"),
        ("Voyage SMS", "voyagesms.com"),
        ("Recart", "recart.com"),
        ("Tone", "tone.email"),
        ("Bold Subscriptions", "boldcommerce.com"),
        ("Skio", "skio.com"),
        ("Awtomic", "awtomic.com"),
        ("Smartrr", "smartrr.com"),
        ("Stay AI", "stay.ai"),
        ("Loop Subscriptions", "loopwork.co"),
        ("Subbly", "subbly.co"),
        ("Appstle", "appstle.com"),
    ],
    # healthtech — digital MSK / physical-therapy platforms + value-based
    # primary care adjacencies for Hinge Health + Aledade.
    "healthtech": [
        ("Sword Health", "swordhealth.com"),
        ("Kaia Health", "kaiahealth.com"),
        ("Omada Health", "omadahealth.com"),
        ("Vida Health", "vida.com"),
        ("Lyra Health", "lyrahealth.com"),
        ("Spring Health", "springhealth.com"),
        ("Livongo", "livongo.com"),
        ("Big Health", "bighealth.com"),
        ("Privia Health", "priviahealth.com"),
        ("Agilon Health", "agilonhealth.com"),
        ("Iora Health", "iorahealth.com"),
        ("VillageMD", "villagemd.com"),
        ("Oak Street Health", "oakstreethealth.com"),
        ("Pearl Health", "pearlhealth.com"),
        ("ChenMed", "chenmed.com"),
    ],
    # home-services SaaS + national chains — vertical SaaS for contractors
    # (ServiceTitan-class) and multi-location service brands
    # (Roto-Rooter-class). Separate from "trades" which is independent
    # local contractors.
    "home-services": [
        # vertical SaaS for trades
        ("Jobber", "getjobber.com"),
        ("Housecall Pro", "housecallpro.com"),
        ("FieldEdge", "fieldedge.com"),
        ("BuildOps", "buildops.com"),
        ("ServiceFusion", "servicefusion.com"),
        ("FieldRoutes", "fieldroutes.com"),
        ("Workiz", "workiz.com"),
        ("FieldPulse", "fieldpulse.com"),
        # multi-location HVAC + plumbing service brands
        ("Mr. Rooter Plumbing", "mrrooter.com"),
        ("Benjamin Franklin Plumbing", "benjaminfranklinplumbing.com"),
        ("ARS / Rescue Rooter", "ars.com"),
        ("One Hour Heating & Air", "onehourheatandair.com"),
        ("Service Champions", "servicechampions.com"),
        ("Goettl Air Conditioning & Plumbing", "goettl.com"),
        ("Len The Plumber", "lentheplumber.com"),
    ],
    # local trades — independent residential HVAC / plumbing / electrical
    # service contractors. Single-location or small regional businesses.
    # Adjacencies for Point Loma, JDV Electric, etc.
    "trades": [
        ("Mr. Electric of Salem", "mrelectric.com/salem"),
        ("Bright Eye Electric", "brighteyeelectric.com"),
        ("Mister Sparky Electric", "mistersparky.com"),
        ("Power Pro Plumbing Heating & Air", "powerproplumbing.com"),
        ("West Coast Heating, Air & Plumbing", "westcoastair.com"),
        ("Bonney Plumbing", "bonney.com"),
        ("Donnelly's Plumbing Heating & Cooling", "donnellysphc.com"),
        ("Anthony Plumbing, Heating, Cooling & Electric", "anthonyphce.com"),
        ("L.J. Kruse Co.", "ljkruseco.com"),
        ("Reliable & Affordable Plumbing", "reliable-aff.com"),
        ("Albert Culver Co.", "albertculver.com"),
        ("McMurray Plumbing", "mcmurrayplumbing.com"),
        ("Lansing Heating & Air Conditioning", "lansingheating.com"),
        ("Vincent's Heating & Plumbing", "vincentsheating.com"),
        ("Sun Glow Heating & Air", "sun-glow.com"),
    ],
    # real estate — multifamily property management operators. Both Emerge
    # Living and the rest of this pool are vertically-integrated apartment
    # operators / property managers in the Sunbelt + national markets.
    "real-estate": [
        ("Cortland", "cortland.com"),
        ("RPM Living", "rpmliving.com"),
        ("Bozzuto", "bozzuto.com"),
        ("Asset Living", "assetliving.com"),
        ("BH Companies", "bhmanagement.com"),
        ("Greystar", "greystar.com"),
        ("Mill Creek Residential", "millcreekplaces.com"),
        ("AION Management", "aionmgmt.com"),
        ("ZRS Management", "zrsmanagement.com"),
        ("FPI Management", "fpimgt.com"),
        ("Pinnacle (Cushman & Wakefield)", "trustpinnacle.com"),
        ("Highmark Residential", "highmarkres.com"),
        ("Camden Property Trust", "camdenliving.com"),
        ("Maa", "maac.com"),
        ("UDR", "udr.com"),
    ],
    "fintech": [
        ("Ramp", "ramp.com"),
        ("Mercury", "mercury.com"),
        ("Wise", "wise.com"),
        ("Chime", "chime.com"),
        ("Monzo", "monzo.com"),
        ("N26", "n26.com"),
        ("Airwallex", "airwallex.com"),
        ("Adyen", "adyen.com"),
        ("Checkout.com", "checkout.com"),
        ("Plaid", "plaid.com"),
        ("Marqeta", "marqeta.com"),
        ("Bill", "bill.com"),
        ("Spendesk", "spendesk.com"),
        ("Rho", "rho.co"),
        ("Qonto", "qonto.com"),
    ],
    "cybersecurity": [
        ("SentinelOne", "sentinelone.com"),
        ("Palo Alto Networks", "paloaltonetworks.com"),
        ("Zscaler", "zscaler.com"),
        ("Okta", "okta.com"),
        ("Wiz", "wiz.io"),
        ("Snyk", "snyk.io"),
        ("Tenable", "tenable.com"),
        ("Rapid7", "rapid7.com"),
        ("Cloudflare", "cloudflare.com"),
        ("Check Point", "checkpoint.com"),
        ("Fortinet", "fortinet.com"),
        ("Sophos", "sophos.com"),
        ("Arctic Wolf", "arcticwolf.com"),
        ("Huntress", "huntress.com"),
        ("Bitdefender", "bitdefender.com"),
    ],
    "industrial": [
        ("John Deere", "deere.com"),
        ("Komatsu", "komatsu.com"),
        ("Volvo Construction Equipment", "volvoce.com"),
        ("Hitachi Construction Machinery", "hitachicm.com"),
        ("CNH Industrial", "cnhindustrial.com"),
        ("ABB", "abb.com"),
        ("Schneider Electric", "se.com"),
        ("Rockwell Automation", "rockwellautomation.com"),
        ("Emerson", "emerson.com"),
        ("Honeywell", "honeywell.com"),
        ("Bosch Rexroth", "boschrexroth.com"),
        ("Mitsubishi Electric", "mitsubishielectric.com"),
        ("Danfoss", "danfoss.com"),
        ("Parker Hannifin", "parker.com"),
        ("Eaton", "eaton.com"),
    ],
    "logistics": [
        ("UPS", "ups.com"),
        ("DHL", "dhl.com"),
        ("Maersk", "maersk.com"),
        ("XPO", "xpo.com"),
        ("C.H. Robinson", "chrobinson.com"),
        ("J.B. Hunt", "jbhunt.com"),
        ("Ryder", "ryder.com"),
        ("Flexport", "flexport.com"),
        ("Gojek", "gojek.com"),
        ("DoorDash", "doordash.com"),
        ("Uber", "uber.com"),
        ("Lyft", "lyft.com"),
        ("Delhivery", "delhivery.com"),
        ("Lalamove", "lalamove.com"),
        ("Ninja Van", "ninjavan.co"),
    ],
    "hospitality": [
        ("Hilton", "hilton.com"),
        ("Hyatt", "hyatt.com"),
        ("IHG Hotels & Resorts", "ihg.com"),
        ("Accor", "accor.com"),
        ("Wyndham Hotels & Resorts", "wyndhamhotels.com"),
        ("Choice Hotels", "choicehotels.com"),
        ("Airbnb", "airbnb.com"),
        ("Booking.com", "booking.com"),
        ("Expedia Group", "expediagroup.com"),
        ("Olo", "olo.com"),
        ("Square for Restaurants", "squareup.com"),
        ("Clover", "clover.com"),
        ("Lightspeed", "lightspeedhq.com"),
        ("SevenRooms", "sevenrooms.com"),
        ("Resy", "resy.com"),
    ],
    "energy": [
        ("Duke Energy", "duke-energy.com"),
        ("Southern Company", "southerncompany.com"),
        ("Dominion Energy", "dominionenergy.com"),
        ("Exelon", "exeloncorp.com"),
        ("Enel", "enel.com"),
        ("Iberdrola", "iberdrola.com"),
        ("Orsted", "orsted.com"),
        ("First Solar", "firstsolar.com"),
        ("Sunrun", "sunrun.com"),
        ("AES", "aes.com"),
        ("Constellation Energy", "constellationenergy.com"),
        ("National Grid", "nationalgrid.com"),
        ("Xcel Energy", "xcelenergy.com"),
        ("Eversource", "eversource.com"),
        ("Entergy", "entergy.com"),
    ],
}


def make_runner(provider_slug: str, provider_name: str):
    """Build a vendor-specific mock runner closure. Each vendor's mock
    output is deterministic *and* slightly different from the others (so
    the matrix doesn't look identical row-by-row)."""

    def run(seed: Seed, k: int, config: dict[str, Any]) -> RunResult:
        pool = list(POOLS.get(seed.category, []))
        if not pool:
            return RunResult(
                seed_slug=seed.seed_slug,
                provider_slug=provider_slug,
                config_name=config["name"],
                config=config,
                candidates=[],
                latency_ms=120,
                error=f"mock: no pool for category {seed.category}",
            )
        # Deterministic shuffle per (vendor, seed).
        keyed = sorted(
            pool,
            key=lambda nd: hashlib.sha256(
                f"{provider_slug}|{seed.seed_slug}|{nd[1]}".encode("utf-8")
            ).hexdigest(),
        )
        # Drop the seed itself if present.
        keyed = [
            (n, d)
            for (n, d) in keyed
            if d.lower() != (seed.seed_domain or "").lower()
        ]
        cands = [
            Candidate(
                name=n,
                domain=d,
                description=f"Mock candidate from {seed.category} pool — {n}",
                extra={"vendor": provider_slug},
            )
            for (n, d) in keyed
        ]
        return RunResult(
            seed_slug=seed.seed_slug,
            provider_slug=provider_slug,
            config_name=config["name"],
            config=config,
            candidates=take_top(cands, k),
            latency_ms=180 + (hash(provider_slug) % 200),
            cost_usd=0.0,
            requested_k=k,
        )

    return run
