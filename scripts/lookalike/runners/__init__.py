"""Per-vendor runners. Each module exposes:

    VENDOR_SLUG: str
    VENDOR_NAME: str
    CONFIGS: list[dict]                       # named knob-sets to sweep
    def run(seed, k, config) -> RunResult     # one API call → top-K candidates

The orchestrator imports each runner by slug and sweeps `CONFIGS`,
picking the one that maximizes Precision@K for the seed.
"""
from . import (  # noqa: F401
    exa,
    lusha,
    mock,
    ocean,
    openfunnel,
    parallel,
    predictleads,
)

# Two vendors intentionally absent from REGISTRY:
#   - ZoomInfo: company-lookalike API is not on self-serve (sales-gated).
#   - Lusha:    POST /v3/companies/lookalike requires 5-100 seeds per
#               request. Our benchmark scores one seed per cell so the
#               input contracts don't line up — Lusha would either need
#               its own multi-seed cell design (different units) or we'd
#               have to inject 4 padding seeds per call, which would
#               leak signal across cells.
# Both are surfaced under NOT_SURVEYED_PROVIDERS on the page so the
# omission stays visible to readers.
REGISTRY = {
    openfunnel.VENDOR_SLUG: openfunnel,
    ocean.VENDOR_SLUG: ocean,
    exa.VENDOR_SLUG: exa,
    parallel.VENDOR_SLUG: parallel,
    predictleads.VENDOR_SLUG: predictleads,
}
