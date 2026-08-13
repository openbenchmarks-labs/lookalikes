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
    extruct,
    cufinder,
    discolike,
    lusha,
    mock,
    ocean,
    parallel,
    predictleads,
    zoominfo,
)

# One vendor intentionally absent from REGISTRY:
#   - Lusha:    POST /v3/companies/lookalike requires 5-100 seeds per
#               request. Our benchmark scores one seed per cell so the
#               input contracts don't line up — Lusha would either need
#               its own multi-seed cell design (different units) or we'd
#               have to inject 4 padding seeds per call, which would
#               leak signal across cells.
# It is surfaced under NOT_SURVEYED_PROVIDERS on the page so the omission
# stays visible to readers.
#
# ZoomInfo runs through the `gtm` CLI rather than an HTTP API, so it needs the
# binary on PATH and a ZoomInfo contract; the runner fails the cell cleanly
# when `gtm` is missing rather than skipping it silently.
REGISTRY = {
    ocean.VENDOR_SLUG: ocean,
    exa.VENDOR_SLUG: exa,
    extruct.VENDOR_SLUG: extruct,
    cufinder.VENDOR_SLUG: cufinder,
    discolike.VENDOR_SLUG: discolike,
    parallel.VENDOR_SLUG: parallel,
    predictleads.VENDOR_SLUG: predictleads,
    zoominfo.VENDOR_SLUG: zoominfo,
}
