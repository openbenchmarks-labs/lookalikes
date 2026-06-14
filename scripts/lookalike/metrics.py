"""Pluggable metric registry for the lookalike benchmark.

A metric is a named function over a cell's judged candidates. `precision_at_k`
is the primary metric (drives best-of-config selection and the leaderboard) and
is byte-identical to the original `JudgedRun.precision_at_k`. New metrics
(recall@k, nDCG, MAP, MRR) drop in by adding a `Metric` to `METRIC_REGISTRY` —
no changes to the runner, orchestrator, or DB schema.

A metric fn takes the ordered list of `JudgedCandidate`s (rank order) plus the
requested K, and returns `(value | None, detail_dict)`. Metrics flagged
`seed_context=True` (recall) need a cross-vendor relevance pool and are computed
in a second pass via `compute_seed_context_metric`.

`relevant` on each `JudgedCandidate` is the post-aggregation (majority) label, so
every metric is judge-panel agnostic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .common import JudgedCandidate

# fn(judged, k) -> (value, detail). seed_context metrics are invoked separately.
MetricFn = Callable[[list[JudgedCandidate], int], "tuple[Optional[float], dict[str, Any]]"]


RECALL_KS = (10, 50, 100)  # recall-cohort fetch/eval depths


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str           # "percent" | "score" | "count"
    direction: str      # "higher_is_better" | "lower_is_better"
    definition: str
    fn: MetricFn
    primary: bool = False
    seed_context: bool = False  # needs a cross-vendor relevance pool (2nd pass)
    gold_context: bool = False  # needs the frozen gold set + gold_size (2nd pass, judge-free)


def _labels(judged: list[JudgedCandidate]) -> list[int]:
    """Relevance (0/1) in rank order. Candidates carry an explicit rank for some
    vendors; fall back to list order otherwise (matches take_top numbering)."""
    ordered = sorted(
        enumerate(judged),
        key=lambda iv: (iv[1].candidate.rank if iv[1].candidate.rank is not None else iv[0] + 1),
    )
    return [1 if jc.relevant else 0 for _, jc in ordered]


def _precision_at_k(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    n = len(judged)
    if n == 0:
        return None, {}
    relevant = sum(1 for j in judged if j.relevant)
    # Denominator is the number actually judged (== returned, capped at k),
    # byte-identical to the original JudgedRun.precision_at_k.
    return round(100.0 * relevant / n, 2), {"relevant": relevant, "judged": n}


def _relevant_count(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    return float(sum(1 for j in judged if j.relevant)), {}


def _ndcg_at_k(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    labels = _labels(judged)[:k]
    if not labels:
        return None, {}
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0, {"note": "no relevant candidates"}
    return round(dcg / idcg, 4), {}


def _map_at_k(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    labels = _labels(judged)[:k]
    if not labels:
        return None, {}
    hits = 0
    summed = 0.0
    for i, g in enumerate(labels):
        if g:
            hits += 1
            summed += hits / (i + 1)
    if hits == 0:
        return 0.0, {}
    return round(summed / hits, 4), {}


def _mrr(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    labels = _labels(judged)[:k]
    for i, g in enumerate(labels):
        if g:
            return round(1.0 / (i + 1), 4), {"first_relevant_rank": i + 1}
    return 0.0, {}


def _recall_unavailable(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    # Recall needs the cross-vendor pool; use compute_seed_context_metric instead.
    return None, {"note": "recall is computed in a second pass with the seed pool"}


def _gold_metric_unavailable(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    # Gold-context metrics need the frozen gold set + gold_size (not in MetricFn's
    # signature); they are computed via compute_gold_recall_metrics in a 2nd pass.
    return None, {"note": "computed in a second pass against the frozen gold set"}


def _gold_metric(key: str, label: str, unit: str, definition: str) -> "Metric":
    return Metric(
        key=key, label=label, unit=unit, direction="higher_is_better",
        definition=definition, fn=_gold_metric_unavailable, gold_context=True,
    )


METRIC_REGISTRY: dict[str, Metric] = {
    "precision_at_k": Metric(
        key="precision_at_k", label="Precision@K", unit="percent",
        direction="higher_is_better",
        definition="Fraction of the K returned companies the judge panel (majority vote) deemed relevant.",
        fn=_precision_at_k, primary=True,
    ),
    "relevant_count": Metric(
        key="relevant_count", label="Relevant", unit="count",
        direction="higher_is_better",
        definition="Count of returned companies judged relevant (majority vote).",
        fn=_relevant_count,
    ),
    "ndcg_at_k": Metric(
        key="ndcg_at_k", label="nDCG@K", unit="score",
        direction="higher_is_better",
        definition="Normalized DCG over the vendor's ranking, binary gain from judge-majority labels (not graded relevance).",
        fn=_ndcg_at_k,
    ),
    "map_at_k": Metric(
        key="map_at_k", label="MAP@K", unit="score",
        direction="higher_is_better",
        definition="Mean average precision over the vendor's ranking, judge-majority labels.",
        fn=_map_at_k,
    ),
    "mrr": Metric(
        key="mrr", label="MRR", unit="score",
        direction="higher_is_better",
        definition="Reciprocal rank of the first judge-relevant company.",
        fn=_mrr,
    ),
    "recall_at_k": Metric(
        key="recall_at_k", label="Recall@K (pooled)", unit="percent",
        direction="higher_is_better",
        definition="Relative pooled recall (not absolute): relevant returned / distinct relevant companies any vendor surfaced for the seed.",
        fn=_recall_unavailable, seed_context=True,
    ),
    # --- TAM-recall (gold-context, judge-free). Recall is against a frozen,
    # vendor-independent REFERENCE set (NOT absolute TAM); denominator = |reference
    # set|. Distinct `*_vs_ref` keys so they never collide with the pooled
    # `recall_at_k` above and never run in the precision first pass. ---
    "recall_vs_ref_at_10": _gold_metric(
        "recall_vs_ref_at_10", "Recall@10 (vs ref)", "percent",
        "Recall against a frozen, vendor-independent reference set, top-10: gold companies found in the first 10 results / |reference set|.",
    ),
    "recall_vs_ref_at_50": _gold_metric(
        "recall_vs_ref_at_50", "Recall@50 (vs ref)", "percent",
        "Recall against a frozen, vendor-independent reference set, top-50 / |reference set|.",
    ),
    "recall_vs_ref_at_100": _gold_metric(
        "recall_vs_ref_at_100", "Recall@100 (vs ref)", "percent",
        "Recall against a frozen, vendor-independent reference set, top-100 / |reference set|. The recall leaderboard's primary sort key.",
    ),
    "r_precision_vs_ref": _gold_metric(
        "r_precision_vs_ref", "R-Precision (vs ref)", "percent",
        "Precision@R where R = |reference set| (the precision-recall break-even point) against the frozen reference set.",
    ),
    "hit_at_100": _gold_metric(
        "hit_at_100", "Hit@100 (vs ref)", "score",
        "1 if at least one reference-set company appears in the top 100, else 0 (averaged → hit rate on the leaderboard).",
    ),
    "coverage_vs_ref": _gold_metric(
        "coverage_vs_ref", "Coverage (vs ref)", "percent",
        "Fraction of the reference set found anywhere in the returned list (matched / |reference set|).",
    ),
    "matched_count": _gold_metric(
        "matched_count", "Matched", "count",
        "Count of reference-set companies the vendor surfaced anywhere in its returned list.",
    ),
}

# Keys written by the gold-context second pass (compute_gold_recall_metrics).
RECALL_METRIC_KEYS = (
    "recall_vs_ref_at_10", "recall_vs_ref_at_50", "recall_vs_ref_at_100",
    "r_precision_vs_ref", "hit_at_100", "coverage_vs_ref", "matched_count",
)


def primary_metric() -> Metric:
    for m in METRIC_REGISTRY.values():
        if m.primary:
            return m
    raise RuntimeError("no primary metric registered")


def compute_cell_metrics(judged: list[JudgedCandidate], k: int) -> dict[str, tuple[Optional[float], dict]]:
    """All first-pass metrics for one cell (excludes seed-context pooled recall and
    gold-context recall, which need cross-cell / gold-set data in a second pass)."""
    return {
        key: m.fn(judged, k)
        for key, m in METRIC_REGISTRY.items()
        if not m.seed_context and not m.gold_context
    }


def compute_gold_recall_metrics(
    *,
    recall_at: dict[int, Optional[float]],
    r_precision: Optional[float],
    hit_at_max: Optional[int],
    matched_count: Optional[int],
    gold_size: int,
) -> dict[str, tuple[Optional[float], dict]]:
    """Second pass for the recall cohort: map the deterministic gold-overlap values
    (computed by recall.py against the frozen reference set — see
    RECALL_METHODOLOGY.md) onto the `*_vs_ref` metric keys for persistence. Single
    source of truth for the math stays in recall.py; this only labels + packages.

    `recall_at` is keyed by K (10/50/100); `hit_at_max` is Hit@100 (0/1);
    `gold_size` = |reference set| (the recall denominator)."""
    detail = {"gold_size": gold_size, "matched": matched_count}
    coverage = (
        round(100.0 * matched_count / gold_size, 2)
        if gold_size and matched_count is not None
        else None
    )
    return {
        "recall_vs_ref_at_10": (recall_at.get(10), detail),
        "recall_vs_ref_at_50": (recall_at.get(50), detail),
        "recall_vs_ref_at_100": (recall_at.get(100), detail),
        "r_precision_vs_ref": (r_precision, detail),
        "hit_at_100": (None if hit_at_max is None else float(hit_at_max), detail),
        "coverage_vs_ref": (coverage, detail),
        "matched_count": (None if matched_count is None else float(matched_count), detail),
    }


def compute_seed_context_metric(
    metric_key: str, judged: list[JudgedCandidate], k: int, relevant_pool: set[str]
) -> tuple[Optional[float], dict]:
    """Second-pass metric needing the cross-vendor relevant-domain pool for a seed.
    `relevant_pool` = distinct domains any vendor returned for this seed that were
    judged relevant (majority)."""
    if metric_key == "recall_at_k":
        if not relevant_pool:
            return None, {"pool": 0}
        my_relevant = {
            (j.candidate.domain or "").lower()
            for j in judged
            if j.relevant and j.candidate.domain
        }
        hit = len(my_relevant & relevant_pool)
        return round(100.0 * hit / len(relevant_pool), 2), {"pool": len(relevant_pool), "hit": hit}
    raise KeyError(metric_key)
