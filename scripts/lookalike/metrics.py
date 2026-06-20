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


def _ordered_judged(judged: list[JudgedCandidate]) -> list[JudgedCandidate]:
    """Relevance (0/1) in rank order. Candidates carry an explicit rank for some
    vendors; fall back to list order otherwise (matches take_top numbering)."""
    return [
        jc
        for _, jc in sorted(
            enumerate(judged),
            key=lambda iv: (iv[1].candidate.rank if iv[1].candidate.rank is not None else iv[0] + 1),
        )
    ]


def _labels(judged: list[JudgedCandidate]) -> list[int]:
    return [1 if jc.relevant else 0 for jc in _ordered_judged(judged)]


def _precision_at_n(judged: list[JudgedCandidate], n: int) -> tuple[Optional[float], dict]:
    top = _ordered_judged(judged)[:n]
    if n <= 0:
        return None, {}
    relevant = sum(1 for j in top if j.relevant)
    return round(100.0 * relevant / n, 2), {
        "relevant": relevant,
        "judged": len(top),
        "denominator": n,
        "cutoff": n,
    }


def _relevant_count_at_n(judged: list[JudgedCandidate], n: int) -> tuple[Optional[float], dict]:
    top = _ordered_judged(judged)[:n]
    return float(sum(1 for j in top if j.relevant)), {"judged": len(top), "cutoff": n}


def _make_precision_at_n(n: int) -> MetricFn:
    def _fn(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
        return _precision_at_n(judged, n)

    return _fn


def _make_relevant_count_at_n(n: int) -> MetricFn:
    def _fn(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
        return _relevant_count_at_n(judged, n)

    return _fn


def _precision_at_k(judged: list[JudgedCandidate], k: int) -> tuple[Optional[float], dict]:
    return _precision_at_n(judged, k)


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



METRIC_REGISTRY: dict[str, Metric] = {
    "precision_at_k": Metric(
        key="precision_at_k", label="Precision@K", unit="percent",
        direction="higher_is_better",
        definition="Fraction of the K returned companies the judge panel (majority vote) deemed relevant.",
        fn=_precision_at_k, primary=True,
    ),
    "precision_at_10": Metric(
        key="precision_at_10", label="Precision@10", unit="percent",
        direction="higher_is_better",
        definition="Fraction of the top 10 returned companies the judge panel deemed relevant.",
        fn=_make_precision_at_n(10),
    ),
    "precision_at_50": Metric(
        key="precision_at_50", label="Precision@50", unit="percent",
        direction="higher_is_better",
        definition="Fraction of the top 50 returned companies the judge panel deemed relevant.",
        fn=_make_precision_at_n(50),
    ),
    "precision_at_100": Metric(
        key="precision_at_100", label="Precision@100", unit="percent",
        direction="higher_is_better",
        definition="Fraction of the top 100 returned companies the judge panel deemed relevant.",
        fn=_make_precision_at_n(100),
    ),
    "relevant_count": Metric(
        key="relevant_count", label="Relevant", unit="count",
        direction="higher_is_better",
        definition="Count of returned companies judged relevant (majority vote).",
        fn=_relevant_count,
    ),
    "relevant_count_at_10": Metric(
        key="relevant_count_at_10", label="Relevant@10", unit="count",
        direction="higher_is_better",
        definition="Count of top-10 returned companies judged relevant.",
        fn=_make_relevant_count_at_n(10),
    ),
    "relevant_count_at_50": Metric(
        key="relevant_count_at_50", label="Relevant@50", unit="count",
        direction="higher_is_better",
        definition="Count of top-50 returned companies judged relevant.",
        fn=_make_relevant_count_at_n(50),
    ),
    "relevant_count_at_100": Metric(
        key="relevant_count_at_100", label="Relevant@100", unit="count",
        direction="higher_is_better",
        definition="Count of top-100 returned companies judged relevant.",
        fn=_make_relevant_count_at_n(100),
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
}


def primary_metric() -> Metric:
    for m in METRIC_REGISTRY.values():
        if m.primary:
            return m
    raise RuntimeError("no primary metric registered")


def compute_cell_metrics(judged: list[JudgedCandidate], k: int) -> dict[str, tuple[Optional[float], dict]]:
    """All first-pass metrics for one cell
    (excludes seed-context pooled recall, which needs cross-cell data)."""
    return {
        key: m.fn(judged, k)
        for key, m in METRIC_REGISTRY.items()
        if not m.seed_context
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
