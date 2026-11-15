"""MAP@12, the H&M competition metric.

For each customer with at least one purchase in the evaluated week:
  AP@k = (1 / min(|actual|, k)) * sum_{r=1..k} P(r) * rel(r)
where P(r) is precision over the first r recommendations and rel(r) is 1 if the r-th recommendation
was bought. MAP@k is the mean AP@k over those customers (customers with no purchases are not scored).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl


def ap_at_k(actual: Iterable[int], predicted: Sequence[int], k: int = 12) -> float:
    """Reference implementation for one customer (used by the tests)."""
    act = set(actual)
    if not act:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for r, p in enumerate(predicted[:k], start=1):
        if p in act and p not in seen:
            hits += 1
            score += hits / r
        seen.add(p)
    return score / min(len(act), k)


def map_at_k(pred: pl.DataFrame, actual: pl.DataFrame, k: int = 12) -> float:
    """Vectorised MAP@k.

    pred:   customer, article_id, rank (1 = best). Duplicate (customer, article) rows are dropped.
    actual: customer, article_id (purchases in the evaluated week; duplicates ignored).
    """
    act = actual.select("customer", "article_id").unique()
    n_act = act.group_by("customer").agg(pl.len().alias("n_act"))
    p = (
        pred.select("customer", "article_id", "rank")
        .sort(["customer", "rank"])
        .unique(["customer", "article_id"], keep="first", maintain_order=True)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").alias("r"))
        .filter(pl.col("r") <= k)
    )
    hits = (
        p.join(act, on=["customer", "article_id"], how="inner")
        .sort(["customer", "r"])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").alias("hit_no"))
        .group_by("customer")
        .agg((pl.col("hit_no") / pl.col("r")).sum().alias("prec_sum"))
    )
    scored = n_act.join(hits, on="customer", how="left").with_columns(
        (pl.col("prec_sum").fill_null(0.0) / pl.min_horizontal(pl.col("n_act"), pl.lit(k))).alias("ap")
    )
    mean = scored["ap"].mean()
    return float(mean) if isinstance(mean, int | float) else 0.0


def candidate_recall(cands: pl.DataFrame, actual: pl.DataFrame) -> float:
    """Share of (customer, article) purchases that appear among the candidates."""
    act = actual.select("customer", "article_id").unique()
    if act.height == 0:
        return 0.0
    covered = act.join(
        cands.select("customer", "article_id").unique(), on=["customer", "article_id"], how="semi"
    )
    return covered.height / act.height
