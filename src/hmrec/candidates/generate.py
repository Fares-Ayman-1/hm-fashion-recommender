"""Stage 1: candidate generation for one target week W.

Every function only receives transactions from weeks < W (``hist``), so nothing from the target week
can leak in. Each source returns (customer, article_id, <source features>); ``generate`` merges them
into one row per (customer, article) with a flag per source, so the ranker knows *why* an article
was suggested and the API can explain it.

Sources
  repurchase  the customer's own purchases in the last N weeks (fashion basics get re-bought)
  popular     top-N articles of the last week within the customer's age band
  cooccur     articles bought in the same week as the customer's recent purchases ("bought together")
  similar     nearest neighbours of recent purchases in CLIP image+text embedding space
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hmrec.config import CandidateConfig

AGE_LABELS = ["unknown", "<20", "20-24", "25-29", "30-39", "40-49", "50-59", "60+"]
SOURCES = ["repurchase", "popular", "cooccur", "similar"]


def add_band(customers: pl.DataFrame) -> pl.DataFrame:
    """Age band label per customer ('unknown' when age is missing)."""
    a = pl.col("age").cast(pl.Int16)
    band = (
        pl.when(a.is_null())
        .then(pl.lit("unknown"))
        .when(a < 20)
        .then(pl.lit("<20"))
        .when(a < 25)
        .then(pl.lit("20-24"))
        .when(a < 30)
        .then(pl.lit("25-29"))
        .when(a < 40)
        .then(pl.lit("30-39"))
        .when(a < 50)
        .then(pl.lit("40-49"))
        .when(a < 60)
        .then(pl.lit("50-59"))
        .otherwise(pl.lit("60+"))
    )
    return customers.with_columns(band.alias("band"))


def active_catalog(hist: pl.DataFrame, week: int, weeks: int) -> pl.DataFrame:
    return hist.filter(pl.col("week") >= week - weeks).select("article_id").unique()


def repurchase(hist: pl.DataFrame, targets: pl.DataFrame, week: int, cfg: CandidateConfig) -> pl.DataFrame:
    recent = hist.filter(pl.col("week") >= week - cfg.repurchase_weeks).join(
        targets, on="customer", how="semi"
    )
    return (
        recent.group_by("customer", "article_id")
        .agg(pl.len().alias("rep_count"), (week - pl.col("week").max()).cast(pl.Int16).alias("rep_weeks_ago"))
        .sort(
            ["customer", "rep_weeks_ago", "rep_count", "article_id"], descending=[False, False, True, False]
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").cast(pl.Int16).alias("rep_rank"))
        .filter(pl.col("rep_rank") <= cfg.repurchase_max)
    )


def popular_by_band(hist: pl.DataFrame, customers: pl.DataFrame, week: int, n: int) -> pl.DataFrame:
    """Top-n articles of week-1 per age band (unique buyers). Band 'unknown' uses the global ranking."""
    last = hist.filter(pl.col("week") == week - 1).join(
        add_band(customers).select("customer", "band"), on="customer", how="left"
    )

    def top(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("article_id")
            .agg(pl.col("customer").n_unique().alias("pop_count"))
            .sort(["pop_count", "article_id"], descending=[True, False])
            .head(n)
            .with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int16).alias("pop_rank"))
        )

    parts = [top(last).with_columns(pl.lit("unknown").alias("band"))]
    for band in [b for b in last["band"].unique().to_list() if b not in (None, "unknown")]:
        parts.append(top(last.filter(pl.col("band") == band)).with_columns(pl.lit(band).alias("band")))
    return pl.concat(parts)


def popular(
    hist: pl.DataFrame, customers: pl.DataFrame, targets: pl.DataFrame, week: int, cfg: CandidateConfig
) -> pl.DataFrame:
    table = popular_by_band(hist, customers, week, cfg.popular_n)
    tb = add_band(customers.join(targets, on="customer", how="semi")).select("customer", "band")
    return tb.join(table, on="band", how="inner").drop("band")


def seeds(hist: pl.DataFrame, targets: pl.DataFrame, n: int) -> pl.DataFrame:
    """Each target customer's ``n`` most recently bought distinct articles."""
    return (
        hist.join(targets, on="customer", how="semi")
        .group_by("customer", "article_id")
        .agg(pl.col("t_dat").max())
        .sort(["customer", "t_dat", "article_id"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").alias("seed_rank"))
        .filter(pl.col("seed_rank") <= n)
        .select("customer", "article_id")
    )


def cooccurrence_table(
    hist: pl.DataFrame, week: int, cfg: CandidateConfig, max_basket: int = 50
) -> pl.DataFrame:
    """article -> neighbour with P(neighbour bought in same week | article bought), top neighbours only."""
    baskets = (
        hist.filter(pl.col("week") >= week - cfg.cooccur_weeks)
        .select("customer", "week", "article_id")
        .unique()
    )
    sizes = baskets.group_by("customer", "week").agg(pl.len().alias("n"))
    baskets = baskets.join(sizes.filter(pl.col("n") <= max_basket), on=["customer", "week"], how="semi")
    pairs = (
        baskets.join(baskets, on=["customer", "week"], suffix="_n")
        .filter(pl.col("article_id") != pl.col("article_id_n"))
        .group_by("article_id", "article_id_n")
        .agg(pl.len().alias("pair_count"))
    )
    item_count = baskets.group_by("article_id").agg(pl.len().alias("item_count"))
    return (
        pairs.join(item_count, on="article_id")
        .with_columns((pl.col("pair_count") / pl.col("item_count")).cast(pl.Float32).alias("co_score"))
        .sort(["article_id", "co_score", "article_id_n"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("article_id").alias("_r"))
        .filter(pl.col("_r") <= cfg.cooccur_neighbors)
        .select("article_id", pl.col("article_id_n").alias("neighbor"), "co_score")
    )


def from_seeds(seed: pl.DataFrame, table: pl.DataFrame, score: str) -> pl.DataFrame:
    """Expand seeds through a neighbour table; aggregate over seeds (max and count)."""
    return (
        seed.join(table, on="article_id", how="inner")
        .group_by("customer", pl.col("neighbor").alias("article_id"))
        .agg(pl.col(score).max().alias(f"{score}_max"), pl.len().cast(pl.Int16).alias(f"{score}_n"))
    )


def similar_table(
    emb: np.ndarray,
    emb_ids: np.ndarray,
    seed_ids: np.ndarray,
    catalog_ids: np.ndarray,
    k: int,
    chunk: int = 4096,
) -> pl.DataFrame:
    """Top-k cosine neighbours (within the active catalog) of each seed article.

    Embeddings are L2-normalised, so the dot product is the cosine similarity.
    """
    pos = {int(a): i for i, a in enumerate(emb_ids)}
    # sorted so the matrix products (and their float rounding) are identical from run to run
    s_ids = np.sort(np.array([a for a in seed_ids if int(a) in pos], dtype=np.int64))
    c_ids = np.sort(np.array([a for a in catalog_ids if int(a) in pos], dtype=np.int64))
    if len(s_ids) == 0 or len(c_ids) == 0 or k == 0:
        return pl.DataFrame(schema={"article_id": pl.Int32, "neighbor": pl.Int32, "sim_score": pl.Float32})
    cat = emb[[pos[int(a)] for a in c_ids]].astype(np.float32)
    rows_a, rows_n, rows_s = [], [], []
    kk = min(k + 1, len(c_ids))
    for i in range(0, len(s_ids), chunk):
        ids = s_ids[i : i + chunk]
        sims = emb[[pos[int(a)] for a in ids]].astype(np.float32) @ cat.T
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r, a in enumerate(ids):
            for j in top[r]:
                if c_ids[j] != a:
                    rows_a.append(a)
                    rows_n.append(c_ids[j])
                    rows_s.append(sims[r, j])
    return (
        pl.DataFrame(
            {"article_id": rows_a, "neighbor": rows_n, "sim_score": rows_s},
            schema={"article_id": pl.Int32, "neighbor": pl.Int32, "sim_score": pl.Float32},
        )
        .sort(["article_id", "sim_score", "neighbor"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("article_id").alias("_r"))
        .filter(pl.col("_r") <= k)
        .drop("_r")
    )


def generate(
    hist: pl.DataFrame,
    customers: pl.DataFrame,
    targets: pl.DataFrame,
    week: int,
    cfg: CandidateConfig,
    emb: tuple[np.ndarray, np.ndarray] | None = None,
    co_table: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """All candidates for ``targets`` (a frame with a ``customer`` column) for target week ``week``."""
    targets = targets.select("customer").unique()
    catalog = active_catalog(hist, week, cfg.active_catalog_weeks)
    seed = seeds(hist, targets, cfg.seed_items)
    co_table = co_table if co_table is not None else cooccurrence_table(hist, week, cfg)

    parts = {
        "repurchase": repurchase(hist, targets, week, cfg),
        "popular": popular(hist, customers, targets, week, cfg),
        "cooccur": from_seeds(seed, co_table, "co_score"),
    }
    if emb is not None and cfg.similar_neighbors > 0:
        sim = similar_table(
            emb[0],
            emb[1],
            seed["article_id"].unique().to_numpy(),
            catalog["article_id"].to_numpy(),
            cfg.similar_neighbors,
        )
        parts["similar"] = from_seeds(seed, sim, "sim_score")

    keys = pl.concat([p.select("customer", "article_id") for p in parts.values()]).unique()
    out = keys
    for name, p in parts.items():
        out = out.join(
            p.with_columns(pl.lit(True).alias(f"src_{name}")), on=["customer", "article_id"], how="left"
        )
    flags = [f"src_{s}" for s in SOURCES]
    out = out.with_columns(
        [pl.col(f).fill_null(False) if f in out.columns else pl.lit(False).alias(f) for f in flags]
    )
    # only recommend articles that are still selling
    return out.join(catalog, on="article_id", how="semi")
