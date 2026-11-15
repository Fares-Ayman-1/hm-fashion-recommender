"""Stage 2 features for (customer, article) candidates of target week W.

All inputs come from ``hist`` (weeks < W), so features never see the target week.
"""

from __future__ import annotations

import polars as pl

ARTICLE_CODES = [
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
]


def user_features(hist: pl.DataFrame, customers: pl.DataFrame, week: int) -> pl.DataFrame:
    agg = hist.group_by("customer").agg(
        pl.len().cast(pl.Int32).alias("u_n_tx"),
        (pl.col("week") >= week - 4).sum().cast(pl.Int32).alias("u_n_tx_4w"),
        pl.col("price").mean().cast(pl.Float32).alias("u_avg_price"),
        (pl.col("channel") == 2).mean().cast(pl.Float32).alias("u_online_share"),
        (week - pl.col("week").max()).cast(pl.Int16).alias("u_weeks_since_last"),
        pl.col("article_id").n_unique().cast(pl.Int32).alias("u_n_articles"),
    )
    base = customers.select(
        "customer",
        pl.col("age").cast(pl.Float32).alias("u_age"),
        pl.col("fn").alias("u_fn"),
        pl.col("active").alias("u_active"),
        pl.col("club_member_status").to_physical().cast(pl.Int16).alias("u_club"),
        pl.col("fashion_news_frequency").to_physical().cast(pl.Int16).alias("u_news"),
    )
    return base.join(agg, on="customer", how="left")


def item_features(
    hist: pl.DataFrame, articles: pl.DataFrame, customers: pl.DataFrame, week: int
) -> pl.DataFrame:
    w1 = hist.filter(pl.col("week") == week - 1)
    w4 = hist.filter(pl.col("week") >= week - 4)
    ages = customers.select("customer", "age")
    agg1 = w1.group_by("article_id").agg(
        pl.len().cast(pl.Int32).alias("i_cnt_1w"), pl.col("price").mean().cast(pl.Float32).alias("i_price_1w")
    )
    agg4 = (
        w4.join(ages, on="customer", how="left")
        .group_by("article_id")
        .agg(
            pl.len().cast(pl.Int32).alias("i_cnt_4w"),
            pl.col("age").mean().cast(pl.Float32).alias("i_buyer_age"),
            (pl.col("channel") == 2).mean().cast(pl.Float32).alias("i_online_share"),
        )
    )
    first = hist.group_by("article_id").agg(
        (week - pl.col("week").min()).cast(pl.Int16).alias("i_weeks_on_sale")
    )
    codes = articles.select(
        "article_id", *[pl.col(c).cast(pl.Int32) for c in ARTICLE_CODES if c in articles.columns]
    )
    return (
        codes.join(agg1, on="article_id", how="left")
        .join(agg4, on="article_id", how="left")
        .join(first, on="article_id", how="left")
        .with_columns(
            pl.col("i_cnt_1w").fill_null(0),
            pl.col("i_cnt_4w").fill_null(0),
        )
        .with_columns(
            (pl.col("i_cnt_1w") / (pl.col("i_cnt_4w") / 4 + 1)).cast(pl.Float32).alias("i_trend"),
        )
    )


def type_affinity(hist: pl.DataFrame, articles: pl.DataFrame) -> pl.DataFrame:
    """How often each customer has bought each product type (joins on candidate product type)."""
    return (
        hist.join(articles.select("article_id", "product_type_no"), on="article_id", how="left")
        .group_by("customer", "product_type_no")
        .agg(pl.len().cast(pl.Int32).alias("u_type_count"))
    )


def build(
    cands: pl.DataFrame,
    hist: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    week: int,
) -> pl.DataFrame:
    tgt = cands.select("customer").unique()
    users = user_features(
        hist.join(tgt, on="customer", how="semi"), customers.join(tgt, on="customer", how="semi"), week
    )
    items = item_features(hist, articles, customers, week)
    aff = type_affinity(hist.join(tgt, on="customer", how="semi"), articles)
    df = (
        cands.join(users, on="customer", how="left")
        .join(items, on="article_id", how="left")
        .join(aff, on=["customer", "product_type_no"], how="left")
        .with_columns(
            pl.col("u_type_count").fill_null(0),
            (pl.col("u_age") - pl.col("i_buyer_age")).abs().alias("age_gap"),
            (pl.col("i_price_1w") / pl.col("u_avg_price")).alias("price_ratio"),
        )
    )
    return df.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def add_labels(df: pl.DataFrame, target_tx: pl.DataFrame) -> pl.DataFrame:
    bought = (
        target_tx.select("customer", "article_id").unique().with_columns(pl.lit(1, pl.Int8).alias("label"))
    )
    return df.join(bought, on=["customer", "article_id"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )


def feature_names(df: pl.DataFrame) -> list[str]:
    skip = {"customer", "article_id", "label"}
    return [c for c in df.columns if c not in skip]


CATEGORICAL = [c for c in ARTICLE_CODES if c != "perceived_colour_master_id"] + ["u_club", "u_news"]
