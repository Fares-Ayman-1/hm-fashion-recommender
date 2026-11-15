from __future__ import annotations

from pathlib import Path

import polars as pl

from hmrec.candidates.generate import SOURCES, add_band, cooccurrence_table, generate, popular_by_band
from hmrec.config import Config
from hmrec.embed import load as load_emb
from hmrec.features.build import add_labels, build, feature_names


def _data(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tx = pl.read_parquet(cfg.paths.interim / "transactions.parquet")
    return (
        tx,
        pl.read_parquet(cfg.paths.interim / "customers.parquet"),
        pl.read_parquet(cfg.paths.interim / "articles.parquet"),
    )


def test_weeks_are_aligned(project: tuple[Path, Config]) -> None:
    _, cfg = project
    tx, _, _ = _data(cfg)
    last = tx.filter(pl.col("week") == cfg.valid_week)["t_dat"]
    assert str(last.min()) == "2020-09-16" and str(last.max()) == "2020-09-22"


def test_no_leakage_from_the_target_week(with_embeddings: tuple[Path, Config]) -> None:
    """Scrambling the purchases of week W (and later) must not change candidates or features for W."""
    from hmrec.train import week_features

    _, cfg = with_embeddings
    tx, customers, articles = _data(cfg)
    w = cfg.train_weeks[-1]
    targets = tx.filter(pl.col("week") == w).select("customer").unique()
    emb = load_emb(cfg)
    scrambled = pl.concat(
        [
            tx.filter(pl.col("week") < w),
            tx.filter(pl.col("week") >= w).with_columns(
                pl.col("article_id").shuffle(seed=1), pl.col("price") * 3
            ),
        ]
    )
    keys = ["customer", "article_id"]
    a = week_features(tx, customers, articles, w, targets, cfg, emb, labels=False).sort(keys)
    b = week_features(scrambled, customers, articles, w, targets, cfg, emb, labels=False).sort(keys)
    assert a.equals(b)
    assert set(a["customer"].unique()) <= set(targets["customer"])


def test_candidates_are_deterministic(with_embeddings: tuple[Path, Config]) -> None:
    _, cfg = with_embeddings
    tx, customers, _ = _data(cfg)
    w = cfg.valid_week
    hist = tx.filter(pl.col("week") < w)
    targets = tx.filter(pl.col("week") == w).select("customer").unique()
    emb = load_emb(cfg)
    a = generate(hist, customers, targets, w, cfg.candidates, emb).sort(["customer", "article_id"])
    b = generate(hist.sample(fraction=1.0, shuffle=True, seed=7), customers, targets, w, cfg.candidates, emb)
    assert a.equals(b.sort(["customer", "article_id"]))
    assert all(f"src_{s}" in a.columns for s in SOURCES)
    assert a.filter(~pl.any_horizontal([pl.col(f"src_{s}") for s in SOURCES])).height == 0


def test_popular_is_per_age_band(project: tuple[Path, Config]) -> None:
    _, cfg = project
    tx, customers, _ = _data(cfg)
    table = popular_by_band(tx.filter(pl.col("week") < cfg.valid_week), customers, cfg.valid_week, 5)
    assert "unknown" in table["band"].to_list()
    assert table.group_by("band").agg(pl.len())["len"].max() <= 5
    assert add_band(pl.DataFrame({"age": [None, 19, 33, 70]}, schema={"age": pl.Int8}))["band"].to_list() == [
        "unknown",
        "<20",
        "30-39",
        "60+",
    ]


def test_cooccurrence_hand_example(project: tuple[Path, Config]) -> None:
    _, cfg = project
    hist = pl.DataFrame(
        {
            "customer": [1, 1, 2, 2, 3],
            "week": [100, 100, 100, 100, 100],
            "article_id": [10, 20, 10, 30, 10],
        },
        schema={"customer": pl.UInt64, "week": pl.Int16, "article_id": pl.Int32},
    )
    t = cooccurrence_table(hist, 101, cfg.candidates)
    # article 10 is in 3 baskets; 20 and 30 each co-occur once -> 1/3
    row = t.filter(pl.col("article_id") == 10).sort("neighbor")
    assert row["neighbor"].to_list() == [20, 30]
    assert row["co_score"].to_list() == [pl.Series([1 / 3], dtype=pl.Float32)[0]] * 2


def test_features_and_labels(with_embeddings: tuple[Path, Config]) -> None:
    _, cfg = with_embeddings
    tx, customers, articles = _data(cfg)
    w = cfg.valid_week
    hist = tx.filter((pl.col("week") < w) & (pl.col("week") >= w - cfg.history_weeks))
    targets = tx.filter(pl.col("week") == w).select("customer").unique()
    cands = generate(hist, customers, targets, w, cfg.candidates, load_emb(cfg))
    df = add_labels(build(cands, hist, customers, articles, w), tx.filter(pl.col("week") == w))
    assert df.height == cands.height
    assert df["label"].sum() > 0
    names = feature_names(df)
    assert "label" not in names and "customer" not in names
    assert {"u_age", "i_cnt_1w", "age_gap", "src_repurchase"} <= set(names)
