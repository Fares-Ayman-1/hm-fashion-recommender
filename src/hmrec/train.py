"""Validation, final training and batch scoring.

1. Validation: train the ranker on label weeks ``train_weeks`` and score week ``valid_week`` for every
   customer who bought something that week. Compare MAP@12 with two baselines (global popularity, and
   repurchase + age-band popularity), log everything to MLflow.
2. Final: retrain on the same number of weeks shifted by one (ending at ``valid_week``) and score the
   hidden test week for all 1.37M customers in batches -> serving bundle + Kaggle submission.

    uv run python -m hmrec.train                    # validation + final
    uv run python -m hmrec.train --no-final         # validation only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from hmrec.candidates.generate import SOURCES, cooccurrence_table, generate, popular_by_band
from hmrec.config import PROJECT_ROOT, Config, load_config, setup_mlflow
from hmrec.embed import load as load_embeddings
from hmrec.features.build import CATEGORICAL, add_labels, build, feature_names
from hmrec.metrics import candidate_recall, map_at_k

Emb = tuple[np.ndarray, np.ndarray] | None


# ------------------------------------------------------------------------------------------- data
def load_data(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    first = min(cfg.train_weeks) - cfg.history_weeks
    tx = pl.read_parquet(cfg.paths.interim / "transactions.parquet").filter(pl.col("week") >= first)
    customers = pl.read_parquet(cfg.paths.interim / "customers.parquet")
    articles = pl.read_parquet(cfg.paths.interim / "articles.parquet")
    return tx, customers, articles


def history(tx: pl.DataFrame, week: int, cfg: Config) -> pl.DataFrame:
    return tx.filter((pl.col("week") < week) & (pl.col("week") >= week - cfg.history_weeks))


def week_features(
    tx: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    week: int,
    targets: pl.DataFrame,
    cfg: Config,
    emb: Emb,
    co_table: pl.DataFrame | None = None,
    labels: bool = True,
) -> pl.DataFrame:
    hist = history(tx, week, cfg)
    cands = generate(hist, customers, targets, week, cfg.candidates, emb, co_table)
    df = build(cands, hist, customers, articles, week)
    return add_labels(df, tx.filter(pl.col("week") == week)) if labels else df


def training_set(
    tx: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    weeks: list[int],
    cfg: Config,
    emb: Emb,
    seed: int = 42,
) -> pl.DataFrame:
    parts = []
    for w in weeks:
        t0 = time.perf_counter()
        buyers = tx.filter(pl.col("week") == w).select("customer").unique()
        df = week_features(tx, customers, articles, w, buyers, cfg, emb)
        pos_customers = df.filter(pl.col("label") == 1).select("customer").unique()
        df = df.join(pos_customers, on="customer", how="semi")
        rng = np.random.default_rng(seed + w)
        keep = (df["label"].to_numpy() == 1) | (rng.random(df.height) < cfg.neg_sample_rate)
        df = df.filter(pl.Series(keep)).with_columns(pl.lit(w, pl.Int16).alias("_week"))
        print(
            f"  train week {w}: {df.height:,} rows, {pos_customers.height:,} customers, "
            f"{int(df['label'].sum()):,} positives ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        parts.append(df)
    return pl.concat(parts, how="diagonal_relaxed")


# ------------------------------------------------------------------------------------------- model
def as_matrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    """Features as one float32 matrix (half the memory of pandas' float64; nulls become NaN)."""
    return df.select(pl.col(names).cast(pl.Float32)).to_numpy()


def fit(train: pl.DataFrame, names: list[str], cfg: Config, rounds: int) -> lgb.Booster:
    train = train.sort(["_week", "customer"])
    groups = train.group_by(["_week", "customer"], maintain_order=True).len()["len"].to_numpy()
    cats = [c for c in CATEGORICAL if c in names]
    ds = lgb.Dataset(
        as_matrix(train, names),
        label=train["label"].to_numpy(),
        group=groups,
        feature_name=names,
        categorical_feature=cats,
        free_raw_data=True,
    )
    return lgb.train(cfg.lgbm, ds, num_boost_round=rounds)


def predict(df: pl.DataFrame, booster: lgb.Booster, names: list[str], chunk: int = 2_000_000) -> np.ndarray:
    """Score in row chunks so ~10M candidates never need one huge feature matrix."""
    out = np.empty(df.height, dtype=np.float32)
    for i in range(0, df.height, chunk):
        out[i : i + chunk] = booster.predict(as_matrix(df.slice(i, chunk), names))
    return out


def rank(df: pl.DataFrame, booster: lgb.Booster, names: list[str], k: int) -> pl.DataFrame:
    score = predict(df, booster, names)
    return (
        df.select("customer", "article_id", *[f"src_{s}" for s in SOURCES])
        .with_columns(pl.Series("score", score, dtype=pl.Float32))
        .sort(["customer", "score"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").cast(pl.Int16).alias("rank"))
        .filter(pl.col("rank") <= k)
    )


def fill_popular(pred: pl.DataFrame, targets: pl.DataFrame, popular: pl.DataFrame, k: int) -> pl.DataFrame:
    """Top up customers with fewer than k items from the global popular list (keeps rank order)."""
    pop = popular.select("article_id", pl.col("pop_rank").cast(pl.Int16).alias("fill_rank"))
    fill = (
        targets.select("customer")
        .unique()
        .join(pop, how="cross")
        .join(pred.select("customer", "article_id"), on=["customer", "article_id"], how="anti")
        .with_columns((pl.col("fill_rank") + 1000).alias("rank"), pl.lit(-1e9, pl.Float32).alias("score"))
        .drop("fill_rank")
    )
    both = pl.concat([pred, fill], how="diagonal_relaxed").sort(["customer", "rank"])
    return (
        both.with_columns(pl.int_range(1, pl.len() + 1).over("customer").cast(pl.Int16).alias("rank"))
        .filter(pl.col("rank") <= k)
        .with_columns([pl.col(f"src_{s}").fill_null(False) for s in SOURCES])
    )


# ------------------------------------------------------------------------------------------- baselines
def baseline_global(targets: pl.DataFrame, popular: pl.DataFrame, k: int) -> pl.DataFrame:
    top = popular.head(k).select("article_id", pl.col("pop_rank").alias("rank"))
    return targets.select("customer").unique().join(top, how="cross")


def baseline_repurchase_popular(feats: pl.DataFrame, k: int) -> pl.DataFrame:
    """Own recent purchases first (most recent, most frequent), then age-band popularity."""
    big = 10_000
    return (
        feats.select("customer", "article_id", "rep_rank", "pop_rank")
        .with_columns(
            pl.when(pl.col("rep_rank").is_not_null())
            .then(pl.col("rep_rank").cast(pl.Int32))
            .otherwise(pl.col("pop_rank").cast(pl.Int32).fill_null(big) + 100)
            .alias("prio")
        )
        .sort(["customer", "prio"])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer").alias("rank"))
        .filter(pl.col("rank") <= k)
    )


# ------------------------------------------------------------------------------------------- run
def validate(
    cfg: Config, tx: pl.DataFrame, customers: pl.DataFrame, articles: pl.DataFrame, emb: Emb, rounds: int
) -> dict[str, Any]:
    vw, k = cfg.valid_week, cfg.k
    t0 = time.perf_counter()
    train = training_set(tx, customers, articles, cfg.train_weeks, cfg, emb)
    names = [n for n in feature_names(train) if n != "_week"]
    booster = fit(train, names, cfg, rounds)
    mean = train["label"].mean()
    pos_rate = float(mean) if isinstance(mean, int | float) else 0.0
    n_train = train.height
    del train

    actual = tx.filter(pl.col("week") == vw).select("customer", "article_id")
    targets = actual.select("customer").unique()
    feats = week_features(tx, customers, articles, vw, targets, cfg, emb, labels=False)
    glob_pop = popular_by_band(history(tx, vw, cfg), customers, vw, cfg.k).filter(pl.col("band") == "unknown")

    pred = fill_popular(rank(feats, booster, names, k), targets, glob_pop, k)
    scores = {
        "lgbm_ranker": map_at_k(pred, actual, k),
        "baseline_popular": map_at_k(baseline_global(targets, glob_pop, k), actual, k),
        "baseline_repurchase_popular": map_at_k(baseline_repurchase_popular(feats, k), actual, k),
    }
    recall = {"all": candidate_recall(feats, actual)}
    for s in SOURCES:
        recall[s] = candidate_recall(feats.filter(pl.col(f"src_{s}")), actual)
    imp = dict(zip(names, booster.feature_importance("gain").tolist(), strict=True))
    return {
        "map12": scores,
        "candidate_recall": recall,
        "names": names,
        "importance": imp,
        "n_train_rows": n_train,
        "train_pos_rate": pos_rate,
        "valid_customers": targets.height,
        "candidates_per_customer": feats.height / max(targets.height, 1),
        "seconds": time.perf_counter() - t0,
        "pred": pred,
    }


def final(
    cfg: Config,
    tx: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    emb: Emb,
    rounds: int,
    names: list[str],
) -> pl.DataFrame:
    """Retrain on weeks shifted by one and score the hidden test week for every customer."""
    tw, k = cfg.test_week, cfg.k
    weeks = [w + 1 for w in cfg.train_weeks]
    train = training_set(tx, customers, articles, weeks, cfg, emb)
    booster = fit(train, names, cfg, rounds)
    del train
    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(cfg.paths.models / "ranker.txt"))

    hist = history(tx, tw, cfg)
    co = cooccurrence_table(hist, tw, cfg.candidates)
    glob_pop = popular_by_band(hist, customers, tw, k).filter(pl.col("band") == "unknown")
    all_c = customers.select("customer")
    out = []
    for i in range(0, all_c.height, cfg.inference_batch):
        t0 = time.perf_counter()
        batch = all_c.slice(i, cfg.inference_batch)
        feats = week_features(tx, customers, articles, tw, batch, cfg, emb, co_table=co, labels=False)
        out.append(fill_popular(rank(feats, booster, names, k), batch, glob_pop, k))
        print(
            f"  scored customers {i + batch.height:,}/{all_c.height:,} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    return pl.concat(out, how="diagonal_relaxed")


def write_serving(
    cfg: Config,
    recs: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    emb: Emb,
    tx: pl.DataFrame,
    meta: dict[str, Any],
) -> None:
    out = cfg.paths.reports / "serving"
    out.mkdir(parents=True, exist_ok=True)
    reasons = (
        pl.concat_list([pl.when(pl.col(f"src_{s}")).then(pl.lit(s)).otherwise(pl.lit(None)) for s in SOURCES])
        .list.drop_nulls()
        .list.join(",")
    )
    grouped = (
        recs.with_columns(reasons.alias("reasons"))
        .sort(["customer", "rank"])
        .group_by("customer", maintain_order=True)
        .agg(pl.col("article_id"), pl.col("reasons"))
        .join(customers.select("customer", "customer_id"), on="customer", how="left")
    )
    grouped.select("customer_id", "article_id", "reasons").write_parquet(out / "recommendations.parquet")

    # Kaggle submission: one row per customer, 10-digit zero-padded article ids
    sub = grouped.select(
        "customer_id",
        pl.col("article_id")
        .list.eval(pl.element().cast(pl.String).str.zfill(10))
        .list.join(" ")
        .alias("prediction"),
    )
    sub.write_csv(cfg.paths.reports / "submission.csv")

    keep = [
        "article_id",
        "prod_name",
        "product_type_name",
        "colour_group_name",
        "index_group_name",
        "section_name",
        "detail_desc",
    ]
    arts = articles.select([c for c in keep if c in articles.columns])
    arts = arts.with_columns(
        pl.Series("has_image", [(cfg.paths.thumbs / f"{a:010d}.jpg").exists() for a in arts["article_id"]])
    )
    arts.write_parquet(out / "articles.parquet")

    # recent purchase history for the demo customers shown in the dashboard (most active last month)
    last4 = tx.filter(pl.col("week") >= cfg.test_week - 4)
    demo = last4.group_by("customer").len().sort("len", descending=True).head(200).select("customer")
    hist = (
        tx.join(demo, on="customer", how="semi")
        .filter(pl.col("week") >= cfg.test_week - 8)
        .join(customers.select("customer", "customer_id"), on="customer")
        .select("customer_id", "t_dat", "article_id")
        .sort(["customer_id", "t_dat"], descending=[False, True])
    )
    hist.write_parquet(out / "history.parquet")

    # articles the demo shows (demo customers' recent purchases + their recommendations), most-shown first:
    # the only articles whose photos `scripts/download_data.py --images N` fetches
    demo_ids = demo.join(customers.select("customer", "customer_id"), on="customer")["customer_id"]
    shown = pl.concat(
        [
            hist.select("article_id"),
            grouped.filter(pl.col("customer_id").is_in(demo_ids.implode())).select(
                pl.col("article_id").explode()
            ),
        ]
    )
    shown.group_by("article_id").len().sort(["len", "article_id"], descending=[True, False]).select(
        "article_id"
    ).write_parquet(out / "demo_articles.parquet")

    if emb is not None:
        from hmrec.candidates.generate import similar_table

        catalog = last4.select("article_id").unique()["article_id"].to_numpy()
        similar_table(emb[0], emb[1], catalog, catalog, 12).write_parquet(out / "similar.parquet")
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def run(cfg: Config, rounds: int, do_final: bool, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    t0 = time.perf_counter()
    tx, customers, articles = load_data(cfg)
    emb = load_embeddings(cfg)
    print(f"data: {tx.height:,} transactions, {customers.height:,} customers, embeddings: {emb is not None}")

    v = validate(cfg, tx, customers, articles, emb, rounds)
    print(json.dumps({"map12": v["map12"], "candidate_recall": v["candidate_recall"]}, indent=2))

    setup_mlflow(cfg, root)
    tags = {"embeddings": str(emb is not None)}
    run_ids = {}
    with mlflow.start_run(run_name="baselines", tags=tags) as r:
        for name in ("baseline_popular", "baseline_repurchase_popular"):
            mlflow.log_metric(f"{name}_map12", v["map12"][name])
        run_ids["baselines"] = r.info.run_id
    with mlflow.start_run(run_name="lgbm-lambdarank", tags=tags) as r:
        mlflow.log_params({f"lgbm.{k}": val for k, val in cfg.lgbm.items()})
        mlflow.log_params(
            {
                "num_boost_round": rounds,
                "train_weeks": cfg.train_weeks,
                "valid_week": cfg.valid_week,
                "neg_sample_rate": cfg.neg_sample_rate,
                "n_features": len(v["names"]),
                **{f"cand.{k}": val for k, val in cfg.candidates.model_dump().items()},
            }
        )
        mlflow.log_metric("lgbm_ranker_map12", v["map12"]["lgbm_ranker"])
        for s, val in v["candidate_recall"].items():
            mlflow.log_metric(f"candidate_recall_{s}", val)
        mlflow.log_metrics(
            {
                "n_train_rows": v["n_train_rows"],
                "train_pos_rate": v["train_pos_rate"],
                "valid_customers": v["valid_customers"],
                "candidates_per_customer": v["candidates_per_customer"],
            }
        )
        cfg.paths.reports.mkdir(parents=True, exist_ok=True)
        imp_path = cfg.paths.reports / "feature_importance.json"
        imp_path.write_text(json.dumps(dict(sorted(v["importance"].items(), key=lambda x: -x[1])), indent=2))
        mlflow.log_artifact(str(imp_path))
        run_ids["lgbm"] = r.info.run_id

    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "k": cfg.k,
        "valid_week": cfg.valid_week,
        "test_week": cfg.test_week,
        "map12": v["map12"],
        "candidate_recall": v["candidate_recall"],
        "mlflow_run_ids": run_ids,
        "embeddings": emb is not None,
    }
    if do_final:
        recs = final(cfg, tx, customers, articles, emb, rounds, v["names"])
        write_serving(cfg, recs, customers, articles, emb, tx, meta)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min; runs {run_ids}")
    return {"validation": v, "run_ids": run_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run(cfg, a.rounds or cfg.num_boost_round, do_final=not a.no_final)


if __name__ == "__main__":
    main()
