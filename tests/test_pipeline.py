from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from hmrec.config import Config


def test_validation_scores(trained: tuple[Path, Config, dict[str, object]]) -> None:
    _, _, result = trained
    v = result["validation"]
    for name in ("lgbm_ranker", "baseline_popular", "baseline_repurchase_popular"):
        assert 0.0 <= v["map12"][name] <= 1.0 and math.isfinite(v["map12"][name])  # type: ignore[index]
    assert 0.0 < v["candidate_recall"]["all"] <= 1.0  # type: ignore[index]


def test_serving_bundle_and_submission(trained: tuple[Path, Config, dict[str, object]]) -> None:
    _, cfg, _ = trained
    serving = cfg.paths.reports / "serving"
    recs = pl.read_parquet(serving / "recommendations.parquet")
    customers = pl.read_parquet(cfg.paths.interim / "customers.parquet")
    assert recs.height == customers.height  # every customer gets recommendations
    assert recs["article_id"].list.len().min() == cfg.k
    assert recs["article_id"].list.n_unique().min() == cfg.k  # no duplicates inside a list

    sub = pl.read_csv(cfg.paths.reports / "submission.csv")
    assert sub.columns == ["customer_id", "prediction"]
    first = sub["prediction"][0].split(" ")
    assert len(first) == cfg.k and all(len(a) == 10 for a in first)

    meta = json.loads((serving / "metrics.json").read_text(encoding="utf-8"))
    assert set(meta["mlflow_run_ids"]) == {"baselines", "lgbm"}
    assert (serving / "similar.parquet").exists()
    assert (cfg.paths.models / "ranker.txt").exists()
