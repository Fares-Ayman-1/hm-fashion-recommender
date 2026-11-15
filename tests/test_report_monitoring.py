from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.config import Config
from hmrec.monitoring import check, psi, to_markdown
from hmrec.report import build


def test_psi() -> None:
    a = np.random.default_rng(0).normal(size=4000)
    assert psi(a, a.copy()) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, a + 2) > 0.2


def test_monitoring_on_synthetic(project: tuple[Path, Config]) -> None:
    _, cfg = project
    tx = pl.read_parquet(cfg.paths.interim / "transactions.parquet")
    r = check(tx, cfg.valid_week)
    assert set(r["values"]) == {"new_article_share", "top100_overlap", "price_psi"}
    assert 0 <= r["values"]["top100_overlap"] <= 1
    assert "Retrain recommended" in to_markdown(r)


def test_results_from_mlflow(trained: tuple[Path, Config, dict[str, object]]) -> None:
    root, cfg, result = trained
    md = build(cfg, root)
    assert f"`{result['run_ids']['lgbm']}`" in md  # type: ignore[index]
    assert f"{result['validation']['map12']['lgbm_ranker']:.5f}" in md  # type: ignore[index]
