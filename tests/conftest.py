from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import write_config, write_raw

from hmrec.config import Config, load_config


@pytest.fixture(scope="session")
def project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Config]:
    root = tmp_path_factory.mktemp("hm")
    write_raw(root / "data" / "raw")
    cfg = load_config(write_config(root), root=root)
    from hmrec.data.prepare import prepare_tables

    prepare_tables(cfg)
    return root, cfg


@pytest.fixture(scope="session")
def with_embeddings(project: tuple[Path, Config]) -> tuple[Path, Config]:
    """Random unit embeddings standing in for CLIP (the real model is not run in tests)."""
    import polars as pl

    root, cfg = project
    ids = pl.read_parquet(cfg.paths.interim / "articles.parquet")["article_id"].to_numpy()
    rng = np.random.default_rng(0)
    e = rng.normal(size=(len(ids), 16)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.save(cfg.paths.interim / "article_emb.npy", e.astype(np.float16))
    np.save(cfg.paths.interim / "article_emb_ids.npy", ids.astype(np.int32))
    return root, cfg


@pytest.fixture(scope="session")
def trained(with_embeddings: tuple[Path, Config]) -> tuple[Path, Config, dict[str, object]]:
    from hmrec.train import run

    root, cfg = with_embeddings
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    result = run(cfg, rounds=cfg.num_boost_round, do_final=True, root=root)
    return root, cfg, result
