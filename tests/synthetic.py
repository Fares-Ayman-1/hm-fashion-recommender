"""Small synthetic dataset in the exact H&M file formats (CSV columns, id formats, dates).

Customers have favourite product types and re-buy some items, so every candidate source has signal.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_CUSTOMERS, N_ARTICLES = 400, 120
TEST_WEEK_START = date(2020, 9, 23)
FIRST_WEEK = 90  # synthetic history starts 15 weeks before the test week


def customer_hex(i: int) -> str:
    return hashlib.sha256(f"customer-{i}".encode()).hexdigest()


def write_raw(raw: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True, exist_ok=True)
    types = rng.integers(0, 6, N_ARTICLES)
    art_ids = 100_000_001 + np.arange(N_ARTICLES) * 7
    pd.DataFrame(
        {
            "article_id": art_ids,
            "product_code": art_ids // 1000,
            "prod_name": [f"Item {i}" for i in range(N_ARTICLES)],
            "product_type_no": 250 + types,
            "product_type_name": [f"Type {t}" for t in types],
            "product_group_name": "Garment Upper body",
            "graphical_appearance_no": 1010016,
            "graphical_appearance_name": "Solid",
            "colour_group_code": rng.integers(1, 20, N_ARTICLES),
            "colour_group_name": "Black",
            "perceived_colour_value_id": 4,
            "perceived_colour_value_name": "Dark",
            "perceived_colour_master_id": rng.integers(1, 10, N_ARTICLES),
            "perceived_colour_master_name": "Black",
            "department_no": 1600 + types,
            "department_name": "Dept",
            "index_code": "A",
            "index_name": "Ladieswear",
            "index_group_no": 1,
            "index_group_name": "Ladieswear",
            "section_no": 10 + types,
            "section_name": "Section",
            "garment_group_no": 1000 + types,
            "garment_group_name": "Jersey",
            "detail_desc": [f"A soft jersey item number {i}" for i in range(N_ARTICLES)],
        }
    ).to_csv(raw / "articles.csv", index=False)

    ages = rng.integers(16, 75, N_CUSTOMERS).astype(float)
    ages[rng.random(N_CUSTOMERS) < 0.05] = np.nan
    ids = [customer_hex(i) for i in range(N_CUSTOMERS)]
    pd.DataFrame(
        {
            "customer_id": ids,
            "FN": np.where(rng.random(N_CUSTOMERS) < 0.4, 1.0, np.nan),
            "Active": np.where(rng.random(N_CUSTOMERS) < 0.4, 1.0, np.nan),
            "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE"], N_CUSTOMERS),
            "fashion_news_frequency": rng.choice(["NONE", "Regularly", "None"], N_CUSTOMERS),
            "age": ages,
            "postal_code": "x" * 64,
        }
    ).to_csv(raw / "customers.csv", index=False)

    rows = []
    start = TEST_WEEK_START - timedelta(weeks=105 - FIRST_WEEK)
    fav_type = rng.integers(0, 6, N_CUSTOMERS)
    staple = rng.integers(0, N_ARTICLES, N_CUSTOMERS)
    for c in range(N_CUSTOMERS):
        pool = np.flatnonzero(types == fav_type[c])
        for d in range((105 - FIRST_WEEK) * 7):
            if rng.random() < 0.06:
                a = (
                    staple[c]
                    if rng.random() < 0.35
                    else (rng.choice(pool) if rng.random() < 0.7 else rng.integers(0, N_ARTICLES))
                )
                rows.append(
                    (start + timedelta(days=d), ids[c], art_ids[a], 0.01 + 0.002 * types[a], 1 + (c % 2))
                )
    tx = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
    tx = tx[tx["t_dat"] < TEST_WEEK_START]
    tx["t_dat"] = tx["t_dat"].astype(str)
    tx.to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame({"customer_id": ids, "prediction": ""}).to_csv(raw / "sample_submission.csv", index=False)


def write_config(root: Path) -> Path:
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text("utf-8")
    )
    cfg.update(
        {
            "tracking_uri": str((root / "mlflow.db").resolve()),
            "artifact_root": str((root / "mlruns").resolve()),
            "experiment": "test",
            "history_weeks": 8,
            "inference_batch": 150,
            "num_boost_round": 15,
            "neg_sample_rate": 0.5,
        }
    )
    cfg["lgbm"].update({"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 2})
    cfg["candidates"].update({"popular_n": 12, "similar_neighbors": 4})
    path = root / "configs" / "test.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path
