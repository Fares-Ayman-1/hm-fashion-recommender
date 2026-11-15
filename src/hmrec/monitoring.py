"""Weekly monitoring for the recommender, with a retrain trigger.

Recommendations go stale when the catalogue or tastes move. Three checks compare the latest week with
the week before:
  catalogue churn     share of last week's purchases on articles first sold in the last 2 weeks
  popularity shift    overlap (Jaccard) of the top-100 articles of the two weeks
  price drift         PSI of transaction prices
Any alarm sets ``retrain_recommended`` (the model is retrained weekly anyway; an alarm means sooner).

    uv run python -m hmrec.monitoring
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from hmrec.config import load_config

THRESHOLDS = {"new_article_share": 0.30, "top100_overlap": 0.40, "price_psi": 0.20}


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / reference.size, 1e-6, None)
    cur = np.clip(np.histogram(current, edges)[0] / current.size, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def check(tx: pl.DataFrame, week: int) -> dict[str, Any]:
    cur, ref = tx.filter(pl.col("week") == week), tx.filter(pl.col("week") == week - 1)
    first_week = tx.group_by("article_id").agg(pl.col("week").min().alias("first"))
    new_share = float(
        cur.join(first_week, on="article_id").select((pl.col("first") >= week - 1).mean()).item()
    )

    def top(df: pl.DataFrame) -> set[int]:
        return set(
            df.group_by("article_id").len().sort("len", descending=True).head(100)["article_id"].to_list()
        )

    a, b = top(cur), top(ref)
    overlap = len(a & b) / max(len(a | b), 1)
    values = {
        "new_article_share": new_share,
        "top100_overlap": overlap,
        "price_psi": psi(ref["price"].to_numpy(), cur["price"].to_numpy()),
    }
    alerts = {
        "new_article_share": values["new_article_share"] > THRESHOLDS["new_article_share"],
        "top100_overlap": values["top100_overlap"] < THRESHOLDS["top100_overlap"],
        "price_psi": values["price_psi"] > THRESHOLDS["price_psi"],
    }
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "week": week,
        "values": values,
        "alerts": alerts,
        "retrain_recommended": any(alerts.values()),
    }


def to_markdown(r: dict[str, Any]) -> str:
    ops = {"top100_overlap": "<"}
    L = [
        "# Monitoring report",
        "",
        f"Checked {r['checked_at']}: week {r['week']} vs week {r['week'] - 1}.",
        "",
        "| Check | Value | Alert if | Alert |",
        "|---|---:|---:|---|",
    ]
    for k, v in r["values"].items():
        L.append(
            f"| {k.replace('_', ' ')} | {v:.4f} | {ops.get(k, '>')} {THRESHOLDS[k]} | "
            f"{'yes' if r['alerts'][k] else 'no'} |"
        )
    L += ["", f"**Retrain recommended: {'yes' if r['retrain_recommended'] else 'no'}**"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    tx = pl.read_parquet(cfg.paths.interim / "transactions.parquet").filter(
        pl.col("week") >= cfg.valid_week - 30
    )
    r = check(tx, cfg.valid_week)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "monitoring.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (cfg.paths.reports / "monitoring.md").write_text(to_markdown(r), encoding="utf-8")
    print(to_markdown(r))


if __name__ == "__main__":
    main()
