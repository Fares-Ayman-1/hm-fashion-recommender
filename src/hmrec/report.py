"""Write RESULTS.md from the latest MLflow runs, so every published number traces back to a run.

uv run python -m hmrec.report
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from hmrec.config import PROJECT_ROOT, Config, load_config, setup_mlflow

LABELS = {
    "lgbm_ranker": "Two-stage: 4 candidate sources + LightGBM LambdaRank",
    "baseline_repurchase_popular": "Repurchase, then age-band popularity (heuristic)",
    "baseline_popular": "Global top-12 of last week",
}
SOURCE_LABELS = {
    "all": "All sources combined",
    "repurchase": "Repurchase (own recent purchases)",
    "popular": "Age-band popularity",
    "cooccur": "Bought together (same week)",
    "similar": "CLIP similarity",
}


def latest(exp_id: str, name: str) -> mlflow.entities.Run:
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"no finished '{name}' run in MLflow; run `python -m hmrec.train` first")
    return runs[0]


def deployed_runs(cfg: Config) -> dict[str, str] | None:
    """MLflow run ids recorded in the serving bundle: the runs behind the model the API actually serves."""
    f = cfg.paths.reports / "serving" / "metrics.json"
    if not f.exists():
        return None
    ids: dict[str, str] = json.loads(f.read_text(encoding="utf-8")).get("mlflow_run_ids", {})
    return ids if {"lgbm", "baselines"} <= set(ids) else None


def build(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    exp = setup_mlflow(cfg, root)
    ids = deployed_runs(cfg)
    if ids:  # prefer exact provenance over "most recent run" (robust to clock changes and re-runs)
        ranker, base = mlflow.get_run(ids["lgbm"]), mlflow.get_run(ids["baselines"])
    else:
        ranker, base = latest(exp, "lgbm-lambdarank"), latest(exp, "baselines")
    m = {**base.data.metrics, **ranker.data.metrics}
    scores = {
        "lgbm_ranker": (m["lgbm_ranker_map12"], ranker.info.run_id),
        "baseline_repurchase_popular": (m["baseline_repurchase_popular_map12"], base.info.run_id),
        "baseline_popular": (m["baseline_popular_map12"], base.info.run_id),
    }
    best_base = max(scores["baseline_repurchase_popular"][0], scores["baseline_popular"][0])
    p = ranker.data.params
    L = [
        "# Results",
        "",
        f"Generated from MLflow on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m hmrec.report`.",
        "Metric: **MAP@12** (the competition metric; higher is better) on validation week "
        f"{p.get('valid_week')}",
        f"(2020-09-16 to 2020-09-22), over the {int(m['valid_customers']):,} customers "
        "who bought something that week.",
        f"The ranker was trained on label weeks {p.get('train_weeks')}; "
        "nothing from the validation week is used.",
        "",
        "| Model | MAP@12 | MLflow run |",
        "|---|---:|---|",
    ]
    for key, (val, rid) in scores.items():
        L.append(
            f"| {LABELS[key]} | {'**' if key == 'lgbm_ranker' else ''}{val:.5f}"
            f"{'**' if key == 'lgbm_ranker' else ''} | `{rid}` |"
        )
    L += [
        "",
        "The ranker changes MAP@12 by "
        f"**{scores['lgbm_ranker'][0] / best_base - 1:+.1%}** relative to the best baseline.",
        "",
        "## Candidate recall (share of validation-week purchases that stage 1 surfaces)",
        "",
        "| Source | Recall |",
        "|---|---:|",
    ]
    for s, label in SOURCE_LABELS.items():
        if f"candidate_recall_{s}" in m:
            L.append(f"| {label} | {m[f'candidate_recall_{s}']:.4f} |")
    L += [
        "",
        f"Average candidates per customer: {m['candidates_per_customer']:.1f}. "
        "Recall caps what the ranker can achieve: it can only reorder what stage 1 proposes.",
        "",
        "## Training setup (from the run's parameters)",
        "",
        f"- {int(m['n_train_rows']):,} training rows (negatives sampled at {p.get('neg_sample_rate')}), "
        f"{p.get('n_features')} features, {p.get('num_boost_round')} boosting rounds, "
        f"learning rate {p.get('lgbm.learning_rate')}, {p.get('lgbm.num_leaves')} leaves",
        f"- CLIP similarity source used: {ranker.data.tags.get('embeddings')}",
    ]
    kaggle = cfg.paths.reports / "kaggle_score.json"
    if kaggle.exists():
        k = json.loads(kaggle.read_text(encoding="utf-8"))
        L += [
            "",
            "## Kaggle late submission",
            "",
            f"Private leaderboard MAP@12 **{k['private']}** (public {k['public']}), submitted {k['date']}.",
        ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    (PROJECT_ROOT / "RESULTS.md").write_text(build(cfg), encoding="utf-8")
    print(f"wrote {PROJECT_ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
