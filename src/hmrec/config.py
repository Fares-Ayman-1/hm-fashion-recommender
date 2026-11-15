"""Typed configuration loaded from YAML, plus the shared MLflow setup."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    raw: Path
    interim: Path
    thumbs: Path
    models: Path
    reports: Path

    def resolve(self, root: Path) -> Paths:
        def r(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        return Paths(**{k: r(v) for k, v in self.model_dump().items()})


class CandidateConfig(BaseModel):
    repurchase_weeks: int = Field(gt=0)
    repurchase_max: int = Field(gt=0)
    popular_n: int = Field(gt=0)
    cooccur_weeks: int = Field(gt=0)
    cooccur_neighbors: int = Field(gt=0)
    seed_items: int = Field(gt=0)
    similar_neighbors: int = Field(ge=0)
    active_catalog_weeks: int = Field(gt=0)


class Config(BaseModel):
    competition: str
    paths: Paths
    tracking_uri: str
    artifact_root: str
    experiment: str
    test_week_start: date
    valid_week: int
    train_weeks: list[int]
    history_weeks: int = Field(gt=0)
    k: int = Field(gt=0)
    candidates: CandidateConfig
    neg_sample_rate: float = Field(gt=0, le=1)
    inference_batch: int = Field(gt=0)
    lgbm: dict[str, Any]
    num_boost_round: int = Field(gt=0)

    @field_validator("train_weeks")
    @classmethod
    def _train_before_valid(cls, v: list[int], info: Any) -> list[int]:
        valid = info.data.get("valid_week")
        if valid is not None and any(w >= valid for w in v):
            raise ValueError(f"train weeks {v} must all be before the validation week {valid}")
        return v

    @property
    def test_week(self) -> int:
        return self.valid_week + 1


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    p = Path(path) if path else root / "configs" / "default.yaml"
    cfg = Config.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    cfg.paths = cfg.paths.resolve(root)
    return cfg


def _abs(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def tracking_uri(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    db = _abs(cfg.tracking_uri, root)
    db.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db.as_posix()


def setup_mlflow(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(cfg, root))
    exp = mlflow.get_experiment_by_name(cfg.experiment)
    if exp is not None:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        return str(exp.experiment_id)
    artifacts = _abs(cfg.artifact_root, root)
    artifacts.mkdir(parents=True, exist_ok=True)
    exp_id = mlflow.create_experiment(cfg.experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return str(exp_id)
