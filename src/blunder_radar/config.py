"""
Configuration schema and loader for BlunderRadar.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source_url: str = (
        "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
    )
    local_zst_path: str | None = "data/raw/lichess_sample.pgn.zst"
    output_parquet_dir: str = "data/processed/positions.parquet"
    checkpoint_file: str = "data/processed/pipeline_state.json"
    quality_report_md: str = "data/data_quality_report.md"
    quality_report_json: str = "data/data_quality_report.json"
    target_positions: int = 300000
    batch_size: int = 5000
    min_rating: int = 800
    max_rating: int = 2400
    allowed_speeds: list[str] = field(default_factory=lambda: ["blitz", "rapid"])
    blunder_threshold_cp: float = 200.0
    mate_score_cp: float = 10000.0
    dead_lost_threshold_cp: float | None = -800.0
    ignore_dead_lost: bool = True
    win_prob_scaling: float = 400.0
    win_prob_blunder_threshold: float = 0.15
    clearly_winning_threshold: float = 0.80
    dead_lost_threshold_prob: float = 0.02
    partition_cols: list[str] = field(default_factory=lambda: ["rating_band"])


@dataclass
class FeaturesConfig:
    include_eval_before: bool = True
    time_pressure_threshold_seconds: float = 30.0


@dataclass
class ModelConfig:
    random_seed: int = 42
    data_path: str = "data/processed/positions.parquet"
    features_path: str = "data/processed/features.parquet"
    models_dir: str = "models"
    docs_dir: str = "docs"
    tune_sample_size: int = 60000
    target_col: str = "is_blunder"
    test_size: float = 0.15
    val_size: float = 0.15
    cv_folds: int = 5
    rating_bands: dict[str, list[int]] = field(
        default_factory=lambda: {
            "low": [800, 1200],
            "mid": [1200, 1800],
            "high": [1800, 2400],
        }
    )
    models: list[str] = field(
        default_factory=lambda: ["majority_class", "logistic_regression", "xgboost", "lightgbm"]
    )
    optuna_trials: int = 20
    optuna_timeout: int = 600
    calibration_method: str = "isotonic"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "blunder_radar_experiments"


@dataclass
class AppConfig:
    project_name: str = "BlunderRadar"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(file_path: str | Path = "configs/data_config.yaml") -> DataConfig:
    data = load_yaml(file_path)
    return DataConfig(**{k: v for k, v in data.items() if hasattr(DataConfig, k)})


def _flatten_model_yaml(data: dict[str, Any]) -> dict[str, Any]:
    """Map the nested sections of model_config.yaml onto flat ModelConfig fields."""
    flat = dict(data)
    optuna_cfg = flat.pop("optuna", None) or {}
    if "n_trials" in optuna_cfg:
        flat["optuna_trials"] = optuna_cfg["n_trials"]
    if "timeout_seconds" in optuna_cfg:
        flat["optuna_timeout"] = optuna_cfg["timeout_seconds"]
    calib = flat.pop("calibration", None) or {}
    if "method" in calib:
        flat["calibration_method"] = calib["method"]
    mlflow_cfg = flat.pop("mlflow", None) or {}
    if "tracking_uri" in mlflow_cfg:
        flat["mlflow_tracking_uri"] = mlflow_cfg["tracking_uri"]
    if "experiment_name" in mlflow_cfg:
        flat["experiment_name"] = mlflow_cfg["experiment_name"]
    return flat


def load_model_config(file_path: str | Path = "configs/model_config.yaml") -> ModelConfig:
    data = _flatten_model_yaml(load_yaml(file_path))
    return ModelConfig(**{k: v for k, v in data.items() if hasattr(ModelConfig, k)})


def load_app_config(file_path: str | Path = "configs/config.yaml") -> AppConfig:
    data = load_yaml(file_path)
    data_cfg = DataConfig(**data.get("data", {}))
    feat_cfg = FeaturesConfig(**data.get("features", {}))
    model_cfg = ModelConfig(**data.get("model", {}))
    return AppConfig(
        project_name=data.get("project_name", "BlunderRadar"),
        random_seed=data.get("random_seed", 42),
        data=data_cfg,
        features=feat_cfg,
        model=model_cfg,
    )
