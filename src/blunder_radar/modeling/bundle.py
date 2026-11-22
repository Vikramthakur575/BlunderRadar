"""A self-describing, serialisable model: estimator + calibrator + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def default_thresholds(base_rate: float) -> dict[str, float]:
    """Risk bands relative to the base blunder rate (0.5x, 2x, 4x)."""
    return {
        "low": 0.5 * base_rate,
        "moderate": 2.0 * base_rate,
        "high": 4.0 * base_rate,
    }


@dataclass
class ModelBundle:
    model: Any
    feature_names: list[str]
    algorithm: str
    with_eval: bool
    calibrator: Any = None
    base_rate: float = 0.09
    risk_thresholds: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing[:5]}")
        return X[self.feature_names]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(blunder)."""
        return np.asarray(self.model.predict_proba(self._frame(X))[:, 1], dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(blunder) (falls back to raw if no calibrator is stored)."""
        raw = self.predict_raw(X)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def risk_band(self, p: float) -> str:
        t = self.risk_thresholds or default_thresholds(self.base_rate)
        if p < t["low"]:
            return "low"
        if p < t["moderate"]:
            return "moderate"
        if p < t["high"]:
            return "high"
        return "very high"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        return joblib.load(path)
