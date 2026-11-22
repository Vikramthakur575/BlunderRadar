import numpy as np
import pandas as pd
import pytest

from blunder_radar.modeling.dataset import (
    assert_no_game_leakage,
    chronological_game_split,
    feature_columns,
)
from blunder_radar.modeling.metrics import evaluate, expected_calibration_error
from blunder_radar.modeling.train import PlattCalibrator, fit_calibrator


def _frame(n_games=60, per_game=10):
    rows = []
    for g in range(n_games):
        for p in range(per_game):
            rows.append({"game_id": f"g{g:03d}", "ply": p + 1,
                         "game_timestamp": f"2024.01.01 00:{g // 60:02d}:{g % 60:02d}"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_split_has_no_game_overlap():
    df = _frame()
    assert_no_game_leakage(df, chronological_game_split(df))


def test_split_is_chronological():
    df = _frame()
    df["split"] = chronological_game_split(df)
    last = df.groupby("split")["game_timestamp"].agg(["min", "max"])
    assert last.loc["train", "max"] <= last.loc["val", "min"]
    assert last.loc["val", "max"] <= last.loc["test", "min"]


def test_split_proportions_are_close():
    df = _frame(200)
    shares = chronological_game_split(df, 0.15, 0.15).value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)


def test_leakage_check_catches_a_split_game():
    df = _frame(10)
    bad = pd.Series(["train"] * len(df))
    bad.iloc[0] = "test"
    with pytest.raises(AssertionError):
        assert_no_game_leakage(df, bad)


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        chronological_game_split(_frame(5), 0.6, 0.6)


def test_eval_features_are_opt_in():
    base = feature_columns(with_eval=False)
    assert "eval_before" not in base
    assert set(feature_columns(with_eval=True)) - set(base) == {"eval_before", "win_prob_before"}


def test_ece_is_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.4, 50000)
    y = (rng.random(50000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_is_large_when_miscalibrated():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.1).astype(int)
    assert expected_calibration_error(y, np.full(20000, 0.6)) > 0.4


def test_evaluate_reports_all_metrics_and_beats_baseline():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.1).astype(int)
    informative = np.clip(0.05 + 0.5 * y * rng.random(5000), 0, 1)
    m = evaluate(y, informative)
    assert {"pr_auc", "roc_auc", "log_loss", "brier", "ece", "base_rate"} <= set(m)
    assert m["pr_auc"] > m["base_rate"] * 2


def test_isotonic_calibrator_is_monotone_and_fixes_overconfidence():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < raw * 0.3).astype(int)  # truth is 0.3x the raw score
    cal = fit_calibrator(raw, y, "isotonic")
    grid = np.linspace(0, 1, 50)
    assert np.all(np.diff(cal.predict(grid)) >= -1e-12)
    assert evaluate(y, cal.predict(raw))["ece"] < evaluate(y, raw)["ece"]


def test_platt_calibrator_outputs_valid_probabilities():
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.01, 0.99, 5000)
    y = (rng.random(5000) < raw * 0.5).astype(int)
    out = PlattCalibrator().fit(raw, y).predict(raw)
    assert out.min() > 0 and out.max() < 1
