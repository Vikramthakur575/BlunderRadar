"""Shared fixtures: a tiny, fast model bundle built from synthetic self-play positions."""

from __future__ import annotations

import random

import chess
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from blunder_radar.features.pipeline import extract_all_features
from blunder_radar.modeling.bundle import ModelBundle, default_thresholds
from blunder_radar.modeling.dataset import feature_columns


def random_games(n_games: int = 40, max_plies: int = 60, seed: int = 0) -> pd.DataFrame:
    """Positions from random legal play, shaped like the real positions dataset."""
    rng = random.Random(seed)
    rows = []
    for g in range(n_games):
        board = chess.Board()
        clocks = {chess.WHITE: 300.0, chess.BLACK: 300.0}
        w_elo, b_elo = rng.randint(900, 2300), rng.randint(900, 2300)
        for ply in range(1, max_plies + 1):
            moves = list(board.legal_moves)
            if not moves:
                break
            mover = board.turn
            move = rng.choice(moves)
            clocks[mover] = max(0.0, clocks[mover] - rng.uniform(0.5, 9.0))
            rows.append(
                {
                    "game_id": f"g{g:04d}",
                    "game_timestamp": f"2024.01.01 00:{g // 60:02d}:{g % 60:02d}",
                    "ply": ply,
                    "fen_before": board.fen(),
                    "mover_color": "white" if mover == chess.WHITE else "black",
                    "mover_rating": w_elo if mover == chess.WHITE else b_elo,
                    "opponent_rating": b_elo if mover == chess.WHITE else w_elo,
                    "clock_remaining_seconds": clocks[mover],
                    "increment_seconds": 0.0,
                    "rating_band": "mid",
                }
            )
            board.push(move)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def random_positions() -> pd.DataFrame:
    return random_games()


@pytest.fixture(scope="session")
def tiny_features(random_positions) -> pd.DataFrame:
    feats = extract_all_features(random_positions, include_eval_before=False, num_workers=1)
    rng = np.random.default_rng(0)
    signal = (
        0.8 * feats["mover_hanging_pieces"]
        + 0.6 * feats["is_time_trouble"]
        + 0.3 * feats["is_in_check"]
    )
    p = 1 / (1 + np.exp(-(signal - 2.2)))
    feats["is_blunder"] = (rng.random(len(feats)) < p).astype(int)
    return feats


@pytest.fixture(scope="session")
def tiny_bundle(tiny_features) -> ModelBundle:
    cols = feature_columns(with_eval=False)
    X, y = tiny_features[cols], tiny_features["is_blunder"]
    model = XGBClassifier(
        n_estimators=40, max_depth=3, tree_method="hist", n_jobs=1, random_state=0, verbosity=0
    )
    model.fit(X, y)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    cal.fit(model.predict_proba(X)[:, 1], y)
    base = float(y.mean())
    return ModelBundle(
        model=model, feature_names=cols, algorithm="xgboost", with_eval=False, calibrator=cal,
        base_rate=base, risk_thresholds=default_thresholds(base),
        meta={"calibration": "isotonic", "params": {}},
    )


@pytest.fixture(scope="session")
def tiny_bundle_path(tiny_bundle, tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("models") / "best_model.joblib"
    tiny_bundle.save(path)
    return str(path)
