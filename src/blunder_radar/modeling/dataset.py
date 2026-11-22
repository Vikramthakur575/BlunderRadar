"""
Dataset loading and leakage-safe splitting for model training.

Positions from the same game are highly correlated, so every split is made at the
GAME level, and chronologically by game start time (train on earlier games, validate and
test on later ones). A game can never appear in two splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from blunder_radar.features.chess_features import CHESS_FEATURE_NAMES
from blunder_radar.features.context_features import CONTEXT_FEATURE_NAMES, EVAL_FEATURE_NAMES

TARGET_COL = "is_blunder"


def feature_columns(with_eval: bool) -> list[str]:
    """Ordered model input columns. eval_before features are an explicit opt-in."""
    cols = list(CHESS_FEATURE_NAMES) + list(CONTEXT_FEATURE_NAMES)
    if with_eval:
        cols += list(EVAL_FEATURE_NAMES)
    return cols


def load_features(path: str | Path) -> pd.DataFrame:
    """Read a (hive-partitioned) features dataset into a DataFrame."""
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    df = dataset.to_table().to_pandas()
    if "rating_band" in df.columns:
        df["rating_band"] = df["rating_band"].astype(str)
    return df.reset_index(drop=True)


def chronological_game_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.Series:
    """
    Label every row 'train', 'val' or 'test'.

    Games are ordered by start timestamp (game_id breaks ties) and cut so that roughly
    (1 - val_size - test_size) of POSITIONS fall in train, then val, then test. Cuts
    happen only at game boundaries.
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")

    games = (
        df.groupby("game_id", sort=False)
        .agg(ts=("game_timestamp", "min"), n=("game_id", "size"))
        .reset_index()
        .sort_values(["ts", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    cum_share = games["n"].cumsum() / games["n"].sum()
    train_end = 1.0 - val_size - test_size
    val_end = 1.0 - test_size
    split = np.where(
        cum_share <= train_end, "train", np.where(cum_share <= val_end, "val", "test")
    )
    mapping = dict(zip(games["game_id"], split, strict=True))
    return df["game_id"].map(mapping).rename("split")


def assert_no_game_leakage(df: pd.DataFrame, split: pd.Series) -> None:
    """Raise if any game spans more than one split."""
    per_game = (
        pd.DataFrame({"game_id": df["game_id"], "split": split})
        .groupby("game_id")["split"]
        .nunique()
    )
    bad = int((per_game > 1).sum())
    if bad:
        raise AssertionError(f"{bad} games appear in more than one split")
