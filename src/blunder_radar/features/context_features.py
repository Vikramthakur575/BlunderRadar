"""
Context and temporal features extracted strictly from pre-move game metadata.
Vectorized over pandas DataFrames for high-performance extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_FEATURE_NAMES: list[str] = [
    "move_number",
    "ply",
    "mover_rating",
    "opponent_rating",
    "rating_diff",
    "clock_remaining_seconds",
    "is_time_trouble",
    "clock_missing",
    "increment_seconds",
    "prev_move_time_spent",
]

EVAL_FEATURE_NAMES: list[str] = [
    "eval_before",
    "win_prob_before",
]


def add_pre_move_clock_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive strictly pre-move clock columns from Lichess [%clk] annotations.

    Lichess stores, on every move, the mover's clock AFTER that move was played. Using that
    value directly as a "pre-move" feature would leak the time spent on the move itself.
    The mover's clock at the START of their turn is their own clock after their previous
    move, so this function shifts the series by one move per (game, colour).

    Adds:
      clock_before_seconds   mover's remaining time when their turn began. For a player's
                             first move (free on Lichess) their own post-move clock is used.
      prev_move_time_spent   seconds the mover spent on their previous move
                             (clock_before + increment - clock_after); NaN when unknown.

    Rows must contain game_id, mover_color, ply, clock_remaining_seconds. The original
    index and row order are preserved.
    """
    required = {"game_id", "mover_color", "ply", "clock_remaining_seconds"}
    if not required.issubset(df.columns):
        return df.copy()

    out = df.copy()
    order = out.sort_values(["game_id", "ply"], kind="stable").index
    work = out.loc[order]
    inc = (
        work["increment_seconds"].fillna(0.0).astype(float)
        if "increment_seconds" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    clock_after = work["clock_remaining_seconds"].astype(float)
    keys = [work["game_id"], work["mover_color"]]

    prev_after = clock_after.groupby(keys, sort=False).shift(1)
    is_first = prev_after.isna() & clock_after.notna()
    clock_before = prev_after.copy()
    clock_before[is_first] = clock_after[is_first]

    spent = (clock_before + inc - clock_after).clip(lower=0.0)
    spent[is_first | clock_before.isna() | clock_after.isna()] = np.nan
    prev_spent = spent.groupby(keys, sort=False).shift(1)

    out["clock_before_seconds"] = clock_before.reindex(out.index)
    out["prev_move_time_spent"] = prev_spent.reindex(out.index)
    return out


def extract_context_features_vectorized(
    df: pd.DataFrame,
    time_trouble_threshold: float = 30.0,
    clock_default_imputation: float = 180.0,
) -> pd.DataFrame:
    """
    Extract vectorized context features from a DataFrame of position records.
    Strictly pre-move: only uses ply, ratings, clock remaining before move, and increment.
    """
    features = pd.DataFrame(index=df.index)

    # 1. Ply and Move Number
    ply = df["ply"].astype(int) if "ply" in df.columns else pd.Series(1, index=df.index)
    features["ply"] = ply
    features["move_number"] = (ply + 1) // 2

    # 2. Ratings and Rating Difference
    mover_r = (
        df["mover_rating"].astype(float)
        if "mover_rating" in df.columns
        else pd.Series(1500.0, index=df.index)
    )
    opp_r = (
        df["opponent_rating"].astype(float)
        if "opponent_rating" in df.columns
        else pd.Series(1500.0, index=df.index)
    )
    features["mover_rating"] = mover_r
    features["opponent_rating"] = opp_r
    features["rating_diff"] = mover_r - opp_r

    # 3. Clock Remaining & Time Trouble
    # Prefer the derived pre-move clock. Without it the caller must supply a clock that
    # was already measured before the move (as the live scorer does).
    clock_col = (
        "clock_before_seconds"
        if "clock_before_seconds" in df.columns
        else "clock_remaining_seconds"
    )
    if clock_col in df.columns:
        clock_raw = df[clock_col].astype(float)
        clock_missing = clock_raw.isna().astype(int)
        # Impute missing clocks with median/default to guarantee 0 nulls for downstream ML models
        clock_imputed = clock_raw.fillna(clock_default_imputation)
        # Time trouble flag: remaining time is below threshold (e.g. 30s) and not missing
        is_trouble = ((clock_raw <= time_trouble_threshold) & (clock_missing == 0)).astype(int)
    else:
        clock_imputed = pd.Series(clock_default_imputation, index=df.index)
        clock_missing = pd.Series(1, index=df.index)
        is_trouble = pd.Series(0, index=df.index)

    features["clock_remaining_seconds"] = clock_imputed
    features["is_time_trouble"] = is_trouble
    features["clock_missing"] = clock_missing

    # 4. Increment
    if "increment_seconds" in df.columns:
        features["increment_seconds"] = df["increment_seconds"].fillna(0.0).astype(float)
    else:
        features["increment_seconds"] = 0.0

    # 5. Time spent on the mover's previous move (pre-move information)
    if "prev_move_time_spent" in df.columns:
        features["prev_move_time_spent"] = df["prev_move_time_spent"].fillna(0.0).astype(float)
    else:
        features["prev_move_time_spent"] = 0.0

    return features


def extract_eval_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract optional pre-move evaluation features (mover's perspective).
    Strictly pre-move: only uses eval_before and win_prob_before.
    """
    features = pd.DataFrame(index=df.index)

    if "eval_before" in df.columns:
        features["eval_before"] = df["eval_before"].fillna(0.0).astype(float)
    else:
        features["eval_before"] = 0.0

    if "win_prob_before" in df.columns:
        features["win_prob_before"] = df["win_prob_before"].fillna(0.5).astype(float)
    elif "eval_before" in df.columns:
        # Compute if not already in df
        scaled = np.clip(features["eval_before"] / 400.0, -10.0, 10.0)
        features["win_prob_before"] = 1.0 / (1.0 + (10.0 ** (-scaled)))
    else:
        features["win_prob_before"] = 0.5

    return features
