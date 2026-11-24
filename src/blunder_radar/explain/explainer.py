"""
Per-prediction explanations.

Tree models (XGBoost / LightGBM) expose exact TreeSHAP values natively, so serving does not
need the heavy `shap` package. Contributions are in log-odds space (the model's margin):
positive values push the blunder risk UP, negative values push it DOWN. For the logistic
regression baseline the equivalent is coefficient * standardised value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from blunder_radar.modeling.bundle import ModelBundle


def tree_contributions(bundle: ModelBundle, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Return (contributions[n, n_features], bias) in log-odds for the bundle's raw model.
    """
    frame = bundle._frame(X)
    model = bundle.model
    name = bundle.algorithm

    if name == "xgboost":
        import xgboost as xgb

        booster = model.get_booster()
        contribs = booster.predict(xgb.DMatrix(frame), pred_contribs=True)
        return np.asarray(contribs[:, :-1]), float(np.mean(contribs[:, -1]))

    if name == "lightgbm":
        contribs = model.predict(frame, pred_contrib=True)
        return np.asarray(contribs[:, :-1]), float(np.mean(contribs[:, -1]))

    if name == "logistic_regression":
        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]
        z = (frame.to_numpy(dtype=float) - scaler.mean_) / np.where(scaler.scale_ == 0, 1, scaler.scale_)
        return z * clf.coef_[0], float(clf.intercept_[0])

    raise ValueError(f"No contribution method for algorithm '{name}'")


def _secs(v: float) -> str:
    return f"{v:.0f}s"


# feature -> function(value) -> human phrase describing the *situation*
PHRASES: dict[str, Callable[[float], str]] = {
    "clock_remaining_seconds": lambda v: f"{_secs(v)} left on the clock",
    "clock_missing": lambda v: "clock time not recorded" if v >= 1 else "clock time recorded",
    "is_time_trouble": lambda v: "in time trouble" if v >= 1 else "not in time trouble",
    "prev_move_time_spent": lambda v: f"spent {_secs(v)} on the previous move",
    "increment_seconds": lambda v: f"{v:.0f}s increment per move",
    "mover_rating": lambda v: f"player rated {v:.0f}",
    "opponent_rating": lambda v: f"opponent rated {v:.0f}",
    "rating_diff": lambda v: f"rating gap of {v:+.0f}",
    "move_number": lambda v: f"move number {v:.0f}",
    "ply": lambda v: (
        f"early in the game (ply {v:.0f})" if v < 20 else f"deep into the game (ply {v:.0f})"
    ),
    "material_diff": lambda v: f"material balance {v:+.0f}",
    "mover_mobility": lambda v: f"{v:.0f} legal moves available",
    "opp_mobility": lambda v: f"opponent has {v:.0f} potential moves",
    "mobility_diff": lambda v: f"mobility difference {v:+.0f}",
    "mover_hanging_pieces": lambda v: f"{v:.0f} undefended piece(s) under attack",
    "mover_hanging_value": lambda v: f"{v:.0f} points of undefended material attacked",
    "mover_attacked_pieces": lambda v: f"{v:.0f} of your pieces attacked",
    "mover_attacked_value": lambda v: f"{v:.0f} points of material attacked",
    "opp_hanging_pieces": lambda v: f"{v:.0f} opponent piece(s) hanging",
    "opp_hanging_value": lambda v: f"{v:.0f} points of opponent material hanging",
    "mover_king_zone_attackers": lambda v: f"{v:.0f} enemy piece(s) near your king",
    "mover_king_zone_attacks": lambda v: f"{v:.0f} attacks on your king zone",
    "mover_pawn_shield": lambda v: f"{v:.0f} pawn(s) shielding your king",
    "mover_pinned_pieces": lambda v: f"{v:.0f} of your pieces pinned",
    "is_in_check": lambda v: "your king is in check" if v >= 1 else "not in check",
    "num_checkers": lambda v: f"{v:.0f} piece(s) giving check",
    "can_give_check": lambda v: "a checking move is available" if v >= 1 else "no checks available",
    "total_piece_count": lambda v: f"{v:.0f} pieces on the board",
    "game_phase_score": lambda v: "a complex middlegame" if v > 0.5 else "a simplified endgame",
    "is_endgame": lambda v: "an endgame" if v >= 1 else "not an endgame",
    "is_opening": lambda v: "still in the opening" if v >= 1 else "past the opening",
    "board_tension_total_attacked_value": lambda v: f"{v:.0f} points of material in tension",
    "board_tension_piece_attacks": lambda v: f"{v:.0f} pieces attacked in total",
    "board_tension_pawn_clashes": lambda v: f"{v:.0f} pawn clash(es)",
    "eval_before": lambda v: f"engine evaluation {v / 100:+.1f}",
    "win_prob_before": lambda v: f"{v * 100:.0f}% estimated winning chances",
    # piece inventory
    "mover_pawns": lambda v: f"{v:.0f} pawn(s) left for the player",
    "mover_knights": lambda v: f"{v:.0f} knight(s) for the player",
    "mover_bishops": lambda v: f"{v:.0f} bishop(s) for the player",
    "mover_rooks": lambda v: f"{v:.0f} rook(s) for the player",
    "mover_queens": lambda v: f"{v:.0f} queen(s) for the player",
    "opp_pawns": lambda v: f"{v:.0f} pawn(s) left for the opponent",
    "opp_knights": lambda v: f"{v:.0f} knight(s) for the opponent",
    "opp_bishops": lambda v: f"{v:.0f} bishop(s) for the opponent",
    "opp_rooks": lambda v: f"{v:.0f} rook(s) for the opponent",
    "opp_queens": lambda v: f"{v:.0f} queen(s) for the opponent",
    "mover_piece_count": lambda v: f"{v:.0f} pieces for the player",
    "opp_piece_count": lambda v: f"{v:.0f} pieces for the opponent",
    "mover_material": lambda v: f"{v:.0f} points of material for the player",
    "opp_material": lambda v: f"{v:.0f} points of material for the opponent",
    "mover_has_bishop_pair": lambda v: "the player has the bishop pair" if v >= 1
    else "the player lacks the bishop pair",
    "opp_has_bishop_pair": lambda v: "the opponent has the bishop pair" if v >= 1
    else "the opponent lacks the bishop pair",
    # castling
    "mover_can_castle_kingside": lambda v: "can still castle kingside" if v >= 1
    else "cannot castle kingside",
    "mover_can_castle_queenside": lambda v: "can still castle queenside" if v >= 1
    else "cannot castle queenside",
    "mover_has_castled": lambda v: "the player has castled" if v >= 1 else "the player has not castled",
    "opp_can_castle_kingside": lambda v: "the opponent can still castle kingside" if v >= 1
    else "the opponent cannot castle kingside",
    "opp_can_castle_queenside": lambda v: "the opponent can still castle queenside" if v >= 1
    else "the opponent cannot castle queenside",
    "opp_has_castled": lambda v: "the opponent has castled" if v >= 1 else "the opponent has not castled",
    # king safety and pressure
    "opp_king_zone_attackers": lambda v: f"{v:.0f} of the player's pieces bearing on the opponent's king",
    "opp_king_zone_attacks": lambda v: f"{v:.0f} attacks on the opponent's king zone",
    "opp_pawn_shield": lambda v: f"{v:.0f} pawn(s) shielding the opponent's king",
    "opp_pinned_pieces": lambda v: f"{v:.0f} opponent piece(s) pinned",
    "opp_attacked_pieces": lambda v: f"{v:.0f} opponent piece(s) under attack",
    "opp_attacked_value": lambda v: f"{v:.0f} points of opponent material under attack",
}


def _phrase(feature: str, value: float) -> str:
    fn = PHRASES.get(feature)
    if fn is not None:
        try:
            return fn(float(value))
        except (TypeError, ValueError):
            pass
    return f"{feature.replace('_', ' ')} = {value:g}"


def explain_row(
    bundle: ModelBundle,
    X: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Explain the first row of X. Returns the top_k factors by absolute contribution:
    [{feature, value, contribution, direction, text}, ...].
    """
    contribs, _ = tree_contributions(bundle, X.iloc[[0]])
    row = contribs[0]
    values = bundle._frame(X.iloc[[0]]).iloc[0]
    order = np.argsort(-np.abs(row))[:top_k]
    out: list[dict[str, Any]] = []
    for i in order:
        feat = bundle.feature_names[i]
        c = float(row[i])
        direction = "raises risk" if c > 0 else "lowers risk"
        out.append(
            {
                "feature": feat,
                "value": float(values[feat]),
                "contribution": round(c, 4),
                "direction": direction,
                "text": f"{direction}: {_phrase(feat, values[feat])}",
            }
        )
    return out


def summarise(prob: float, band: str, base_rate: float, factors: list[dict[str, Any]]) -> str:
    """One-sentence plain-English summary of a prediction."""
    ratio = prob / base_rate if base_rate > 0 else float("nan")
    up = [f["text"].split(": ", 1)[1] for f in factors if f["direction"] == "raises risk"][:3]
    head = f"{band.capitalize()} blunder risk ({prob * 100:.1f}%, {ratio:.1f}x the average)"
    if up:
        return f"{head}, driven by: " + "; ".join(up) + "."
    return head + "."
