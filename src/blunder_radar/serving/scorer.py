"""
Turns a position (FEN) plus rating/clock context into a calibrated blunder probability,
a risk band, plain-English reasons and a danger heatmap.

Feature construction reuses exactly the code used in training, so there is no
train/serve skew. Only information available BEFORE the move is played is used.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import pandas as pd

from blunder_radar.data.eval_utils import (
    calculate_blunder,
    centipawns_to_win_prob,
    convert_to_mover_perspective,
    extract_clock_seconds,
    extract_eval_cp,
)
from blunder_radar.data.stream_parser import extract_increment_seconds
from blunder_radar.explain.explainer import explain_row, summarise, tree_contributions
from blunder_radar.explain.heatmap import risky_squares
from blunder_radar.features.chess_features import extract_chess_features_from_board
from blunder_radar.features.context_features import (
    add_pre_move_clock_columns,
    extract_context_features_vectorized,
    extract_eval_features_vectorized,
)
from blunder_radar.modeling.bundle import ModelBundle

DEFAULT_MODEL_PATH = "models/best_model.joblib"
DEFAULT_EVAL_MODEL_PATH = "models/best_model_with_eval.joblib"
DEFAULT_CLOCK_SECONDS = 300.0


def ply_from_board(board: chess.Board) -> int:
    """Half-move number of the NEXT move (white's first move is ply 1)."""
    return (board.fullmove_number - 1) * 2 + (1 if board.turn == chess.WHITE else 2)


def validate_board(fen: str) -> chess.Board:
    """Parse a FEN and make sure the side to move actually has a move to play."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc
    if not board.is_valid():
        raise ValueError(f"Illegal position: {board.status().name}")
    if not any(board.legal_moves):
        raise ValueError("Position is game over (no legal moves)")
    return board


class BlunderScorer:
    """Scores positions with the deployed model bundle(s)."""

    def __init__(self, bundle: ModelBundle, eval_bundle: ModelBundle | None = None) -> None:
        self.bundle = bundle
        self.eval_bundle = eval_bundle

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path | None = None,
        eval_model_path: str | Path | None = None,
    ) -> BlunderScorer:
        model_path = Path(model_path or os.getenv("BLUNDER_RADAR_MODEL", DEFAULT_MODEL_PATH))
        bundle = ModelBundle.load(model_path)
        eval_path = Path(
            eval_model_path
            or os.getenv("BLUNDER_RADAR_EVAL_MODEL", "")
            or model_path.with_name("best_model_with_eval.joblib")
        )
        eval_bundle = ModelBundle.load(eval_path) if eval_path.exists() else None
        return cls(bundle, eval_bundle)

    # ------------------------------------------------------------------ features
    def build_features(
        self,
        board: chess.Board,
        mover_rating: float,
        opponent_rating: float,
        clock_seconds: float | None = None,
        increment_seconds: float = 0.0,
        prev_move_time_spent: float | None = None,
        eval_before_cp: float | None = None,
        ply: int | None = None,
    ) -> pd.DataFrame:
        """One-row feature frame in the model's column order."""
        row = extract_chess_features_from_board(board)
        ctx_in = pd.DataFrame(
            {
                "ply": [ply if ply is not None else ply_from_board(board)],
                "mover_rating": [float(mover_rating)],
                "opponent_rating": [float(opponent_rating)],
                "clock_before_seconds": [np.nan if clock_seconds is None else float(clock_seconds)],
                "increment_seconds": [float(increment_seconds or 0.0)],
                "prev_move_time_spent": [
                    np.nan if prev_move_time_spent is None else float(prev_move_time_spent)
                ],
            }
        )
        ctx = extract_context_features_vectorized(ctx_in).iloc[0].to_dict()
        row.update(ctx)
        if eval_before_cp is not None:
            ev_in = pd.DataFrame(
                {
                    "eval_before": [float(eval_before_cp)],
                    "win_prob_before": [centipawns_to_win_prob(float(eval_before_cp))],
                }
            )
            row.update(extract_eval_features_vectorized(ev_in).iloc[0].to_dict())
        return pd.DataFrame([row])

    def _pick(self, eval_before_cp: float | None) -> ModelBundle:
        if eval_before_cp is not None and self.eval_bundle is not None:
            return self.eval_bundle
        return self.bundle

    # ------------------------------------------------------------------ scoring
    def score(
        self,
        fen: str,
        mover_rating: float = 1500,
        opponent_rating: float = 1500,
        clock_seconds: float | None = None,
        increment_seconds: float = 0.0,
        prev_move_time_spent: float | None = None,
        eval_before_cp: float | None = None,
        ply: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Score the side to move in `fen`."""
        board = validate_board(fen)
        bundle = self._pick(eval_before_cp)
        feats = self.build_features(
            board, mover_rating, opponent_rating, clock_seconds, increment_seconds,
            prev_move_time_spent, eval_before_cp, ply,
        )
        prob = float(bundle.predict_proba(feats)[0])
        band = bundle.risk_band(prob)
        factors = explain_row(bundle, feats, top_k=top_k)
        return {
            "probability": round(prob, 5),
            "risk_band": band,
            "base_rate": round(bundle.base_rate, 5),
            "relative_risk": round(prob / bundle.base_rate, 3) if bundle.base_rate else None,
            "summary": summarise(prob, band, bundle.base_rate, factors),
            "factors": factors,
            "hot_squares": {
                chess.square_name(sq): round(w, 3) for sq, w in risky_squares(board).items()
            },
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "model": {
                "algorithm": bundle.algorithm,
                "uses_engine_eval": bundle.with_eval,
                "calibration": bundle.meta.get("calibration"),
            },
        }

    # ------------------------------------------------------------------ whole games
    def score_pgn(
        self,
        pgn_text: str,
        default_rating: float = 1500,
        default_clock: float | None = None,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Score every move of a PGN game: for each position the probability that the side to
        move blunders. Uses [%clk] annotations and Elo headers when present; if the PGN has
        [%eval] annotations the true outcome (did they blunder?) is attached for comparison.
        """
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None or not list(game.mainline_moves()):
            raise ValueError("No moves found in PGN")
        h = game.headers
        white_elo = _int_or(h.get("WhiteElo"), default_rating)
        black_elo = _int_or(h.get("BlackElo"), default_rating)
        inc = extract_increment_seconds(h.get("TimeControl", ""))

        board = game.board()
        records: list[dict[str, Any]] = []
        boards: list[chess.Board] = []
        eval_prev_white = 20.0
        node = game
        ply = 0
        while node.variations:
            nxt = node.variation(0)
            ply += 1
            color = "white" if board.turn == chess.WHITE else "black"
            ev_after_white = extract_eval_cp(nxt)
            rec: dict[str, Any] = {
                "game_id": "pgn",
                "ply": ply,
                "mover_color": color,
                "mover_rating": white_elo if board.turn == chess.WHITE else black_elo,
                "opponent_rating": black_elo if board.turn == chess.WHITE else white_elo,
                "clock_remaining_seconds": extract_clock_seconds(nxt),
                "increment_seconds": inc,
                "fen": board.fen(),
                "san": board.san(nxt.move),
                "uci": nxt.move.uci(),
                "actual_blunder": None,
            }
            if ev_after_white is not None:
                before = convert_to_mover_perspective(eval_prev_white, board.turn)
                after = convert_to_mover_perspective(ev_after_white, board.turn)
                rec["actual_blunder"] = bool(calculate_blunder(before, after)[0])
                rec["eval_before_cp"] = before
                eval_prev_white = ev_after_white
            records.append(rec)
            boards.append(board.copy())
            board.push(nxt.move)
            node = nxt

        frame = add_pre_move_clock_columns(pd.DataFrame(records))
        if default_clock is not None:
            frame["clock_before_seconds"] = frame["clock_before_seconds"].fillna(default_clock)

        bundle = self.bundle
        feat_rows = []
        for i, b in enumerate(boards):
            r = frame.iloc[i]
            feat_rows.append(
                self.build_features(
                    b, r["mover_rating"], r["opponent_rating"],
                    None if pd.isna(r["clock_before_seconds"]) else r["clock_before_seconds"],
                    r["increment_seconds"],
                    None if pd.isna(r["prev_move_time_spent"]) else r["prev_move_time_spent"],
                    None, int(r["ply"]),
                )
            )
        X = pd.concat(feat_rows, ignore_index=True)
        probs = bundle.predict_proba(X)
        contribs, _ = tree_contributions(bundle, X)

        out: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            p = float(probs[i])
            band = bundle.risk_band(p)
            order = np.argsort(-np.abs(contribs[i]))[:top_k]
            row = X.iloc[i]
            from blunder_radar.explain.explainer import _phrase

            factors = [
                {
                    "feature": bundle.feature_names[j],
                    "value": float(row[bundle.feature_names[j]]),
                    "contribution": round(float(contribs[i][j]), 4),
                    "direction": "raises risk" if contribs[i][j] > 0 else "lowers risk",
                    "text": ("raises risk: " if contribs[i][j] > 0 else "lowers risk: ")
                    + _phrase(bundle.feature_names[j], row[bundle.feature_names[j]]),
                }
                for j in order
            ]
            out.append(
                {
                    "ply": rec["ply"],
                    "move_number": (rec["ply"] + 1) // 2,
                    "mover": rec["mover_color"],
                    "san": rec["san"],
                    "uci": rec["uci"],
                    "fen": rec["fen"],
                    "probability": round(p, 5),
                    "risk_band": band,
                    "summary": summarise(p, band, bundle.base_rate, factors),
                    "factors": factors,
                    "hot_squares": {
                        chess.square_name(sq): round(w, 3)
                        for sq, w in risky_squares(boards[i]).items()
                    },
                    "actual_blunder": rec["actual_blunder"],
                }
            )
        return out


def _int_or(value: str | None, default: float) -> int:
    try:
        return int(value) if value not in (None, "", "?") else int(default)
    except (TypeError, ValueError):
        return int(default)
