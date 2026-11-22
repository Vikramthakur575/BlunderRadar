"""
Unit and integration tests for Phase 2 Feature Engineering.
Validates:
1. Hand-verified positions (starting position, pin, hanging queen, king in check)
2. Mathematical correctness and domain expectations
3. Strict proof that features NEVER use or leak post-move information
4. Vectorized context features and optional eval_before configuration
"""

import chess
import pandas as pd

from blunder_radar.features.chess_features import (
    CHESS_FEATURE_NAMES,
    extract_chess_features_from_fen,
)
from blunder_radar.features.context_features import (
    CONTEXT_FEATURE_NAMES,
    EVAL_FEATURE_NAMES,
    extract_context_features_vectorized,
)
from blunder_radar.features.pipeline import extract_all_features


def test_hand_verified_starting_position():
    """Verify standard initial board state against chess invariants."""
    res = extract_chess_features_from_fen(chess.STARTING_FEN)

    # Material & Piece Counts
    assert res["mover_pawns"] == 8
    assert res["mover_knights"] == 2
    assert res["mover_bishops"] == 2
    assert res["mover_rooks"] == 2
    assert res["mover_queens"] == 1
    assert res["opp_pawns"] == 8
    assert res["opp_knights"] == 2
    assert res["opp_bishops"] == 2
    assert res["opp_rooks"] == 2
    assert res["opp_queens"] == 1
    assert res["total_piece_count"] == 30
    assert res["mover_material"] == 39
    assert res["opp_material"] == 39
    assert res["material_diff"] == 0
    assert res["mover_has_bishop_pair"] == 1
    assert res["opp_has_bishop_pair"] == 1

    # Mobility: 16 pawn moves (1 or 2 squares) + 4 knight moves = 20 legal moves
    assert res["mover_mobility"] == 20
    assert res["opp_mobility"] == 20
    assert res["mobility_diff"] == 0

    # King Safety & Castling Rights
    assert res["mover_can_castle_kingside"] == 1
    assert res["mover_can_castle_queenside"] == 1
    assert res["mover_has_castled"] == 0
    assert res["opp_can_castle_kingside"] == 1
    assert res["opp_can_castle_queenside"] == 1
    assert res["opp_has_castled"] == 0
    # White King on e1 defended by pawns on d2, e2, f2
    assert res["mover_pawn_shield"] == 3
    assert res["opp_pawn_shield"] == 3
    assert res["mover_king_zone_attackers"] == 0
    assert res["opp_king_zone_attackers"] == 0

    # Checks & Pins
    assert res["is_in_check"] == 0
    assert res["num_checkers"] == 0
    assert res["can_give_check"] == 0
    assert res["mover_pinned_pieces"] == 0
    assert res["opp_pinned_pieces"] == 0

    # Hanging & Attacked Pieces
    assert res["mover_attacked_pieces"] == 0
    assert res["opp_attacked_pieces"] == 0
    assert res["mover_hanging_pieces"] == 0
    assert res["opp_hanging_pieces"] == 0

    # Tension & Phase
    assert res["board_tension_piece_attacks"] == 0
    assert res["board_tension_pawn_clashes"] == 0
    assert res["game_phase_score"] == 1.0
    assert res["is_opening"] == 1
    assert res["is_endgame"] == 0


def test_hand_verified_pin_position():
    """
    Absolute pin:
    White King on e1, White Bishop on e4, Black Rook on e8.
    The Bishop on e4 is pinned to White's King along the e-file.
    """
    fen_pin = "4r3/8/8/8/4B3/8/8/4K3 w - - 0 1"
    res = extract_chess_features_from_fen(fen_pin)

    assert res["mover_pinned_pieces"] == 1
    assert res["opp_pinned_pieces"] == 0
    assert res["is_in_check"] == 0
    # Bishop on e4 is attacked by the rook and has 0 defenders -> hanging
    assert res["mover_hanging_pieces"] >= 1
    assert res["mover_hanging_value"] >= 3


def test_hand_verified_hanging_queen():
    """
    Hanging Queen:
    White Queen on d4 attacked by Black Rook on d8 with 0 friendly defenders.
    """
    fen_hanging_q = "3r2k1/8/8/8/3Q4/8/8/6K1 w - - 0 1"
    res = extract_chess_features_from_fen(fen_hanging_q)

    assert res["mover_queens"] == 1
    assert res["mover_attacked_pieces"] == 1
    assert res["mover_attacked_value"] == 9
    assert res["mover_hanging_pieces"] == 1
    assert res["mover_hanging_value"] == 9
    assert res["is_in_check"] == 0


def test_hand_verified_king_in_check():
    """
    King in check:
    White King on e1 checked by Black Queen on e2+.
    """
    fen_check = "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"
    res = extract_chess_features_from_fen(fen_check)

    assert res["is_in_check"] == 1
    assert res["num_checkers"] == 1
    # White King has legal moves to capture the queen or run
    assert res["mover_mobility"] > 0


def test_board_tension_and_pawn_clashes():
    """
    Position with central pawn clash (e4 vs d5 tension):
    1. e4 d5
    """
    fen_tension = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    res = extract_chess_features_from_fen(fen_tension)

    assert res["board_tension_pawn_clashes"] == 1
    assert res["mover_attacked_pieces"] >= 1
    assert res["opp_attacked_pieces"] >= 1


def test_context_features_vectorized():
    df = pd.DataFrame(
        {
            "ply": [1, 2, 40, 60],
            "mover_rating": [1500, 1550, 1800, 1200],
            "opponent_rating": [1520, 1500, 1750, 1400],
            "clock_remaining_seconds": [300.0, 25.0, 12.5, None],
            "increment_seconds": [3.0, 0.0, 5.0, None],
        }
    )

    ctx = extract_context_features_vectorized(df, time_trouble_threshold=30.0)

    # Move numbers
    assert list(ctx["move_number"]) == [1, 1, 20, 30]
    # Rating differences
    assert list(ctx["rating_diff"]) == [-20.0, 50.0, 50.0, -200.0]
    # Time trouble flags: 300s (no), 25s (yes), 12.5s (yes), None (imputed/missing, no)
    assert list(ctx["is_time_trouble"]) == [0, 1, 1, 0]
    # Clock missing flag
    assert list(ctx["clock_missing"]) == [0, 0, 0, 1]
    # Increment
    assert ctx["increment_seconds"].iloc[0] == 3.0
    assert ctx["increment_seconds"].iloc[3] == 0.0


def test_optional_eval_features():
    df = pd.DataFrame(
        {
            "ply": [1, 10],
            "fen_before": [
                chess.STARTING_FEN,
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ],
            "eval_before": [20.0, -15.0],
            "win_prob_before": [0.528, 0.478],
        }
    )

    # Test WITH eval_before
    feat_with_eval = extract_all_features(df, include_eval_before=True)
    assert "eval_before" in feat_with_eval.columns
    assert "win_prob_before" in feat_with_eval.columns

    # Test WITHOUT eval_before
    feat_without_eval = extract_all_features(df, include_eval_before=False)
    assert "eval_before" not in feat_without_eval.columns
    assert "win_prob_before" not in feat_without_eval.columns


def test_strict_no_post_move_leakage():
    """
    CRITICAL LEAKAGE TEST:
    Proves mathematically and programmatically that changing or removing post-move
    attributes (the played move, eval_after, eval_drop, is_blunder, win_prob_after,
    win_prob_drop) has ZERO effect on any extracted feature.
    """
    df_clean = pd.DataFrame(
        {
            "game_id": ["game_01", "game_02"],
            "ply": [15, 30],
            "mover_color": ["white", "black"],
            "mover_rating": [1600, 1750],
            "opponent_rating": [1620, 1700],
            "rating_diff": [-20, 50],
            "rating_band": ["mid", "mid"],
            "clock_remaining_seconds": [180.0, 15.0],
            "increment_seconds": [2.0, 0.0],
            "fen_before": [
                "r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
                "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
            ],
            "eval_before": [30.0, -50.0],
            "win_prob_before": [0.54, 0.42],
            # Post-move columns:
            "move": ["e1g1", "c1d2"],
            "eval_after": [30.0, 300.0],
            "eval_drop": [0.0, 350.0],
            "is_blunder": [0, 1],
            "win_prob_after": [0.54, 0.15],
            "win_prob_drop": [0.0, 0.27],
            "is_blunder_prob": [0, 1],
        }
    )

    # 1. Extract baseline features
    features_base = extract_all_features(df_clean, include_eval_before=True)

    # 2. Corrupt/flip ALL post-move columns
    df_corrupted = df_clean.copy()
    df_corrupted["move"] = ["a2a3", "h7h6"]
    df_corrupted["eval_after"] = [-9999.0, 9999.0]
    df_corrupted["eval_drop"] = [10000.0, -10000.0]
    df_corrupted["is_blunder"] = [1, 0]
    df_corrupted["win_prob_after"] = [0.0001, 0.9999]
    df_corrupted["win_prob_drop"] = [0.99, -0.99]
    df_corrupted["is_blunder_prob"] = [1, 0]

    features_corrupted = extract_all_features(df_corrupted, include_eval_before=True)

    # List of all engineered feature columns
    feature_columns = CHESS_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + EVAL_FEATURE_NAMES

    # Assert 100% bit-for-bit equality on every single feature column
    for col in feature_columns:
        assert col in features_base.columns, f"Feature column {col} missing in base output"
        pd.testing.assert_series_equal(
            features_base[col],
            features_corrupted[col],
            obj=f"Feature {col} leaked post-move information!",
        )

    # 3. Test completely dropping post-move columns from input
    df_no_post = df_clean.drop(
        columns=[
            "move",
            "eval_after",
            "eval_drop",
            "is_blunder",
            "win_prob_after",
            "win_prob_drop",
            "is_blunder_prob",
        ]
    )
    features_no_post = extract_all_features(df_no_post, include_eval_before=True)

    for col in feature_columns:
        pd.testing.assert_series_equal(
            features_base[col],
            features_no_post[col],
            obj=f"Feature {col} changed when post-move columns were removed!",
        )

