"""
Unit tests for eval, centipawn, mate score, clock, and blunder calculation utilities.
"""

import chess

from blunder_radar.data.eval_utils import (
    calculate_blunder,
    calculate_blunder_win_prob,
    centipawns_to_win_prob,
    convert_to_mover_perspective,
    extract_clock_seconds,
    get_rating_band,
    parse_eval_str,
)


class DummyNode:
    def __init__(self, comment="", clk=None):
        self.comment = comment
        self._clk = clk

    def clock(self):
        return self._clk


def test_parse_eval_str_centipawns():
    assert parse_eval_str("0.25") == 25.0
    assert parse_eval_str("-1.50") == -150.0
    assert parse_eval_str("0.0") == 0.0
    assert parse_eval_str("3.45") == 345.0
    assert parse_eval_str("invalid") is None
    assert parse_eval_str("") is None


def test_parse_eval_str_mate():
    # White mate in 1
    score_mate_1 = parse_eval_str("#1", mate_score_cp=10000.0)
    assert score_mate_1 == 9990.0

    # White mate in 2
    score_mate_2 = parse_eval_str("#2", mate_score_cp=10000.0)
    assert score_mate_2 == 9980.0

    # Black mate in 1 (White is getting mated)
    score_black_mate_1 = parse_eval_str("#-1", mate_score_cp=10000.0)
    assert score_black_mate_1 == -9990.0

    # Black mate in 3
    score_black_mate_3 = parse_eval_str("#-3", mate_score_cp=10000.0)
    assert score_black_mate_3 == -9970.0


def test_convert_to_mover_perspective():
    white_score = 150.0
    # From White's perspective
    assert convert_to_mover_perspective(white_score, chess.WHITE) == 150.0
    # From Black's perspective
    assert convert_to_mover_perspective(white_score, chess.BLACK) == -150.0

    black_winning_score = -300.0
    assert convert_to_mover_perspective(black_winning_score, chess.WHITE) == -300.0
    assert convert_to_mover_perspective(black_winning_score, chess.BLACK) == 300.0


def test_calculate_blunder_threshold():
    # Drop 250 cp >= 200 -> Blunder
    is_blunder, drop = calculate_blunder(
        eval_before_mover=100.0, eval_after_mover=-150.0, threshold_cp=200.0
    )
    assert is_blunder is True
    assert drop == 250.0

    # Drop 150 cp < 200 -> Inaccuracy, not blunder
    is_blunder, drop = calculate_blunder(
        eval_before_mover=100.0, eval_after_mover=-50.0, threshold_cp=200.0
    )
    assert is_blunder is False
    assert drop == 150.0

    # Position improved -> Not blunder
    is_blunder, drop = calculate_blunder(
        eval_before_mover=-50.0, eval_after_mover=100.0, threshold_cp=200.0
    )
    assert is_blunder is False
    assert drop == -150.0


def test_calculate_blunder_mate_and_hopeless():
    # Throwing away forced mate: from +9980 (mate in 2) to +50 (equal) -> Blunder
    is_blunder, drop = calculate_blunder(
        eval_before_mover=9980.0,
        eval_after_mover=50.0,
        threshold_cp=200.0,
    )
    assert is_blunder is True
    assert drop == 9930.0

    # Mate in 1 to mate in 2: +9990 to +9980 -> drop 10 < 200 -> Not blunder
    is_blunder, drop = calculate_blunder(
        eval_before_mover=9990.0,
        eval_after_mover=9980.0,
        threshold_cp=200.0,
    )
    assert is_blunder is False
    assert drop == 10.0

    # Dead lost position: eval_before = -1200 cp.
    # When ignore_dead_lost=True, drops in already hopeless positions are ignored
    is_blunder, drop = calculate_blunder(
        eval_before_mover=-1200.0,
        eval_after_mover=-2000.0,
        threshold_cp=200.0,
        dead_lost_threshold_cp=-800.0,
        ignore_dead_lost=True,
    )
    assert is_blunder is False

    # When ignore_dead_lost=False, strict 200 drop is applied
    is_blunder, drop = calculate_blunder(
        eval_before_mover=-1200.0,
        eval_after_mover=-2000.0,
        threshold_cp=200.0,
        dead_lost_threshold_cp=-800.0,
        ignore_dead_lost=False,
    )
    assert is_blunder is True


def test_extract_clock_seconds():
    # Node clock method
    node_with_method = DummyNode(clk=180.5)
    assert extract_clock_seconds(node_with_method) == 180.5

    # Node comment format: [%clk 0:05:00]
    node_hms = DummyNode(comment="Good move [%clk 0:05:00] [%eval 0.2]")
    assert extract_clock_seconds(node_hms) == 300.0

    # Node comment format: [%clk 0:02:45.5]
    node_fraction = DummyNode(comment="[%clk 0:02:45.5]")
    assert extract_clock_seconds(node_fraction) == 165.5

    # Node comment format: [%clk 5:00]
    node_ms = DummyNode(comment="[%clk 5:30]")
    assert extract_clock_seconds(node_ms) == 330.0

    # No clock
    node_none = DummyNode(comment="Just a comment")
    assert extract_clock_seconds(node_none) is None


def test_get_rating_band():
    assert get_rating_band(1000) == "low"
    assert get_rating_band(1199) == "low"
    assert get_rating_band(1200) == "mid"
    assert get_rating_band(1500) == "mid"
    assert get_rating_band(1800) == "mid"
    assert get_rating_band(1801) == "high"
    assert get_rating_band(2300) == "high"


def test_centipawns_to_win_prob():
    # 0 centipawns is exactly 50%
    assert abs(centipawns_to_win_prob(0.0) - 0.5) < 1e-5

    # +400 cp is 1 / (1 + 10^-1) = 1 / 1.1 ≈ 0.90909
    p_400 = centipawns_to_win_prob(400.0)
    assert abs(p_400 - (1.0 / 1.1)) < 1e-4

    # -400 cp is 1 / (1 + 10^1) = 1 / 11 ≈ 0.09091
    p_minus_400 = centipawns_to_win_prob(-400.0)
    assert abs(p_minus_400 - (1.0 / 11.0)) < 1e-4

    # High mate score saturates near 1.0
    assert centipawns_to_win_prob(9980.0) > 0.9999
    # Hopeless mate score saturates near 0.0
    assert centipawns_to_win_prob(-9980.0) < 0.0001


def test_calculate_blunder_win_prob_standard():
    # Drop 0.20 >= threshold 0.15 -> blunder
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.60, win_prob_after=0.40, threshold=0.15
    )
    assert is_blunder is True
    assert abs(drop - 0.20) < 1e-5

    # Small drop 0.10 < 0.15 -> not blunder
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.55, win_prob_after=0.45, threshold=0.15
    )
    assert is_blunder is False
    assert abs(drop - 0.10) < 1e-5

    # Position improved -> not blunder
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.40, win_prob_after=0.60, threshold=0.15
    )
    assert is_blunder is False
    assert drop < 0


def test_calculate_blunder_win_prob_clearly_winning_edge_cases():
    # EDGE CASE 1: Drop starts and ends in clearly winning position (>= 0.80)
    # E.g. starts at 0.99 (overwhelmingly winning) and drops to 0.82 (still winning)
    # Drop is 0.17 >= 0.15, but player is STILL clearly winning -> NOT a blunder!
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.99,
        win_prob_after=0.82,
        threshold=0.15,
        clearly_winning_threshold=0.80,
    )
    assert is_blunder is False
    assert abs(drop - 0.17) < 1e-5

    # EDGE CASE 2: Drop starts clearly winning (>= 0.80) but ends below (e.g. 0.65)
    # Threw away the clear win into an unclear game -> IS a blunder!
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.95,
        win_prob_after=0.65,
        threshold=0.15,
        clearly_winning_threshold=0.80,
    )
    assert is_blunder is True
    assert abs(drop - 0.30) < 1e-5


def test_calculate_blunder_win_prob_dead_lost_edge_case():
    # Hopeless position: starts at 0.015 (below 0.02) and drops to 0.001
    # When ignore_dead_lost is True -> NOT a blunder
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.015,
        win_prob_after=0.001,
        threshold=0.15,
        dead_lost_threshold_prob=0.02,
        ignore_dead_lost=True,
    )
    assert is_blunder is False

    # When ignore_dead_lost is False and drop exceeds threshold
    is_blunder, drop = calculate_blunder_win_prob(
        win_prob_before=0.20,
        win_prob_after=0.01,
        threshold=0.15,
        dead_lost_threshold_prob=0.02,
        ignore_dead_lost=False,
    )
    assert is_blunder is True

