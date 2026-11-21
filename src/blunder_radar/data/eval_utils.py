"""
Evaluation, centipawn, mate score, and blunder calculation utilities.
"""

from __future__ import annotations

import math
import re

import chess
import chess.pgn

# Regex patterns for Lichess annotations
EVAL_PATTERN = re.compile(r"\[%eval\s+([#-]?\d+(?:\.\d+)?)\]")
CLOCK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+)(?::(\d*(?:\.\d+)?))?\]")


def parse_eval_str(eval_str: str, mate_score_cp: float = 10000.0) -> float | None:
    """
    Parse an evaluation string (e.g. '0.25', '-1.50', '#2', '#-1') into centipawns from White's perspective.
    """
    eval_str = eval_str.strip()
    if not eval_str:
        return None
    try:
        if eval_str.startswith("#"):
            mate_moves = int(eval_str[1:])
            sign = 1.0 if mate_moves > 0 else -1.0
            return sign * (mate_score_cp - 10.0 * min(abs(mate_moves), 500))
        # Value in pawns, e.g. 0.25 -> 25.0 centipawns
        val = float(eval_str)
        return val * 100.0
    except (ValueError, TypeError):
        return None


def extract_eval_cp(node: chess.pgn.GameNode, mate_score_cp: float = 10000.0) -> float | None:
    """
    Extract engine evaluation in centipawns from White's perspective.
    Tries python-chess PovScore first, then falls back to regex on comment.
    """
    # 1. Try node.eval() if present
    try:
        ev = node.eval()
        if ev is not None:
            if ev.is_mate():
                mate_moves = ev.white().mate()
                if mate_moves is not None:
                    sign = 1.0 if mate_moves > 0 else -1.0
                    return sign * (mate_score_cp - 10.0 * min(abs(mate_moves), 500))
            score = ev.white().score(mate_score=int(mate_score_cp))
            if score is not None:
                return float(score)
    except Exception:
        pass

    # 2. Try parsing comment
    comment = node.comment
    if comment:
        match = EVAL_PATTERN.search(comment)
        if match:
            return parse_eval_str(match.group(1), mate_score_cp=mate_score_cp)

    return None


def extract_clock_seconds(node: chess.pgn.GameNode) -> float | None:
    """
    Extract remaining clock time in seconds from node or comment [%clk h:m:s].
    """
    # 1. Try node.clock()
    try:
        clk = node.clock()
        if clk is not None:
            return float(clk)
    except Exception:
        pass

    # 2. Try comment regex
    comment = node.comment
    if comment:
        match = CLOCK_PATTERN.search(comment)
        if match:
            groups = match.groups()
            if groups[2] is not None:
                # Format: h:m:s
                hours = int(groups[0])
                minutes = int(groups[1])
                seconds = float(groups[2])
                return hours * 3600.0 + minutes * 60.0 + seconds
            else:
                # Format: m:s
                minutes = int(groups[0])
                seconds = float(groups[1])
                return minutes * 60.0 + seconds

    return None


def convert_to_mover_perspective(white_eval_cp: float, mover_color: chess.Color) -> float:
    """
    Convert centipawn evaluation from White's perspective to mover's perspective.
    """
    return white_eval_cp if mover_color == chess.WHITE else -white_eval_cp


def centipawns_to_win_prob(cp: float, scaling: float = 400.0) -> float:
    """
    Convert centipawns (from mover's perspective) to win probability in [0.0, 1.0]
    using a standard logistic function: W(cp) = 1 / (1 + 10^(-cp / scaling)).
    Clamps exponent to prevent numerical overflow/underflow.
    """
    if math.isnan(cp):
        return 0.5
    # Clamp scaled score to [-10, 10]
    scaled = max(-10.0, min(10.0, cp / scaling))
    return 1.0 / (1.0 + (10.0 ** (-scaled)))


def calculate_blunder(
    eval_before_mover: float,
    eval_after_mover: float,
    threshold_cp: float = 200.0,
    dead_lost_threshold_cp: float | None = -800.0,
    ignore_dead_lost: bool = True,
) -> tuple[bool, float]:
    """
    Calculate whether a move was a blunder and the drop in centipawns from the mover's perspective.

    A move is a blunder if:
    1. eval_drop = eval_before_mover - eval_after_mover >= threshold_cp
    2. Unless ignore_dead_lost is True and eval_before_mover <= dead_lost_threshold_cp
       (i.e., position was already hopelessly lost before the move).

    Returns:
        (is_blunder, eval_drop)
    """
    eval_drop = eval_before_mover - eval_after_mover

    if ignore_dead_lost and dead_lost_threshold_cp is not None:
        if eval_before_mover <= dead_lost_threshold_cp:
            # Already dead lost, does not qualify as turning the game into a loss
            return False, eval_drop

    is_blunder = eval_drop >= threshold_cp
    return is_blunder, eval_drop


def calculate_blunder_win_prob(
    win_prob_before: float,
    win_prob_after: float,
    threshold: float = 0.15,
    clearly_winning_threshold: float = 0.80,
    dead_lost_threshold_prob: float | None = 0.02,
    ignore_dead_lost: bool = True,
) -> tuple[bool, float]:
    """
    Calculate whether a move was a win-probability-based blunder from mover's perspective.

    A move is a win-probability blunder if:
    1. win_prob_drop = win_prob_before - win_prob_after >= threshold
    2. Unless ignore_dead_lost is True and win_prob_before <= dead_lost_threshold_prob
       (i.e., position was already hopeless).
    3. AND ignores drops that start and end in clearly winning positions
       (i.e. win_prob_before >= clearly_winning_threshold and win_prob_after >= clearly_winning_threshold).

    Returns:
        (is_blunder_prob, win_prob_drop)
    """
    win_prob_drop = win_prob_before - win_prob_after

    if ignore_dead_lost and dead_lost_threshold_prob is not None:
        if win_prob_before <= dead_lost_threshold_prob:
            return False, win_prob_drop

    # Ignore drops that start and end in clearly winning positions
    if win_prob_before >= clearly_winning_threshold and win_prob_after >= clearly_winning_threshold:
        return False, win_prob_drop

    is_blunder = win_prob_drop >= threshold
    return is_blunder, win_prob_drop


def get_rating_band(rating: int) -> str:
    """
    Categorize player rating into standard bands:
    - low: < 1200
    - mid: 1200 - 1800
    - high: > 1800
    """
    if rating < 1200:
        return "low"
    elif rating <= 1800:
        return "mid"
    else:
        return "high"

