"""Square-level danger weights used to draw the 'danger heatmap' on the board."""

from __future__ import annotations

import chess

_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def risky_squares(board: chess.Board) -> dict[int, float]:
    """
    Map square -> danger weight in (0, 1] for the side to move. Highest weights:
    undefended attacked pieces and the king when in check or under heavy pressure; medium
    for attacked-but-defended pieces, pinned pieces and the squares around the king.
    """
    mover = board.turn
    opp = not mover
    weights: dict[int, float] = {}

    def bump(sq: int, w: float) -> None:
        weights[sq] = max(weights.get(sq, 0.0), min(1.0, w))

    for pt, val in _VALUES.items():
        for sq in board.pieces(pt, mover):
            if board.attackers(opp, sq):
                scale = 0.45 + 0.05 * val
                if not board.attackers(mover, sq):
                    bump(sq, 0.55 + scale * 0.5)
                else:
                    bump(sq, scale)
            if board.is_pinned(mover, sq):
                bump(sq, 0.7)

    king = board.king(mover)
    if king is not None:
        if board.is_check():
            bump(king, 1.0)
        zone = chess.SquareSet(chess.BB_KING_ATTACKS[king])
        attacked_zone = [sq for sq in zone if board.attackers(opp, sq)]
        if attacked_zone:
            bump(king, 0.4 + 0.1 * min(len(attacked_zone), 4))
            for sq in attacked_zone:
                bump(sq, 0.3)
    return weights
