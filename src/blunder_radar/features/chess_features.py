"""
Chess-specific board features extracted strictly from pre-move FEN positions.
Guarantees complete isolation from post-move information (no move, eval_after, etc.).
"""

from __future__ import annotations

from typing import Any

import chess

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

CHESS_FEATURE_NAMES: list[str] = [
    # Material & Piece Counts
    "mover_pawns",
    "mover_knights",
    "mover_bishops",
    "mover_rooks",
    "mover_queens",
    "opp_pawns",
    "opp_knights",
    "opp_bishops",
    "opp_rooks",
    "opp_queens",
    "mover_piece_count",
    "opp_piece_count",
    "total_piece_count",
    "mover_material",
    "opp_material",
    "material_diff",
    "mover_has_bishop_pair",
    "opp_has_bishop_pair",
    # Mobility
    "mover_mobility",
    "opp_mobility",
    "mobility_diff",
    # King Safety
    "mover_king_zone_attackers",
    "mover_king_zone_attacks",
    "opp_king_zone_attackers",
    "opp_king_zone_attacks",
    "mover_pawn_shield",
    "opp_pawn_shield",
    "mover_can_castle_kingside",
    "mover_can_castle_queenside",
    "mover_has_castled",
    "opp_can_castle_kingside",
    "opp_can_castle_queenside",
    "opp_has_castled",
    # Checks
    "is_in_check",
    "num_checkers",
    "can_give_check",
    # Pins
    "mover_pinned_pieces",
    "opp_pinned_pieces",
    # Hanging & Attacked Pieces
    "mover_attacked_pieces",
    "mover_attacked_value",
    "mover_hanging_pieces",
    "mover_hanging_value",
    "opp_attacked_pieces",
    "opp_attacked_value",
    "opp_hanging_pieces",
    "opp_hanging_value",
    # Board Tension
    "board_tension_piece_attacks",
    "board_tension_pawn_clashes",
    "board_tension_total_attacked_value",
    # Game Phase
    "game_phase_score",
    "is_opening",
    "is_endgame",
]


def extract_chess_features_from_board(board: chess.Board) -> dict[str, Any]:
    """
    Extract all chess domain features from an initialized chess.Board object.
    Strictly pre-move: only inspects the current position on the board.
    """
    mover_color = board.turn
    opp_color = not mover_color

    # 1. Piece counts
    mover_pawns = len(board.pieces(chess.PAWN, mover_color))
    mover_knights = len(board.pieces(chess.KNIGHT, mover_color))
    mover_bishops = len(board.pieces(chess.BISHOP, mover_color))
    mover_rooks = len(board.pieces(chess.ROOK, mover_color))
    mover_queens = len(board.pieces(chess.QUEEN, mover_color))

    opp_pawns = len(board.pieces(chess.PAWN, opp_color))
    opp_knights = len(board.pieces(chess.KNIGHT, opp_color))
    opp_bishops = len(board.pieces(chess.BISHOP, opp_color))
    opp_rooks = len(board.pieces(chess.ROOK, opp_color))
    opp_queens = len(board.pieces(chess.QUEEN, opp_color))

    mover_piece_count = (
        mover_pawns + mover_knights + mover_bishops + mover_rooks + mover_queens
    )
    opp_piece_count = opp_pawns + opp_knights + opp_bishops + opp_rooks + opp_queens
    total_piece_count = mover_piece_count + opp_piece_count

    mover_material = (
        mover_pawns * 1
        + mover_knights * 3
        + mover_bishops * 3
        + mover_rooks * 5
        + mover_queens * 9
    )
    opp_material = (
        opp_pawns * 1 + opp_knights * 3 + opp_bishops * 3 + opp_rooks * 5 + opp_queens * 9
    )
    material_diff = mover_material - opp_material

    mover_has_bishop_pair = int(mover_bishops >= 2)
    opp_has_bishop_pair = int(opp_bishops >= 2)

    # 2. Mobility
    mover_mobility = board.legal_moves.count()
    # Temporarily switch turn to calculate opponent's pre-move mobility
    board.turn = opp_color
    opp_mobility = board.legal_moves.count()
    board.turn = mover_color
    mobility_diff = mover_mobility - opp_mobility

    # 3. King Safety
    mover_king_sq = board.king(mover_color)
    opp_king_sq = board.king(opp_color)

    mover_king_zone_attackers = 0
    mover_king_zone_attacks = 0
    mover_pawn_shield = 0

    if mover_king_sq is not None:
        mover_king_zone = chess.SquareSet(
            chess.BB_KING_ATTACKS[mover_king_sq] | (1 << mover_king_sq)
        )
        zone_attackers: set[chess.Square] = set()
        for sq in mover_king_zone:
            atts = board.attackers(opp_color, sq)
            zone_attackers.update(atts)
            mover_king_zone_attacks += len(atts)
        mover_king_zone_attackers = len(zone_attackers)

        kf = chess.square_file(mover_king_sq)
        kr = chess.square_rank(mover_king_sq)
        forward = 1 if mover_color == chess.WHITE else -1
        shield_squares = [
            chess.square(f, r)
            for f in (kf - 1, kf, kf + 1)
            if 0 <= f <= 7
            for r in (kr + forward, kr + 2 * forward)
            if 0 <= r <= 7
        ]
        mover_pawn_shield = sum(
            1
            for sq in shield_squares
            if board.piece_at(sq) == chess.Piece(chess.PAWN, mover_color)
        )

    opp_king_zone_attackers = 0
    opp_king_zone_attacks = 0
    opp_pawn_shield = 0

    if opp_king_sq is not None:
        opp_king_zone = chess.SquareSet(
            chess.BB_KING_ATTACKS[opp_king_sq] | (1 << opp_king_sq)
        )
        opp_zone_attackers: set[chess.Square] = set()
        for sq in opp_king_zone:
            atts = board.attackers(mover_color, sq)
            opp_zone_attackers.update(atts)
            opp_king_zone_attacks += len(atts)
        opp_king_zone_attackers = len(opp_zone_attackers)

        okf = chess.square_file(opp_king_sq)
        okr = chess.square_rank(opp_king_sq)
        opp_forward = 1 if opp_color == chess.WHITE else -1
        opp_shield_squares = [
            chess.square(f, r)
            for f in (okf - 1, okf, okf + 1)
            if 0 <= f <= 7
            for r in (okr + opp_forward, okr + 2 * opp_forward)
            if 0 <= r <= 7
        ]
        opp_pawn_shield = sum(
            1
            for sq in opp_shield_squares
            if board.piece_at(sq) == chess.Piece(chess.PAWN, opp_color)
        )

    mover_can_castle_kingside = int(board.has_kingside_castling_rights(mover_color))
    mover_can_castle_queenside = int(board.has_queenside_castling_rights(mover_color))
    opp_can_castle_kingside = int(board.has_kingside_castling_rights(opp_color))
    opp_can_castle_queenside = int(board.has_queenside_castling_rights(opp_color))

    # Castled status: King on castled squares (g1/c1 or g8/c8) without remaining castling rights
    mover_castled_squares = (
        (chess.G1, chess.C1) if mover_color == chess.WHITE else (chess.G8, chess.C8)
    )
    mover_has_castled = int(
        (mover_king_sq in mover_castled_squares)
        and not (mover_can_castle_kingside or mover_can_castle_queenside)
    )

    opp_castled_squares = (
        (chess.G1, chess.C1) if opp_color == chess.WHITE else (chess.G8, chess.C8)
    )
    opp_has_castled = int(
        (opp_king_sq in opp_castled_squares)
        and not (opp_can_castle_kingside or opp_can_castle_queenside)
    )

    # 4. Checks
    is_in_check = int(board.is_check())
    num_checkers = len(board.checkers())
    can_give_check = int(any(board.gives_check(m) for m in board.legal_moves))

    # 5. Pins
    mover_pinned_pieces = sum(
        1
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for sq in board.pieces(pt, mover_color)
        if board.is_pinned(mover_color, sq)
    )
    opp_pinned_pieces = sum(
        1
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for sq in board.pieces(pt, opp_color)
        if board.is_pinned(opp_color, sq)
    )

    # 6. Attacked and Hanging Pieces
    mover_attacked_pieces = 0
    mover_attacked_value = 0
    mover_hanging_pieces = 0
    mover_hanging_value = 0

    for pt, val in PIECE_VALUES.items():
        for sq in board.pieces(pt, mover_color):
            opp_atts = board.attackers(opp_color, sq)
            if opp_atts:
                mover_attacked_pieces += 1
                mover_attacked_value += val
                friendly_defs = board.attackers(mover_color, sq)
                if not friendly_defs:
                    mover_hanging_pieces += 1
                    mover_hanging_value += val

    opp_attacked_pieces = 0
    opp_attacked_value = 0
    opp_hanging_pieces = 0
    opp_hanging_value = 0

    for pt, val in PIECE_VALUES.items():
        for sq in board.pieces(pt, opp_color):
            mover_atts = board.attackers(mover_color, sq)
            if mover_atts:
                opp_attacked_pieces += 1
                opp_attacked_value += val
                friendly_defs = board.attackers(opp_color, sq)
                if not friendly_defs:
                    opp_hanging_pieces += 1
                    opp_hanging_value += val

    # 7. Board Tension
    board_tension_piece_attacks = mover_attacked_pieces + opp_attacked_pieces
    board_tension_total_attacked_value = mover_attacked_value + opp_attacked_value

    board_tension_pawn_clashes = sum(
        1
        for sq in board.pieces(chess.PAWN, mover_color)
        if any(
            board.piece_at(att) == chess.Piece(chess.PAWN, opp_color)
            for att in board.attackers(opp_color, sq)
        )
    )

    # 8. Game Phase
    # Classic Fruit / Stockfish phase formula (max = 24)
    phase = (
        (mover_knights + opp_knights) * 1
        + (mover_bishops + opp_bishops) * 1
        + (mover_rooks + opp_rooks) * 2
        + (mover_queens + opp_queens) * 4
    )
    game_phase_score = round(min(max(phase / 24.0, 0.0), 1.0), 4)
    is_opening = int(game_phase_score >= 0.8)
    is_endgame = int(game_phase_score <= 0.3)

    return {
        "mover_pawns": mover_pawns,
        "mover_knights": mover_knights,
        "mover_bishops": mover_bishops,
        "mover_rooks": mover_rooks,
        "mover_queens": mover_queens,
        "opp_pawns": opp_pawns,
        "opp_knights": opp_knights,
        "opp_bishops": opp_bishops,
        "opp_rooks": opp_rooks,
        "opp_queens": opp_queens,
        "mover_piece_count": mover_piece_count,
        "opp_piece_count": opp_piece_count,
        "total_piece_count": total_piece_count,
        "mover_material": mover_material,
        "opp_material": opp_material,
        "material_diff": material_diff,
        "mover_has_bishop_pair": mover_has_bishop_pair,
        "opp_has_bishop_pair": opp_has_bishop_pair,
        "mover_mobility": mover_mobility,
        "opp_mobility": opp_mobility,
        "mobility_diff": mobility_diff,
        "mover_king_zone_attackers": mover_king_zone_attackers,
        "mover_king_zone_attacks": mover_king_zone_attacks,
        "opp_king_zone_attackers": opp_king_zone_attackers,
        "opp_king_zone_attacks": opp_king_zone_attacks,
        "mover_pawn_shield": mover_pawn_shield,
        "opp_pawn_shield": opp_pawn_shield,
        "mover_can_castle_kingside": mover_can_castle_kingside,
        "mover_can_castle_queenside": mover_can_castle_queenside,
        "mover_has_castled": mover_has_castled,
        "opp_can_castle_kingside": opp_can_castle_kingside,
        "opp_can_castle_queenside": opp_can_castle_queenside,
        "opp_has_castled": opp_has_castled,
        "is_in_check": is_in_check,
        "num_checkers": num_checkers,
        "can_give_check": can_give_check,
        "mover_pinned_pieces": mover_pinned_pieces,
        "opp_pinned_pieces": opp_pinned_pieces,
        "mover_attacked_pieces": mover_attacked_pieces,
        "mover_attacked_value": mover_attacked_value,
        "mover_hanging_pieces": mover_hanging_pieces,
        "mover_hanging_value": mover_hanging_value,
        "opp_attacked_pieces": opp_attacked_pieces,
        "opp_attacked_value": opp_attacked_value,
        "opp_hanging_pieces": opp_hanging_pieces,
        "opp_hanging_value": opp_hanging_value,
        "board_tension_piece_attacks": board_tension_piece_attacks,
        "board_tension_pawn_clashes": board_tension_pawn_clashes,
        "board_tension_total_attacked_value": board_tension_total_attacked_value,
        "game_phase_score": game_phase_score,
        "is_opening": is_opening,
        "is_endgame": is_endgame,
    }


def extract_chess_features_from_fen(fen: str) -> dict[str, Any]:
    """
    Extract chess features directly from a FEN string.
    """
    board = chess.Board(fen)
    return extract_chess_features_from_board(board)


def batch_extract_chess_features(fens: list[str]) -> list[dict[str, Any]]:
    """
    Batch extract chess features from a list of FEN strings.
    """
    return [extract_chess_features_from_fen(fen) for fen in fens]
