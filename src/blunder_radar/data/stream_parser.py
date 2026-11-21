"""
Streaming PGN parser for Lichess databases with Stockfish evaluations.
Decompresses incrementally without loading archives into memory.
"""

from __future__ import annotations

import gzip
import io
import re
import urllib.request
from collections.abc import Generator, Iterable
from dataclasses import asdict, dataclass
from typing import BinaryIO, TextIO

import chess
import chess.pgn
import zstandard

from blunder_radar.config import DataConfig
from blunder_radar.data.eval_utils import (
    calculate_blunder,
    calculate_blunder_win_prob,
    centipawns_to_win_prob,
    convert_to_mover_perspective,
    extract_clock_seconds,
    extract_eval_cp,
    get_rating_band,
)

GAME_ID_PATTERN = re.compile(r"lichess\.org/([a-zA-Z0-9]{8,12})")


@dataclass
class MoveRecord:
    game_id: str
    game_idx: int
    game_date: str
    game_timestamp: str
    speed: str
    ply: int
    fen_before: str
    move: str
    mover_color: str
    mover_rating: int
    opponent_rating: int
    rating_diff: int
    rating_band: str
    clock_remaining_seconds: float | None
    increment_seconds: float
    eval_before: float
    eval_after: float
    eval_drop: float
    is_blunder: int
    win_prob_before: float
    win_prob_after: float
    win_prob_drop: float
    is_blunder_prob: int

    def to_dict(self) -> dict:
        return asdict(self)


def is_speed_allowed(
    event: str,
    time_control: str,
    allowed_speeds: Iterable[str] = ("blitz", "rapid"),
) -> str | None:
    """
    Determine if a game matches the allowed speeds (blitz/rapid).
    Returns the speed name if allowed, otherwise None.
    """
    event_lower = event.lower()

    # 1. Check Event header
    for speed in allowed_speeds:
        if speed in event_lower:
            return speed

    # 2. Check TimeControl header if Event was generic
    if time_control and "+" in time_control:
        parts = time_control.split("+")
        try:
            base = int(parts[0])
            inc = int(parts[1])
            duration = base + 40 * inc
            if 180 <= duration < 480 and "blitz" in allowed_speeds:
                return "blitz"
            elif 480 <= duration < 1500 and "rapid" in allowed_speeds:
                return "rapid"
        except (ValueError, IndexError):
            pass

    return None


def extract_game_id(site_header: str, fallback_idx: int) -> str:
    """Extract 8-12 char game id from Lichess Site URL or generate fallback."""
    if site_header:
        match = GAME_ID_PATTERN.search(site_header)
        if match:
            return match.group(1)
    return f"game_{fallback_idx:08d}"


def extract_increment_seconds(time_control: str) -> float:
    """Extract increment in seconds from TimeControl header (e.g. '300+3' -> 3.0)."""
    if time_control and "+" in time_control:
        try:
            return float(time_control.split("+")[1])
        except (ValueError, IndexError):
            pass
    return 0.0


def open_stream(
    source: str | BinaryIO,
    user_agent: str = "BlunderRadar/1.0",
) -> Generator[TextIO, None, None]:
    """
    Open a file path, URL, or stream incrementally and yield a TextIO stream.
    Supports .zst, .gz, or uncompressed PGN.
    """
    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            req = urllib.request.Request(source, headers={"User-Agent": user_agent})
            raw_resp = urllib.request.urlopen(req, timeout=30)
            if source.endswith(".zst"):
                dctx = zstandard.ZstdDecompressor()
                reader = dctx.stream_reader(raw_resp)
                yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            elif source.endswith(".gz"):
                gz = gzip.GzipFile(fileobj=raw_resp, mode="rb")
                yield io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
            else:
                yield io.TextIOWrapper(raw_resp, encoding="utf-8", errors="replace")
            raw_resp.close()
            return

        # Local file
        if source.endswith(".zst"):
            with open(source, "rb") as fh:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        elif source.endswith(".gz"):
            with gzip.open(source, "rt", encoding="utf-8", errors="replace") as fh:
                yield fh
        else:
            with open(source, encoding="utf-8", errors="replace") as fh:
                yield fh
    else:
        # File-like object
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(source) as reader:
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def parse_game_to_moves(
    game: chess.pgn.Game,
    game_idx: int,
    config: DataConfig,
) -> Generator[MoveRecord, None, None]:
    """
    Extract per-move MoveRecords from a single evaluated chess game.
    """
    headers = game.headers
    white_elo_str = headers.get("WhiteElo", "")
    black_elo_str = headers.get("BlackElo", "")
    event = headers.get("Event", "")
    time_control = headers.get("TimeControl", "")
    site = headers.get("Site", "")
    utc_date = headers.get("UTCDate", headers.get("Date", "")).strip()
    utc_time = headers.get("UTCTime", headers.get("Time", "")).strip()
    game_timestamp = f"{utc_date} {utc_time}".strip() if (utc_date or utc_time) else "unknown"
    game_date = utc_date if utc_date else headers.get("Date", "unknown")
    increment_seconds = extract_increment_seconds(time_control)

    try:
        white_elo = int(white_elo_str)
        black_elo = int(black_elo_str)
    except (ValueError, TypeError):
        return

    # Check rating range
    if not (
        config.min_rating <= white_elo <= config.max_rating
        and config.min_rating <= black_elo <= config.max_rating
    ):
        return

    # Check speed
    speed = is_speed_allowed(event, time_control, config.allowed_speeds)
    if speed is None:
        return

    game_id = extract_game_id(site, game_idx)

    # Initial position eval before move 1 is roughly equal (+20 cp White advantage)
    board = game.board()
    eval_before_white = 20.0
    node = game
    ply = 0

    while node.variations:
        next_node = node.variation(0)
        ply += 1
        turn = board.turn
        mover_color = "white" if turn == chess.WHITE else "black"
        mover_rating = white_elo if turn == chess.WHITE else black_elo
        opponent_rating = black_elo if turn == chess.WHITE else white_elo
        rating_diff = mover_rating - opponent_rating
        rating_band = get_rating_band(mover_rating)

        fen_before = board.fen()
        move_str = next_node.move.uci()
        clock_remaining = extract_clock_seconds(next_node)
        white_eval_after = extract_eval_cp(next_node, mate_score_cp=config.mate_score_cp)

        if white_eval_after is not None:
            eval_before_mover = convert_to_mover_perspective(eval_before_white, turn)
            eval_after_mover = convert_to_mover_perspective(white_eval_after, turn)
            is_blunder, eval_drop = calculate_blunder(
                eval_before_mover=eval_before_mover,
                eval_after_mover=eval_after_mover,
                threshold_cp=config.blunder_threshold_cp,
                dead_lost_threshold_cp=config.dead_lost_threshold_cp,
                ignore_dead_lost=config.ignore_dead_lost,
            )

            win_prob_before = centipawns_to_win_prob(
                eval_before_mover, scaling=config.win_prob_scaling
            )
            win_prob_after = centipawns_to_win_prob(
                eval_after_mover, scaling=config.win_prob_scaling
            )
            is_blunder_prob, win_prob_drop = calculate_blunder_win_prob(
                win_prob_before=win_prob_before,
                win_prob_after=win_prob_after,
                threshold=config.win_prob_blunder_threshold,
                clearly_winning_threshold=config.clearly_winning_threshold,
                dead_lost_threshold_prob=config.dead_lost_threshold_prob,
                ignore_dead_lost=config.ignore_dead_lost,
            )

            yield MoveRecord(
                game_id=game_id,
                game_idx=game_idx,
                game_date=game_date,
                game_timestamp=game_timestamp,
                speed=speed,
                ply=ply,
                fen_before=fen_before,
                move=move_str,
                mover_color=mover_color,
                mover_rating=mover_rating,
                opponent_rating=opponent_rating,
                rating_diff=rating_diff,
                rating_band=rating_band,
                clock_remaining_seconds=clock_remaining,
                increment_seconds=increment_seconds,
                eval_before=eval_before_mover,
                eval_after=eval_after_mover,
                eval_drop=eval_drop,
                is_blunder=int(is_blunder),
                win_prob_before=win_prob_before,
                win_prob_after=win_prob_after,
                win_prob_drop=win_prob_drop,
                is_blunder_prob=int(is_blunder_prob),
            )

            eval_before_white = white_eval_after
        else:
            # If eval missing on a move, preserve previous eval for subsequent positions
            pass

        board.push(next_node.move)
        node = next_node


def stream_pgn_positions(
    source: str | BinaryIO,
    config: DataConfig,
    max_positions: int | None = None,
    skip_games: int = 0,
) -> Generator[MoveRecord, None, None]:
    """
    Stream MoveRecords from PGN source, yielding rows up to max_positions.
    """
    target = max_positions or config.target_positions
    count = 0
    game_idx = 0

    for text_stream in open_stream(source):
        # Rapidly skip games if resuming from checkpoint
        while game_idx < skip_games:
            if not chess.pgn.skip_game(text_stream):
                break
            game_idx += 1

        while count < target:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            game_idx += 1

            for record in parse_game_to_moves(game, game_idx, config):
                yield record
                count += 1
                if count >= target:
                    break
