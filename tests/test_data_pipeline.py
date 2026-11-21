"""
Unit and integration tests for BlunderRadar streaming data pipeline.
"""

import tempfile
from pathlib import Path

import chess.pgn

from blunder_radar.config import DataConfig
from blunder_radar.data.pipeline import (
    PipelineCheckpoint,
    read_dataset_as_dataframe,
    run_pipeline,
    write_batch_to_parquet,
)
from blunder_radar.data.quality_report import save_data_quality_report
from blunder_radar.data.stream_parser import (
    is_speed_allowed,
    parse_game_to_moves,
    stream_pgn_positions,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PGN = FIXTURE_DIR / "sample_eval_games.pgn"
SAMPLE_ZST = FIXTURE_DIR / "sample_eval_games.pgn.zst"


def test_is_speed_allowed():
    assert is_speed_allowed("Rated Blitz game", "300+0") == "blitz"
    assert is_speed_allowed("Rated Rapid game", "600+0") == "rapid"
    assert is_speed_allowed("Rated Classical game", "1800+0") is None
    assert is_speed_allowed("Casual Bullet game", "60+0") is None
    # Derived from TimeControl when Event is generic
    assert is_speed_allowed("Rated game", "300+0") == "blitz"
    assert is_speed_allowed("Rated game", "600+5") == "rapid"


def test_parse_game_to_moves():
    cfg = DataConfig(min_rating=800, max_rating=2400)
    with open(SAMPLE_PGN, encoding="utf-8") as f:
        game1 = chess.pgn.read_game(f)
        records = list(parse_game_to_moves(game1, 1, cfg))

    assert len(records) > 0
    first = records[0]
    assert first.game_id == "testgm01"
    assert first.game_timestamp != ""
    assert first.increment_seconds == 0.0
    assert first.ply == 1
    assert first.mover_color == "white"
    assert first.mover_rating == 1500
    assert first.opponent_rating == 1520
    assert first.rating_band == "mid"
    assert first.move == "e2e4"
    assert first.clock_remaining_seconds == 300.0
    assert first.eval_before == 20.0
    assert first.eval_after == 20.0
    assert first.is_blunder == 0
    assert hasattr(first, "win_prob_before")
    assert hasattr(first, "is_blunder_prob")

    # Game 1 has blunder on ply 30 (15... Qxd4 where eval drops to +3.80 for White)
    blunders = [r for r in records if r.is_blunder == 1]
    assert len(blunders) >= 1
    blunder_move = blunders[0]
    assert blunder_move.eval_drop >= 200.0
    assert any(r.is_blunder_prob == 1 for r in records)


def test_stream_pgn_positions_filters_ratings_and_speeds():
    cfg = DataConfig(min_rating=800, max_rating=2400)
    # SAMPLE_PGN has 5 games:
    # 1. Blitz, 1500 vs 1520 -> included
    # 2. Rapid, 980 vs 1020 -> included
    # 3. Blitz, 2150 vs 2180 -> included
    # 4. Classical, 1600 vs 1620 -> filtered out (speed)
    # 5. Blitz, 650 vs 700 -> filtered out (rating < 800)
    records = list(stream_pgn_positions(str(SAMPLE_PGN), cfg))
    game_ids = {r.game_id for r in records}

    assert "testgm01" in game_ids
    assert "testgm02" in game_ids
    assert "testgm03" in game_ids
    assert "testgm04" not in game_ids
    assert "testgm05" not in game_ids


def test_stream_pgn_positions_from_zst():
    cfg = DataConfig(min_rating=800, max_rating=2400)
    records = list(stream_pgn_positions(str(SAMPLE_ZST), cfg))
    assert len(records) > 0
    assert any(r.is_blunder == 1 for r in records)


def test_parquet_partitioned_write_and_read():
    cfg = DataConfig(min_rating=800, max_rating=2400)
    records = list(stream_pgn_positions(str(SAMPLE_PGN), cfg))

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_parquet = Path(tmp_dir) / "positions.parquet"
        write_batch_to_parquet(records, out_parquet, batch_idx=1, partition_cols=["rating_band"])

        df = read_dataset_as_dataframe(out_parquet)
        assert len(df) == len(records)
        assert set(df["rating_band"].unique()).issubset({"low", "mid", "high"})
        assert "is_blunder" in df.columns
        assert "is_blunder_prob" in df.columns
        assert "game_id" in df.columns
        assert "game_timestamp" in df.columns
        assert "fen_before" in df.columns
        assert "clock_remaining_seconds" in df.columns
        assert "increment_seconds" in df.columns


def test_resumable_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_state.json"
        ckpt = PipelineCheckpoint(ckpt_path)
        assert ckpt.state["positions_ingested"] == 0

        ckpt.state["positions_ingested"] = 150
        ckpt.state["batches_written"] = 3
        ckpt.state["last_game_id"] = "testgm01"
        ckpt.save()

        # Reload
        ckpt_reloaded = PipelineCheckpoint(ckpt_path)
        assert ckpt_reloaded.state["positions_ingested"] == 150
        assert ckpt_reloaded.state["batches_written"] == 3
        assert ckpt_reloaded.state["last_game_id"] == "testgm01"

        ckpt_reloaded.reset()
        assert ckpt_reloaded.state["positions_ingested"] == 0
        assert not ckpt_path.exists()


def test_data_quality_report_generation():
    cfg = DataConfig(min_rating=800, max_rating=2400)
    records = list(stream_pgn_positions(str(SAMPLE_PGN), cfg))

    import pandas as pd

    df = pd.DataFrame([r.to_dict() for r in records])

    with tempfile.TemporaryDirectory() as tmp_dir:
        md_file = Path(tmp_dir) / "report.md"
        json_file = Path(tmp_dir) / "report.json"

        report = save_data_quality_report(df, md_path=md_file, json_path=json_file)

        assert md_file.exists()
        assert json_file.exists()
        assert report["summary"]["total_positions"] == len(records)
        assert report["summary"]["unique_games"] == 3
        assert "is_blunder" in report["null_rates"]
        assert report["class_balance"]["total_blunders"] > 0
        assert "mover_rating" in report["rating_distribution"]


def test_pipeline_end_to_end_local():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = DataConfig(
            output_parquet_dir=str(tmp_path / "positions.parquet"),
            checkpoint_file=str(tmp_path / "checkpoint.json"),
            quality_report_md=str(tmp_path / "report.md"),
            quality_report_json=str(tmp_path / "report.json"),
            batch_size=20,
            target_positions=50,
        )

        df = run_pipeline(config=cfg, source=str(SAMPLE_PGN), target_positions=50)
        assert len(df) == 50
        assert Path(cfg.output_parquet_dir).exists()
        assert Path(cfg.quality_report_md).exists()
        assert Path(cfg.quality_report_json).exists()

        # Verify resumability / idempotency
        df2 = run_pipeline(config=cfg, source=str(SAMPLE_PGN), target_positions=50)
        assert len(df2) == 50
