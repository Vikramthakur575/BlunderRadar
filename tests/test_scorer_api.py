import chess
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from blunder_radar.features.pipeline import extract_all_features
from blunder_radar.serving.api import create_app
from blunder_radar.serving.scorer import BlunderScorer, ply_from_board, validate_board

PGN = """[Event "t"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "300+0"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] }
2... Nc6 { [%clk 0:04:57] } 3. Bb5 { [%clk 0:04:55] } 3... a6 { [%clk 0:04:50] } *
"""


@pytest.fixture(scope="module")
def scorer(tiny_bundle):
    return BlunderScorer(tiny_bundle)


@pytest.fixture(scope="module")
def client(tiny_bundle_path):
    with TestClient(create_app(tiny_bundle_path)) as c:
        yield c


def test_score_contract(scorer):
    r = scorer.score(chess.STARTING_FEN, 1500, 1500, 300)
    assert {"probability", "risk_band", "summary", "factors", "hot_squares",
            "side_to_move"} <= set(r)
    assert 0 <= r["probability"] <= 1
    assert r["side_to_move"] == "white"


def test_ply_from_board():
    assert ply_from_board(chess.Board()) == 1
    b = chess.Board()
    b.push_san("e4")
    assert ply_from_board(b) == 2


@pytest.mark.parametrize("bad", ["not a fen", "8/8/8/8/8/8/8/8 w - - 0 1"])
def test_invalid_positions_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_board(bad)


def test_checkmate_position_is_rejected():
    with pytest.raises(ValueError, match="game over"):
        validate_board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")


def test_no_train_serve_skew(scorer, random_positions):
    """The live feature builder must reproduce the batch training pipeline exactly."""
    row = random_positions.iloc[[30]].copy()
    row["clock_before_seconds"] = row["clock_remaining_seconds"]
    row["prev_move_time_spent"] = 3.0
    batch = extract_all_features(row, include_eval_before=False, num_workers=1)
    board = chess.Board(row["fen_before"].iloc[0])
    live = scorer.build_features(
        board, row["mover_rating"].iloc[0], row["opponent_rating"].iloc[0],
        row["clock_before_seconds"].iloc[0], 0.0, 3.0, None, int(row["ply"].iloc[0]),
    )
    cols = scorer.bundle.feature_names
    pd.testing.assert_frame_equal(
        batch[cols].reset_index(drop=True).astype(float),
        live[cols].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_score_pgn_covers_every_move(scorer):
    moves = scorer.score_pgn(PGN)
    assert len(moves) == 6
    assert [m["ply"] for m in moves] == [1, 2, 3, 4, 5, 6]
    assert moves[0]["san"] == "e4" and moves[0]["mover"] == "white"
    assert all(0 <= m["probability"] <= 1 for m in moves)


def test_score_pgn_rejects_empty(scorer):
    with pytest.raises(ValueError):
        scorer.score_pgn("[Event \"x\"]\n\n*")


def test_api_health_and_model_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/model-info").json()
    assert info["algorithm"] == "xgboost" and info["n_features"] > 50


def test_api_score_ok(client):
    r = client.post("/score", json={"fen": chess.STARTING_FEN, "clock_seconds": 120})
    assert r.status_code == 200
    body = r.json()
    assert len(body["factors"]) == 5 and body["risk_band"]


def test_api_rejects_bad_input(client):
    assert client.post("/score", json={"fen": "garbage"}).status_code == 422
    assert client.post("/score", json={"fen": chess.STARTING_FEN, "mover_rating": 5}).status_code == 422
    assert client.post("/score", json={}).status_code == 422


def test_api_score_pgn(client):
    r = client.post("/score-pgn", json={"pgn": PGN})
    assert r.status_code == 200 and r.json()["n_moves"] == 6


def test_api_degrades_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/score", json={"fen": chess.STARTING_FEN}).status_code == 503
