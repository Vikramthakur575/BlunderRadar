"""FastAPI service for BlunderRadar."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from blunder_radar.serving.scorer import DEFAULT_MODEL_PATH, BlunderScorer


class ScoreRequest(BaseModel):
    fen: str = Field(..., description="Position BEFORE the move, FEN notation")
    mover_rating: int = Field(1500, ge=100, le=3500)
    opponent_rating: int = Field(1500, ge=100, le=3500)
    clock_seconds: float | None = Field(
        None, ge=0, le=86400, description="Mover's remaining time when their turn began"
    )
    increment_seconds: float = Field(0.0, ge=0, le=600)
    prev_move_time_spent: float | None = Field(None, ge=0, le=86400)
    eval_before_cp: float | None = Field(
        None, description="Optional engine eval (centipawns, mover's view). Uses the eval model."
    )
    ply: int | None = Field(None, ge=1, le=1000)
    top_k: int = Field(5, ge=1, le=15)


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str
    text: str


class ScoreResponse(BaseModel):
    probability: float
    risk_band: str
    base_rate: float
    relative_risk: float | None
    summary: str
    factors: list[Factor]
    hot_squares: dict[str, float]
    side_to_move: str
    model: dict[str, Any]


class PgnRequest(BaseModel):
    pgn: str = Field(..., min_length=5, max_length=200_000)
    default_rating: int = Field(1500, ge=100, le=3500)
    top_k: int = Field(4, ge=1, le=10)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("BLUNDER_RADAR_MODEL", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = None
        app.state.load_error = None
        try:
            app.state.scorer = BlunderScorer.from_paths(path)
        except Exception as exc:  # service stays up and reports the problem
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="BlunderRadar API",
        version="1.0.0",
        description="Probability that the side to move blunders, with explanations.",
        lifespan=lifespan,
    )

    def scorer_or_503() -> BlunderScorer:
        if app.state.scorer is None:
            raise HTTPException(503, detail=f"Model not loaded: {app.state.load_error}")
        return app.state.scorer

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if app.state.scorer is not None else "degraded",
            "model_loaded": app.state.scorer is not None,
            "error": app.state.load_error,
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        sc = scorer_or_503()
        b = sc.bundle
        return {
            "algorithm": b.algorithm,
            "uses_engine_eval": b.with_eval,
            "n_features": len(b.feature_names),
            "base_rate": b.base_rate,
            "risk_thresholds": b.risk_thresholds,
            "calibration": b.meta.get("calibration"),
            "trained_at": b.meta.get("trained_at"),
            "test_metrics": {
                k.replace("test_cal_", ""): v
                for k, v in b.metrics.items()
                if k.startswith("test_cal_")
            },
            "top_features": b.meta.get("top_features", []),
            "eval_model_available": sc.eval_bundle is not None,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> dict[str, Any]:
        sc = scorer_or_503()
        try:
            return sc.score(
                req.fen, req.mover_rating, req.opponent_rating, req.clock_seconds,
                req.increment_seconds, req.prev_move_time_spent, req.eval_before_cp,
                req.ply, req.top_k,
            )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.post("/score-pgn")
    def score_pgn(req: PgnRequest) -> dict[str, Any]:
        sc = scorer_or_503()
        try:
            moves = sc.score_pgn(req.pgn, req.default_rating, top_k=req.top_k)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return {"n_moves": len(moves), "moves": moves}

    return app


app = create_app()
