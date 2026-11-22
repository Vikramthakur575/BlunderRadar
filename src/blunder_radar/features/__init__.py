"""
Feature engineering module for BlunderRadar.
Strictly pre-move chess representations, board geometry, and temporal context features.
"""

from blunder_radar.features.chess_features import (
    CHESS_FEATURE_NAMES,
    extract_chess_features_from_fen,
)
from blunder_radar.features.context_features import (
    CONTEXT_FEATURE_NAMES,
    add_pre_move_clock_columns,
    extract_context_features_vectorized,
)
from blunder_radar.features.pipeline import FeaturePipeline, extract_all_features

__all__ = [
    "CHESS_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "extract_chess_features_from_fen",
    "extract_context_features_vectorized",
    "add_pre_move_clock_columns",
    "extract_all_features",
    "FeaturePipeline",
]
