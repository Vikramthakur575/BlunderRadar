"""Explainability: per-prediction feature contributions, plain-English reasons, heatmaps."""

from blunder_radar.explain.explainer import explain_row, tree_contributions
from blunder_radar.explain.heatmap import risky_squares

__all__ = ["explain_row", "tree_contributions", "risky_squares"]
