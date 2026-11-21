"""
Data quality report generation for parsed chess positions.
Computes null rates, class balance, rating distributions, and saves Markdown and JSON reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute comprehensive data quality metrics from position DataFrame.
    """
    total_rows = len(df)
    unique_games = int(df["game_id"].nunique()) if "game_id" in df.columns else 0

    # 1. Null rates
    null_rates = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_rates[col] = {
            "null_count": null_count,
            "null_pct": round(float(null_count / total_rows * 100), 3) if total_rows > 0 else 0.0,
        }

    # 2. Class balance
    total_blunders = int(df["is_blunder"].sum()) if "is_blunder" in df.columns else 0
    overall_blunder_rate = (
        round(float(total_blunders / total_rows * 100), 2) if total_rows > 0 else 0.0
    )

    total_prob_blunders = (
        int(df["is_blunder_prob"].sum()) if "is_blunder_prob" in df.columns else 0
    )
    overall_prob_blunder_rate = (
        round(float(total_prob_blunders / total_rows * 100), 2) if total_rows > 0 else 0.0
    )

    blunder_by_band = {}
    if "rating_band" in df.columns and "is_blunder" in df.columns:
        for band, group in df.groupby("rating_band"):
            band_total = len(group)
            band_blunders = int(group["is_blunder"].sum())
            band_prob_blunders = (
                int(group["is_blunder_prob"].sum()) if "is_blunder_prob" in group.columns else 0
            )
            blunder_by_band[str(band)] = {
                "total_positions": band_total,
                "blunders": band_blunders,
                "blunder_rate_pct": round(float(band_blunders / band_total * 100), 2)
                if band_total > 0
                else 0.0,
                "prob_blunders": band_prob_blunders,
                "prob_blunder_rate_pct": round(float(band_prob_blunders / band_total * 100), 2)
                if band_total > 0
                else 0.0,
            }

    blunder_by_color = {}
    if "mover_color" in df.columns:
        for color, group in df.groupby("mover_color"):
            c_total = len(group)
            c_blunders = int(group["is_blunder"].sum())
            blunder_by_color[str(color)] = {
                "total_positions": c_total,
                "blunders": c_blunders,
                "blunder_rate_pct": round(float(c_blunders / c_total * 100), 2)
                if c_total > 0
                else 0.0,
            }

    # 3. Rating distribution
    def get_distribution_stats(series: pd.Series) -> dict[str, float]:
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        return {
            "mean": round(float(clean.mean()), 1),
            "std": round(float(clean.std()), 1),
            "min": round(float(clean.min()), 1),
            "p25": round(float(np.percentile(clean, 25)), 1),
            "median": round(float(np.percentile(clean, 50)), 1),
            "p75": round(float(np.percentile(clean, 75)), 1),
            "max": round(float(clean.max()), 1),
        }

    mover_rating_stats = (
        get_distribution_stats(df["mover_rating"]) if "mover_rating" in df.columns else {}
    )
    opponent_rating_stats = (
        get_distribution_stats(df["opponent_rating"]) if "opponent_rating" in df.columns else {}
    )

    # 4. Clock stats
    clock_stats = (
        get_distribution_stats(df["clock_remaining_seconds"])
        if "clock_remaining_seconds" in df.columns
        else {}
    )

    # 5. Eval and drop stats
    eval_drop_stats = get_distribution_stats(df["eval_drop"]) if "eval_drop" in df.columns else {}

    report = {
        "summary": {
            "total_positions": total_rows,
            "unique_games": unique_games,
            "avg_positions_per_game": round(float(total_rows / unique_games), 1)
            if unique_games > 0
            else 0.0,
        },
        "null_rates": null_rates,
        "class_balance": {
            "total_blunders": total_blunders,
            "total_non_blunders": total_rows - total_blunders,
            "overall_blunder_rate_pct": overall_blunder_rate,
            "total_prob_blunders": total_prob_blunders,
            "overall_prob_blunder_rate_pct": overall_prob_blunder_rate,
            "by_rating_band": blunder_by_band,
            "by_color": blunder_by_color,
        },
        "rating_distribution": {
            "mover_rating": mover_rating_stats,
            "opponent_rating": opponent_rating_stats,
        },
        "clock_distribution": clock_stats,
        "eval_drop_distribution": eval_drop_stats,
    }
    return report


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format the report dictionary into structured GitHub-flavored Markdown."""
    summary = report["summary"]
    cb = report["class_balance"]
    mover_r = report["rating_distribution"]["mover_rating"]
    clock = report.get("clock_distribution", {})

    md = []
    md.append("# BlunderRadar: Data Quality & Ingestion Report\n")
    md.append("## 1. Executive Summary\n")
    md.append(f"- **Total Positions Ingested:** {summary['total_positions']:,}")
    md.append(f"- **Unique Games:** {summary['unique_games']:,}")
    md.append(f"- **Average Plies per Game:** {summary['avg_positions_per_game']}")
    md.append(f"- **Overall CP Blunder Rate:** {cb['overall_blunder_rate_pct']}%")
    if "overall_prob_blunder_rate_pct" in cb:
        md.append(f"- **Overall Win-Prob Blunder Rate:** {cb['overall_prob_blunder_rate_pct']}%\n")
    else:
        md.append("\n")

    md.append("## 2. Class Balance by Rating Band\n")
    md.append("| Rating Band | Positions | CP Blunders | CP Rate (%) | Prob Blunders | Prob Rate (%) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for band, stats in cb.get("by_rating_band", {}).items():
        prob_b = stats.get("prob_blunders", 0)
        prob_r = stats.get("prob_blunder_rate_pct", 0.0)
        md.append(
            f"| **{band.upper()}** | {stats['total_positions']:,} | {stats['blunders']:,} | {stats['blunder_rate_pct']}% | {prob_b:,} | {prob_r}% |"
        )
    md.append("\n")

    md.append("## 3. Rating Distribution (Mover)\n")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    for k, v in mover_r.items():
        md.append(f"| {k.upper()} | {v} |")
    md.append("\n")

    md.append("## 4. Clock & Feature Null Rates\n")
    md.append("| Column | Null Count | Null Rate (%) |")
    md.append("| :--- | :--- | :--- |")
    for col, nstats in report["null_rates"].items():
        md.append(f"| `{col}` | {nstats['null_count']:,} | {nstats['null_pct']}% |")
    md.append("\n")

    if clock:
        md.append("## 5. Remaining Clock Time Summary (Seconds)\n")
        md.append("| Metric | Clock Seconds |")
        md.append("| :--- | :--- |")
        for k, v in clock.items():
            md.append(f"| {k.upper()} | {v}s |")
        md.append("\n")

    return "\n".join(md)


def save_data_quality_report(
    df: pd.DataFrame,
    md_path: str | Path = "data/data_quality_report.md",
    json_path: str | Path = "data/data_quality_report.json",
) -> dict[str, Any]:
    """Compute and save both Markdown and JSON quality reports."""
    report = generate_data_quality_report(df)

    p_json = Path(json_path)
    p_json.parent.mkdir(parents=True, exist_ok=True)
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    p_md = Path(md_path)
    p_md.parent.mkdir(parents=True, exist_ok=True)
    with open(p_md, "w", encoding="utf-8") as f:
        f.write(format_markdown_report(report))

    return report
