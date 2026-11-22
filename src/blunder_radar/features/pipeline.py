"""
End-to-end feature pipeline for BlunderRadar.
Batches and parallelizes chess feature extraction, merges vectorized context features,
and writes partitioned Parquet datasets.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from blunder_radar.features.chess_features import (
    CHESS_FEATURE_NAMES,
    batch_extract_chess_features,
)
from blunder_radar.features.context_features import (
    CONTEXT_FEATURE_NAMES,
    EVAL_FEATURE_NAMES,
    add_pre_move_clock_columns,
    extract_context_features_vectorized,
    extract_eval_features_vectorized,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("blunder_radar.features")


def _process_fen_chunk(fens: list[str]) -> list[dict[str, Any]]:
    """Top-level picklable worker function for multiprocessing."""
    return batch_extract_chess_features(fens)


def extract_all_features(
    df: pd.DataFrame,
    include_eval_before: bool = True,
    num_workers: int | None = None,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Extract all features (chess, context, and optional eval_before) from a positions DataFrame.

    STRICT PRE-MOVE INFORMATION ONLY:
    Only inputs are fen_before, ply, mover_rating, opponent_rating, clock_remaining_seconds,
    increment_seconds, and (optionally) eval_before / win_prob_before.
    Never uses 'move', 'eval_after', 'eval_drop', or post-move target labels.
    """
    if df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    logger.info("Extracting features for %d positions...", total_rows)
    t0 = time.time()

    # 1. Vectorized Context Features (pre-move clock derived once, before any slicing)
    if "clock_before_seconds" not in df.columns:
        df = add_pre_move_clock_columns(df)
    context_df = extract_context_features_vectorized(df)

    # 2. Optional Pre-Move Eval Features
    eval_df = extract_eval_features_vectorized(df) if include_eval_before else None

    # 3. Parallel Chess Features from FEN
    fens = df["fen_before"].tolist()
    workers = (
        num_workers
        if num_workers is not None
        else min(max(1, (os.cpu_count() or 4) - 1), 8)
    )

    chess_dicts: list[dict[str, Any]] = []

    if workers <= 1 or total_rows <= chunk_size:
        chess_dicts = _process_fen_chunk(fens)
    else:
        chunks = [fens[i : i + chunk_size] for i in range(0, total_rows, chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            chunk_results = list(executor.map(_process_fen_chunk, chunks))
        for res in chunk_results:
            chess_dicts.extend(res)

    chess_df = pd.DataFrame(chess_dicts, index=df.index)

    # 4. Assemble Final Feature DataFrame
    # Base columns: identifiers and target labels (kept for modeling & tracking)
    metadata_cols = ["game_id", "game_timestamp", "rating_band"]
    target_cols = [
        "is_blunder",
        "is_blunder_prob",
        "eval_drop",
        "win_prob_drop",
        "eval_after",
    ]

    parts: list[pd.DataFrame] = []

    # Include existing metadata if available
    available_meta = [c for c in metadata_cols if c in df.columns]
    if available_meta:
        parts.append(df[available_meta].copy())

    # Chess features
    parts.append(chess_df)

    # Context features
    parts.append(context_df)

    # Optional pre-move eval features
    if eval_df is not None:
        parts.append(eval_df)

    # Target labels
    available_targets = [c for c in target_cols if c in df.columns]
    if available_targets:
        parts.append(df[available_targets].copy())

    result_df = pd.concat(parts, axis=1)

    elapsed = time.time() - t0
    rate = total_rows / max(elapsed, 0.001)
    logger.info(
        "Extracted %d features across %d positions in %.2fs (%.1f pos/s).",
        len(CHESS_FEATURE_NAMES)
        + len(CONTEXT_FEATURE_NAMES)
        + (len(EVAL_FEATURE_NAMES) if include_eval_before else 0),
        total_rows,
        elapsed,
        rate,
    )

    return result_df


class FeaturePipeline:
    """Manages reading position Parquet datasets, running feature extraction, and writing output."""

    def __init__(
        self,
        include_eval_before: bool = True,
        batch_size: int = 50000,
        num_workers: int | None = None,
        partition_cols: list[str] = ("rating_band",),
    ):
        self.include_eval_before = include_eval_before
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partition_cols = list(partition_cols)

    def process_and_save(
        self,
        input_parquet_dir: str | Path,
        output_parquet_dir: str | Path,
    ) -> dict[str, Any]:
        """
        Process the entire input Parquet dataset and save extracted features.
        """
        in_path = Path(input_parquet_dir)
        out_path = Path(output_parquet_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"Input dataset not found at {in_path}")

        logger.info("Reading input dataset from %s...", in_path)
        start_time = time.time()

        dataset = ds.dataset(str(in_path), format="parquet", partitioning="hive")
        table = dataset.to_table()
        df = table.to_pandas()

        total_rows = len(df)
        logger.info("Loaded %d positions. Running feature extraction...", total_rows)

        # Derive pre-move clocks on the full dataset so games are never split across slices
        df = df.sort_values(["game_id", "ply"], kind="stable").reset_index(drop=True)
        df = add_pre_move_clock_columns(df)

        # Process in batches to control memory if dataset is very large
        features_batches: list[pd.DataFrame] = []
        if total_rows > self.batch_size:
            for i in range(0, total_rows, self.batch_size):
                sub_df = df.iloc[i : i + self.batch_size]
                logger.info(
                    "Processing slice %d to %d / %d...",
                    i,
                    min(i + self.batch_size, total_rows),
                    total_rows,
                )
                feat_sub = extract_all_features(
                    sub_df,
                    include_eval_before=self.include_eval_before,
                    num_workers=self.num_workers,
                )
                features_batches.append(feat_sub)
            df_features = pd.concat(features_batches, axis=0, ignore_index=True)
        else:
            df_features = extract_all_features(
                df,
                include_eval_before=self.include_eval_before,
                num_workers=self.num_workers,
            )

        # Write partitioned Parquet dataset
        logger.info(
            "Writing extracted features to partitioned Parquet at %s...", out_path
        )
        out_table = pa.Table.from_pandas(df_features)

        part_schema = pa.schema([(col, pa.string()) for col in self.partition_cols])
        partitioning = ds.partitioning(part_schema, flavor="hive")

        ds.write_dataset(
            data=out_table,
            base_dir=str(out_path),
            format="parquet",
            partitioning=partitioning,
            basename_template="features_{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )

        total_runtime = time.time() - start_time

        # Feature count calculation
        feature_cols = (
            CHESS_FEATURE_NAMES
            + CONTEXT_FEATURE_NAMES
            + (EVAL_FEATURE_NAMES if self.include_eval_before else [])
        )

        null_summary = {}
        for c in feature_cols:
            if c in df_features.columns:
                null_cnt = int(df_features[c].isna().sum())
                null_summary[c] = {
                    "null_count": null_cnt,
                    "null_pct": round(float(null_cnt / total_rows * 100), 3)
                    if total_rows > 0
                    else 0.0,
                }

        summary = {
            "total_positions": total_rows,
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "runtime_seconds": round(total_runtime, 2),
            "throughput_pos_per_sec": round(total_rows / max(total_runtime, 0.001), 1),
            "include_eval_before": self.include_eval_before,
            "null_rates": null_summary,
            "output_dir": str(out_path),
        }

        logger.info(
            "Feature Pipeline complete! %d features generated for %d positions in %.2fs (%.1f pos/s).",
            len(feature_cols),
            total_rows,
            total_runtime,
            summary["throughput_pos_per_sec"],
        )

        return summary
