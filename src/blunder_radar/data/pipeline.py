"""
End-to-end data pipeline for BlunderRadar.
Reads streamed PGNs, parses positions, writes partitioned Parquet, tracks resumable checkpoints,
and generates data quality reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from blunder_radar.config import DataConfig, load_data_config
from blunder_radar.data.quality_report import save_data_quality_report
from blunder_radar.data.stream_parser import MoveRecord, stream_pgn_positions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("blunder_radar.pipeline")


class PipelineCheckpoint:
    """Manages resumable state for the data ingestion pipeline."""

    def __init__(self, checkpoint_path: str | Path):
        self.path = Path(checkpoint_path)
        self.state: dict[str, Any] = {
            "positions_ingested": 0,
            "games_processed": 0,
            "batches_written": 0,
            "last_game_id": None,
            "completed": False,
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.state = json.load(f)
                logger.info(
                    "Loaded checkpoint from %s: %d positions ingested across %d games.",
                    self.path,
                    self.state.get("positions_ingested", 0),
                    self.state.get("games_processed", 0),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load checkpoint file (%s), starting fresh: %s", self.path, e
                )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def reset(self) -> None:
        self.state = {
            "positions_ingested": 0,
            "games_processed": 0,
            "batches_written": 0,
            "last_game_id": None,
            "completed": False,
        }
        if self.path.exists():
            self.path.unlink()


def write_batch_to_parquet(
    records: list[MoveRecord],
    output_dir: str | Path,
    batch_idx: int,
    partition_cols: list[str] = ("rating_band",),
) -> None:
    """
    Write a batch of MoveRecords into a partitioned Parquet dataset.
    """
    if not records:
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.to_dict() for r in records])
    df["batch_id"] = batch_idx

    table = pa.Table.from_pandas(df)

    # Use Hive-style partitioning (e.g., rating_band=mid)
    part_schema = pa.schema([(col, pa.string()) for col in partition_cols])
    partitioning = ds.partitioning(part_schema, flavor="hive")

    ds.write_dataset(
        data=table,
        base_dir=str(out_path),
        format="parquet",
        partitioning=partitioning,
        basename_template=f"batch_{batch_idx}_{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def read_dataset_as_dataframe(parquet_dir: str | Path) -> pd.DataFrame:
    """
    Read full partitioned Parquet dataset into a pandas DataFrame.
    """
    path = Path(parquet_dir)
    if not path.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()


def run_pipeline(
    config: DataConfig | None = None,
    source: str | None = None,
    target_positions: int | None = None,
    reset_state: bool = False,
) -> pd.DataFrame:
    """
    Execute the data ingestion pipeline end-to-end.
    """
    if config is None:
        config = load_data_config()

    target = target_positions or config.target_positions
    data_source = source or (
        config.local_zst_path if Path(config.local_zst_path or "").exists() else config.source_url
    )
    checkpoint = PipelineCheckpoint(config.checkpoint_file)

    if reset_state:
        logger.info("Resetting pipeline checkpoint and output directory...")
        checkpoint.reset()
        out_dir = Path(config.output_parquet_dir)
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)

    positions_ingested = checkpoint.state.get("positions_ingested", 0)
    games_processed = checkpoint.state.get("games_processed", 0)
    batches_written = checkpoint.state.get("batches_written", 0)

    if positions_ingested >= target:
        logger.info(
            "Target positions already reached (%d >= %d). Skipping streaming.",
            positions_ingested,
            target,
        )
        df_existing = read_dataset_as_dataframe(config.output_parquet_dir)
        save_data_quality_report(
            df_existing,
            md_path=config.quality_report_md,
            json_path=config.quality_report_json,
        )
        return df_existing

    remaining = target - positions_ingested
    logger.info(
        "Starting stream ingestion from: %s | Target: %d | Remaining: %d | Skip Games: %d",
        data_source,
        target,
        remaining,
        games_processed,
    )

    start_time = time.time()
    batch: list[MoveRecord] = []

    try:
        for record in stream_pgn_positions(
            source=data_source,
            config=config,
            max_positions=remaining,
            skip_games=games_processed,
        ):
            batch.append(record)
            positions_ingested += 1

            if len(batch) >= config.batch_size:
                batches_written += 1
                write_batch_to_parquet(
                    records=batch,
                    output_dir=config.output_parquet_dir,
                    batch_idx=batches_written,
                    partition_cols=config.partition_cols,
                )
                checkpoint.state["positions_ingested"] = positions_ingested
                checkpoint.state["batches_written"] = batches_written
                checkpoint.state["games_processed"] = record.game_idx
                checkpoint.state["last_game_id"] = record.game_id
                checkpoint.save()

                elapsed = time.time() - start_time
                rate = positions_ingested / max(elapsed, 0.001)
                logger.info(
                    "Ingested %d / %d positions (batch %d written, %.1f pos/s)",
                    positions_ingested,
                    target,
                    batches_written,
                    rate,
                )
                batch = []

        # Write any remaining records
        if batch:
            batches_written += 1
            write_batch_to_parquet(
                records=batch,
                output_dir=config.output_parquet_dir,
                batch_idx=batches_written,
                partition_cols=config.partition_cols,
            )
            checkpoint.state["positions_ingested"] = positions_ingested
            checkpoint.state["batches_written"] = batches_written
            checkpoint.state["games_processed"] = batch[-1].game_idx
            checkpoint.state["last_game_id"] = batch[-1].game_id
            checkpoint.save()

        checkpoint.state["completed"] = positions_ingested >= target
        checkpoint.save()

    except Exception as e:
        logger.error("Error during streaming pipeline: %s", e, exc_info=True)
        # Flush current batch if any
        if batch:
            batches_written += 1
            write_batch_to_parquet(
                records=batch,
                output_dir=config.output_parquet_dir,
                batch_idx=batches_written,
                partition_cols=config.partition_cols,
            )
            checkpoint.state["positions_ingested"] = positions_ingested
            checkpoint.state["batches_written"] = batches_written
            checkpoint.save()
        raise

    logger.info("Pipeline completed. Reading dataset for Data Quality Report...")
    df_all = read_dataset_as_dataframe(config.output_parquet_dir)
    logger.info("Read %d total positions from Parquet dataset.", len(df_all))

    report = save_data_quality_report(
        df_all,
        md_path=config.quality_report_md,
        json_path=config.quality_report_json,
    )
    logger.info(
        "Data Quality Report saved to %s and %s. Overall blunder rate: %s%%",
        config.quality_report_md,
        config.quality_report_json,
        report["class_balance"]["overall_blunder_rate_pct"],
    )
    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlunderRadar Data Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/data_config.yaml", help="Path to data config YAML"
    )
    parser.add_argument("--source", type=str, default=None, help="Override source PGN path or URL")
    parser.add_argument(
        "--max-positions", type=int, default=None, help="Target number of positions to ingest"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset state and overwrite existing dataset"
    )
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    run_pipeline(
        config=cfg,
        source=args.source,
        target_positions=args.max_positions,
        reset_state=args.reset,
    )
