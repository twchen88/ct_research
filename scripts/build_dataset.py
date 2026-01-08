#!/usr/bin/env python3
"""
scripts/build_dataset.py
------------------------
CLI wrapper that loads a dataset YAML config and calls ct.datasets.build_dataset.build_dataset().

Assumptions (per your note):
  - build_dataset() is defined in: ct.datasets.build_dataset
  - hash_dict is in: ct.utils.hashing (used inside build_dataset, not necessarily here)
  - YAML/JSON I/O helpers are in: ct.utils.io (used inside build_dataset, not necessarily here)
  - config loader is in: ct.config.loader (we use it here)

This script is intentionally thin: it just resolves config + paths and invokes build_dataset().
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ct.datasets.build_dataset import build_dataset, DatasetArtifact
from ct.config.loader import load_config  # expected to load + validate YAML into a dict
from ct.utils.logger import get_logger

logger = get_logger(__name__)


def _coerce_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a dataset artifact from a raw snapshot + YAML config.")
    ap.add_argument("--config", required=True, type=str, help="Path to dataset YAML config.")
    ap.add_argument(
        "--raw-snapshot-dir",
        required=True,
        type=str,
        help="Path to raw snapshot directory (e.g. data/raw/snapshots/<snapshot_id>).",
    )
    ap.add_argument(
        "--raw-snapshot-id",
        required=True,
        type=str,
        help="Snapshot id string for provenance (should match directory name; NOT 'latest').",
    )
    ap.add_argument(
        "--artifact-root",
        required=True,
        type=str,
        help="Artifact root directory (e.g. artifacts/).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if artifact already exists.",
    )
    args = ap.parse_args()

    config_path = _coerce_path(args.config)
    raw_snapshot_dir = _coerce_path(args.raw_snapshot_dir)
    artifact_root = _coerce_path(args.artifact_root)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not raw_snapshot_dir.exists():
        raise FileNotFoundError(f"Raw snapshot dir not found: {raw_snapshot_dir}")

    artifact_root.mkdir(parents=True, exist_ok=True)

    # Load config using your shared loader (can include validation/defaults)
    cfg: Dict[str, Any] = load_config(config_path)

    art: DatasetArtifact = build_dataset(
        cfg=cfg,
        raw_snapshot_dir=raw_snapshot_dir,
        raw_snapshot_id=args.raw_snapshot_id,
        artifact_root=artifact_root,
        config_path=str(config_path),  # provenance
        force=bool(args.force),
    )

    logger.info(f"Built dataset_id={art.dataset_id}")
    logger.info(f"Artifact dir: {art.artifact_dir}")
    logger.info(f"Dataset:      {art.dataset_path}")
    logger.info(f"Schema:       {art.schema_path}")
    logger.info(f"Features:     {art.features_path}")
    logger.info(f"Meta:         {art.meta_path}")
    logger.info(f"Done:         {art.done_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())