"""
ct.io.snapshots
----------------------------
Writes metadata about raw snapshots pulled from external source (SQL database).
"""

import yaml
import uuid
import socket
from pathlib import Path
from datetime import datetime, timezone
from ct.utils.metadata import get_git_commit_hash
from ct.utils.hashing import hash_dict

from ct.utils.logger import get_logger
logger = get_logger(__name__)

def write_snapshot_metadata(
    snapshot_dir: Path,
    *,
    snapshot_id: str,
    pull_cfg: dict,
    output_file: Path,
    row_count: int,
    config_path: str | None = None,
) -> Path:
    """
    Write immutable metadata describing a raw data snapshot.

    This should be called exactly once, after a successful pull.
    """
    meta = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),

        "source": {
            "kind": pull_cfg["source"].get("kind", "sql"),
            "sql_file_path": pull_cfg["source"].get("sql_file_path"),
            "sql_params": pull_cfg["source"].get("sql_params"),
        },

        "output": {
            "filename": output_file.name,
            "format": output_file.suffix.lstrip("."),
            "row_count": row_count,
        },

        "provenance": {
            "config_file": config_path,
            "config_hash": hash_dict(pull_cfg),
            "git_commit_hash": get_git_commit_hash(),
            "hostname": socket.gethostname(),
        },

        "user_metadata": pull_cfg.get("metadata", {}),
    }

    meta_path = snapshot_dir / "meta.yaml"
    with meta_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    logger.info("Snapshot metadata written to %s", meta_path)
    return meta_path

def make_snapshot_id(prefix: str = "raw") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{prefix}_{short}"

def get_snapshot_id_from_path(snapshot_dir: Path) -> str:
    return snapshot_dir.name.split("_")[0]  # Assumes format: {timestamp}_{prefix}_{short}

def write_latest_snapshot(snapshot_root: Path, snapshot_id: str):
    (snapshot_root / "latest").write_text(snapshot_id)

def read_latest_snapshot(snapshot_root: Path) -> str:
    return (snapshot_root / "latest").read_text().strip()