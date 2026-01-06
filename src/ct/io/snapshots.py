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

    Parameters:
        snapshot_dir (Path): Directory where the snapshot is stored.
        snapshot_id (str): The unique snapshot ID.
        pull_cfg (dict): The pull configuration dictionary.
        output_file (Path): The path to the output data file.
        row_count (int): The number of rows in the output data file.
        config_path (str | None): Optional path to the configuration file used for the pull.
    
    Returns:
        Path: The path to the written metadata file.
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
    """
    Takes a prefix and makes a snapshot ID of the form:
    {timestamp}_{prefix}_{short_uuid} where a short_uuid is unique, random 6 hex characters.
    
    Parameters:
        prefix (str): A string prefix to include in the snapshot ID. Defaults to "raw" as this is
                        intended for raw data snapshots.
    
    Returns:
        str: The generated snapshot ID.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{prefix}_{short}"

def get_snapshot_id_from_path(snapshot_dir: Path) -> str:
    """
    Gets the snapshot ID from a snapshot directory path. Assumes format: {timestamp}_{prefix}_{short}
    
    Parameters:
        snapshot_dir (Path): The path to the snapshot directory.
    
    Returns:
        str: The snapshot ID extracted from the directory name.
    """
    return snapshot_dir.name.split("_")[0]

def write_latest_snapshot(snapshot_root: Path, snapshot_id: str) -> None:
    """
    Writes the latest snapshot ID to a 'latest' file in the snapshot root directory.

    Parameters:
        snapshot_root (Path): The root directory where snapshots are stored.
        snapshot_id (str): The snapshot ID to write as the latest.

    Returns:
        None
    """
    (snapshot_root / "latest").write_text(snapshot_id)

def read_latest_snapshot(snapshot_root: Path) -> str:
    """
    Reads the latest snapshot ID from a 'latest' file in the snapshot root directory.

    Parameters:
        snapshot_root (Path): The root directory where snapshots are stored.

    Returns:
        str: The latest snapshot ID.
    """
    return (snapshot_root / "latest").read_text().strip()