"""
scripts/pull_raw.py
----------------------------
Script to pull raw data from an external SQL database and save it to a CSV file,
using a configuration file for database connection details and query information.
"""

import argparse
import ct.utils.io as io
from pathlib import Path
from ct.io.pull_raw import pull_raw
from ct.io.snapshots import make_snapshot_id, write_latest_snapshot
from ct.config.schema import PullRawConfig
from ct.utils.logger import get_logger, configure_logging, install_excepthook

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    raw_cfg = io.load_yaml(args.config)
    cfg = PullRawConfig.model_validate(raw_cfg)

    snapshot_root = Path(cfg.output["snapshot_root"])
    snapshot_id = make_snapshot_id(prefix=cfg.snapshot["prefix"])
    snapshot_dir = snapshot_root / snapshot_id

    log_file = snapshot_dir / "pull.log"
    
    configure_logging(str(log_file), force=True)
    install_excepthook()

    logger = get_logger(__name__)
    logger.info(f"Starting raw data pull.")
    logger.info(f"Using config file: {args.config}")
    logger.info(f"Snapshot directory: {snapshot_dir}")

    pull_raw(cfg.model_dump(), snapshot_dir, snapshot_id=snapshot_id, config_path=args.config)
    logger.info("Raw data pull completed successfully.")
    if cfg.snapshot.get("update_latest", False):
        write_latest_snapshot(snapshot_root, snapshot_id)
        logger.info(f"Updated latest snapshot to {snapshot_id}.")
    

if __name__ == "__main__":
    main()