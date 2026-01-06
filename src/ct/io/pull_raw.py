"""
ct.io.pull_raw
----------------------------
Defines public entrypoint and helper functions that pull raw data from external source (SQL database).
"""

import pandas as pd
from pathlib import Path
from ct.io.db_utils import connect_engine, load_sql
from ct.io.snapshots import write_snapshot_metadata, get_snapshot_id_from_path

from ct.utils.logger import get_logger
logger = get_logger(__name__)

def _write_sessions_to_csv(file_name: Path, df: pd.DataFrame) -> None:
    """
    Writes a DataFrame to a CSV file.
    Parameters:
        file_name (str): The name of the file to write to.
        df (pd.DataFrame): The DataFrame to write.
    """
    df.to_csv(file_name, index=False)
    logger.info(f"Query results saved to {file_name}.")

def pull_raw(pull_cfg: dict, snapshot_dir: Path, snapshot_id: str, config_path: str | None = None) -> Path:
    """
    Public entrypoint to pull raw data from external source (SQL database) based on provided configuration.
    Parameters:
        pull_cfg (dict): Configuration dictionary containing source and output details.
    Returns:
        Path: The path to the saved CSV file containing the raw data.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # establish connection to the database (via SQLAlchemy Engine)
    try:
        with connect_engine(pull_cfg["source"]["sql_params"]) as engine:
            logger.info("Database connection established.")

            # load SQL query results from SQL file (or raw SQL string)
            data: pd.DataFrame = load_sql(pull_cfg["source"]["sql_file_path"], engine)
            logger.info("SQL query results loaded successfully.")

    except Exception as e:
        # If anything fails (tunnel, engine, read_sql), surface a clear error
        logger.error(f"Failed to load data from SQL: {e}")
        raise RuntimeError(f"Failed to load data from SQL: {e}") from e

    ## save the query results to a CSV file
    output_data_file_path =  snapshot_dir / pull_cfg["output"]["filename"]
    _write_sessions_to_csv(output_data_file_path, data)

    # save snapshot metadata
    write_snapshot_metadata(
        snapshot_dir=snapshot_dir,
        snapshot_id=snapshot_id,
        pull_cfg=pull_cfg,
        output_file=output_data_file_path,
        row_count=len(data),
        config_path=config_path,
    )
    return output_data_file_path