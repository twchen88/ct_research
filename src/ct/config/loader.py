from pathlib import Path
from ct.utils.io import load_yaml

from ct.utils.logger import get_logger
logger = get_logger(__name__)

def load_config(filename: str) -> dict:
    config = load_yaml(filename)
    if config.get("schema_version") == 1:
        logger.error("Config schema_version=1 is no longer supported")
        raise ValueError("Config schema_version=1 is no longer supported")
    logger.info(f"Loaded config from {filename}.")
    return config