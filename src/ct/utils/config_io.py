### Functions for config files: loading, saving, and updatings
import json
import yaml
from ct.utils.logger import get_logger

"""
src/utils/config_io.py
--------------------------------
Functions for loading and saving files in JSON and YAML formats. It's named config_io because all JSON/YAML files
in this project are used for configuration or metadata purposes.
"""

logger = get_logger(__name__)

# takes in a string of the file name, returns a dictionary with the contents of the file
def load_json_config(filename: str):
    """
    Load a JSON configuration file.

    Parameters:
        filename (str): The path to the JSON file.
    """
    with open(filename, "r") as f:
        logger.info(f"Loading JSON config from {filename}")
        config = json.load(f)
    return config

def load_yaml_config(filename: str):
    """
    Load a YAML configuration file.

    Parameters:
        filename (str): The path to the YAML file.
    """
    with open(filename, "r") as f:
        logger.info(f"Loading YAML config from {filename}")
        config = yaml.safe_load(f)
    return config

# Functions for saving configuration files in JSON and YAML formats
def save_json_config(config: dict, filename: str):
    """
    Save a dictionary as a JSON configuration file.

    Parameters:
        config (dict): The configuration data to save.
        filename (str): The path to the JSON file.
    """
    with open(filename, "w") as f:
        logger.info(f"Saving JSON config to {filename}")
        json.dump(config, f, indent=4)

def save_yaml_config(config: dict, filename: str):
    """
    Save a dictionary as a YAML configuration file.

    Parameters:
        config (dict): The configuration data to save.
        filename (str): The path to the YAML file.
    """
    with open(filename, "w") as f:
        logger.info(f"Saving YAML config to {filename}")
        yaml.safe_dump(config, f)