### Functions for config files: loading, saving, and updatings
import json
import yaml

"""
src/utils/config_loading.py
Functions for loading configuration files in JSON and YAML formats.
"""

# takes in a string of the file name, returns a dictionary with the contents of the file
def load_json_config(filename: str):
    """
    Load a JSON configuration file.

    Parameters:
        filename (str): The path to the JSON file.
    """
    with open(filename, "r") as f:
        print(f"Loading config from {filename}")
        config = json.load(f)
    return config

def load_yaml_config(filename: str):
    """
    Load a YAML configuration file.

    Parameters:
        filename (str): The path to the YAML file.
    """
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config