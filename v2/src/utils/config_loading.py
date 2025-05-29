### Functions for config files: loading, saving, and updatings
import json
import yaml
from datetime import datetime

# takes in a string of the file name, returns a dictionary with the contents of the file
def load_json_config(filename):
    with open(filename, "r") as f:
        print(f"Loading config from {filename}")
        config = json.load(f)
    return config

def load_yaml_config(filename):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config