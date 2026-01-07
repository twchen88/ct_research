"""
ct.utils.io
--------------------------------
Functions for generic input/output operations (e.g., reading/writing JSON, YAML).
"""
import json
import yaml
from pathlib import Path

def load_json(filename: str | Path) -> dict:
    with open(filename, "r") as f:
        return json.load(f)

def load_yaml(filename: str | Path) -> dict:
    with open(filename, "r") as f:
        return yaml.safe_load(f)

def save_json(data: dict, filename: str | Path):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def save_yaml(data: dict, filename: str | Path):
    with open(filename, "w") as f:
        yaml.safe_dump(data, f)