import torch
import shutil
import yaml
import numpy as np

from datetime import datetime
from pathlib import Path

"""
src/training/file_io.py
------------------------
This module provides functions for reading and writing data, models, metrics, and configuration files.
It includes functions to load data from a specified path, save models and metrics, copy configuration files, and save metadata about the training process.
"""

## read functions
def load_data(path: str) -> np.ndarray:
    """
    Load data from the specified path.
    
    Parameters:
        path (str): The path to the data file.
        
    Returns:
        np.ndarray: The loaded data as a NumPy array.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    data = np.load(path)
    print(f"Data loaded from {path} with shape {data.shape}")
    return data


## write functions
def save_model(model, path: str):
    """
    Save the model to the specified path.
    
    Parameters:
        model: The model to save.
        path (str): The path where the model will be saved.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def save_metrics(metrics: dict, path: str):
    """
    Save the metrics to the specified path.
    
    Parameters:
        metrics (dict): The metrics to save.
        path (str): The path where the metrics will be saved.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **metrics)
    print(f"Metrics saved to {path}")


def copy_config_file(source: str, target: str):
    """
    Copy the configuration file from source to target.
    
    Parameters:
        source (str): The path to the source configuration file.
        target (str): The path where the configuration file will be copied.
    """
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    print(f"Configuration file copied from {source} to {target}")


def save_metadata(run_desc: str, git_commit_hash: str, input_path: str, output_path: str, config_path: str, metrics_path: str, plots_path: str):
    """
    Save metadata about the training process.
    
    Parameters:
        input_path (str): The path to the input data.
        output_path (str): The path to the output data.
        config_path (str): The path to the configuration file used for training.
        metrics_path (str): The path to the saved metrics.
    """
    metadata = {
        "time_stamp": datetime.now().isoformat(),
        "run_desc": run_desc,
        "git_commit_hash": git_commit_hash,
        "input_path": input_path,
        "output_path": output_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "plots_path": plots_path
    }
    metadata_path = Path(output_path) / "metadata.yaml"
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"Metadata saved: {metadata}")
    return metadata