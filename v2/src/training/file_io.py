import os
import shutil
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

# src/training/file_io.py
# -----------------------
# Utilities used by 03_train_predictor.py for loading data and saving artifacts.


# ---------- Read ----------

def load_data(path: Union[str, os.PathLike]) -> np.ndarray:
    """Load model-ready data from disk.
    Supports .npy (NumPy array) and .npz with a 'data' key.
    Returns a NumPy ndarray.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if not isinstance(arr, np.ndarray):
            raise ValueError(".npy did not contain a NumPy array")
        return arr
    if p.suffix.lower() == ".npz":
        with np.load(p) as z:
            if "data" in z:
                return z["data"]
            # If unknown keys, pick the first array
            for k in z.files:
                return z[k]
        raise ValueError(".npz has no arrays")
    raise ValueError(f"Unsupported data format: {p.suffix}")


# ---------- Write ----------

def save_model(model: torch.nn.Module, path: Union[str, os.PathLike]) -> str:
    """Save the model state dict to a .pt file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    print(f"Model saved to {p}")
    return str(p)


def copy_config_file(src: Union[str, os.PathLike], dst: Union[str, os.PathLike]) -> str:
    """Copy the config YAML file into the run directory."""
    src_p, dst_p = Path(src), Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_p, dst_p)
    print(f"Config copied to {dst_p}")
    return str(dst_p)


def save_metrics(metrics: Dict[str, Any], path: Union[str, os.PathLike]) -> str:
    """Save scalar metrics as YAML (e.g., test_loss, test_error)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    print(f"Metrics saved to {p}")
    return str(p)


def save_results(results: Dict[str, Any], path: Union[str, os.PathLike]) -> str:
    """Save arrays and histories as a .npz bundle.
    Expected keys in 'results' may include: train_loss_history (list[float]),
    val_loss_history (list[float]), test_x, test_y, test_predictions (ndarrays).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert any lists to numpy arrays defensively
    pack = {}
    for k, v in results.items():
        if isinstance(v, list):
            pack[k] = np.array(v)
        else:
            pack[k] = v
    np.savez_compressed(p, **pack)
    print(f"Results saved to {p}")
    return str(p)


def save_metadata(*, run_desc: str, git_commit_hash: str, input_path: str, output_path: str,
                  config_path: str, metrics_path: str, plots_path: str) -> Dict[str, Any]:
    """Write a metadata.yaml into the run directory so a single file links everything."""
    metadata = {
        "time_stamp": datetime.now().isoformat(),
        "run_desc": run_desc,
        "git_commit_hash": git_commit_hash,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "plots_path": str(plots_path),
    }
    p = Path(output_path) / "metadata.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f"Metadata saved to {p}")
    return metadata
