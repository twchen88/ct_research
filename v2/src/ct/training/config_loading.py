import yaml
from pathlib import Path
from typing import Any, Dict

# src/utils/config_loading.py
# ---------------------------
# Minimal config loader used by 03_train_predictor.py


def load_yaml_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    # (Optional) quick validation for the keys used in 03_train_predictor.py
    required = [
        ("settings", ["device", "run_desc", "seed", "test_ratio", "val_ratio"]),
        ("data", ["source", "destination_base"]),
        ("hyperparams", ["optimizer", "loss_function", "epochs", "batch_size", "n_samples", "learning_rate", "dims"]),
    ]
    for section, keys in required:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise KeyError(f"Missing section '{section}' in config.")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"Missing keys in '{section}': {missing}")
    return cfg
