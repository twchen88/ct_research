import numpy as np
import os
from pathlib import Path
from typing import Union
import pandas as pd


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


def save_data_as_csv(input_path: Union[str, Path], output_path: Union[str, Path], columns: list[str]) -> None:
    """Load model-ready data and save as a CSV file with column names."""
    data = load_data(input_path)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] != len(columns):
        raise ValueError(f"Number of columns ({len(columns)}) does not match data shape {data.shape}")

    df = pd.DataFrame(data, columns=columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    input_file = "data/encoded/predictor_encoded_20250826.npy"
    output_file = "sandbox/esp-toy/esp_toy_real.csv"
    column_names = ['target_d_1', 'target_d_2', 'target_d_3', 'target_d_4',
       'target_d_5', 'target_d_6', 'target_d_7', 'target_d_8', 'target_d_9',
       'target_d_10', 'target_d_11', 'target_d_12', 'target_d_13',
       'target_d_14', 'd_1_score', 'd_1_ind', 'd_2_score', 'd_2_ind',
       'd_3_score', 'd_3_ind', 'd_4_score', 'd_4_ind', 'd_5_score', 'd_5_ind',
       'd_6_score', 'd_6_ind', 'd_7_score', 'd_7_ind', 'd_8_score', 'd_8_ind',
       'd_9_score', 'd_9_ind', 'd_10_score', 'd_10_ind', 'd_11_score',
       'd_11_ind', 'd_12_score', 'd_12_ind', 'd_13_score', 'd_13_ind',
       'd_14_score', 'd_14_ind', 'd_1_next', 'd_2_next', 'd_3_next',
       'd_4_next', 'd_5_next', 'd_6_next', 'd_7_next', 'd_8_next', 'd_9_next',
       'd_10_next', 'd_11_next', 'd_12_next', 'd_13_next', 'd_14_next']
    save_data_as_csv(input_file, output_file, column_names)
    print(f"Data saved as CSV to {output_file}")