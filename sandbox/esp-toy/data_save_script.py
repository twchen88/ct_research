import numpy as np
import os
from pathlib import Path
from typing import Union


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


def save_data_as_csv(input_path: Union[str, os.PathLike], output_path: Union[str, os.PathLike]) -> None:
    """Load model-ready data using load_data() and save it as a CSV file."""
    data = load_data(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, delimiter=",")

if __name__ == "__main__":
    # Example usage
    input_file = "/Users/964505/CT/ct_research/data/encoded/predictor_encoded_20250826.npy"  # Change to your input file path
    output_file = "esp_toy_real.csv"  # Change to your desired output file path
    save_data_as_csv(input_file, output_file)
    print(f"Data saved as CSV to {output_file}")