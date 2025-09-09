import numpy as np
from typing import List, Tuple
from pathlib import Path

"""
src/experiments/file_io.py
----------------------------
File I/O operations for loading and saving test data and predictions as well as saving necessary results.
"""

## read
def load_test_data(file_path: str, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from a specified file path.

    Parameters:
    - file_path (str): Path to the file containing the test data.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Loaded test data as a tuple of NumPy arrays, in the order of (input, target).
    """
    try:
        data = np.load(file_path)
        return (data[file_names[0]], data[file_names[1]])
    except Exception as e:
        raise IOError(f"Error loading test data from {file_path}: {e}")
    

def load_saved_test_predictions(file_path: str) -> np.ndarray:
    """
    Load saved test predictions from a specified file path.

    Parameters:
    - file_path (str): Path to the file containing the saved predictions.

    Returns:
    - np.ndarray: Loaded predictions as a NumPy array.
    """
    try:
        data = np.load(file_path)
        return data['test_predictions']
    except Exception as e:
        raise IOError(f"Error loading saved predictions from {file_path}: {e}")
    


## write
def save_ground_truth_matrices(ground_truth_dict: dict, output_path: Path):
    """
    Save ground truth matrices for each missing count to a specified output path.

    Parameters:
    - ground_truth_dict (dict): Dictionary containing ground truth matrices indexed by missing count.
    - output_path (str): Path to save the ground truth matrices.
    """
    try:
        np.savez(output_path, **ground_truth_dict)
        print(f"Ground truth matrices saved to {output_path}")
    except Exception as e:
        raise IOError(f"Error saving ground truth matrices to {output_path}: {e}")