import yaml
import pandas as pd
import numpy as np

"""
src/data/encoding.py
-----------------
This module contains functions for encoding data.
Make preprocessed data numerically and structurally compatible with the input expectations of prediction neural network.
* filter_nonzero_rows: Filters rows in a DataFrame that have at most a specified number of zeros.
* create_missing_indicator: Encodes a numpy array by replacing each value with a pair indicating whether it is NaN or not.
"""

def filter_nonzero_rows(df: pd.DataFrame, max_zeros: int) -> pd.DataFrame:
    """
    Returns a copy of the DataFrame containing only rows with at most `max_zeros` number of zeros.

    Parameters:
        df (pd.DataFrame): Input DataFrame to filter.
        max_zeros (int): Maximum allowed number of zeros per row.

    Returns:
        pd.DataFrame: Filtered DataFrame with qualifying rows.
    """
    zero_counts = (df == 0).sum(axis=1)
    filtered_df = df[zero_counts <= max_zeros].copy()
    return filtered_df


def create_missing_indicator(data: np.ndarray, rand_seed: int = 42) -> np.ndarray:
    """
    Encodes a numpy array by replacing each value with a pair:
    - If the value is NaN: [0,0] or [1,1] chosen randomly.
    - If the value is not NaN: [value, 1 - value].

    Parameters:
        data (np.ndarray): 2D array of floats with possible NaNs.
        rand_seed (int): Seed for reproducibility.

    Returns:
        np.ndarray: 2D array with encoded values.
    """
    np.random.seed(rand_seed)
    rows, cols = data.shape
    encoded = np.empty((rows, cols * 2), dtype=data.dtype)

    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            if np.isnan(val):
                choice = np.random.choice([0, 1])
                encoded[i, j * 2] = choice
                encoded[i, j * 2 + 1] = choice
            else:
                encoded[i, j * 2] = val
                encoded[i, j * 2 + 1] = 1 - val

    return encoded.copy()


def encode_target_data(target: np.ndarray, encoded_domains: np.ndarray) -> np.ndarray:
    """
    Encodes target data by replacing non-target values with 0s and target values with target score.
    - if encoding is 0, then target is 0
    - if encoding is 1, then target is target score

    Parameters:
        target (np.ndarray): 2D array of floats with possible NaNs.
        encoded_domains (np.ndarray): 2D array of encoded domain values (0 or 1).

    Returns:
        np.ndarray: 2D array with encoded values.
    """
    product = np.multiply(target, encoded_domains)
    return np.nan_to_num(product, nan=0)


def save_metadata(input_path: str, output_path: str, config_path: str, column_names: list) -> None:
    """
    Saves metadata about the output file and configuration used to generate it.

    Parameters:
        output_file_path (str): Path to the output file.
        config_file_path (str): Path to the configuration file.
        config (dict): Configuration dictionary.
        column_names (list): List of column names in the output file.
    """
    metadata = {
        "input_path": input_path,
        "output_file_path": output_path,
        "config_file_path": config_path,
        "column_names": column_names
    }

    meta_path = output_path.replace(".npy", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)