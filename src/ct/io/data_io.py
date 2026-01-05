import pandas as pd
import numpy as np

from typing import Iterator

"""
src/data/dat_io.py
-----------------
This module contains functions for reading and wrting data files, specifically for session data in CSV and numpy formats.
It is separated into two main sections: write functions and read functions.
* Write functions handle writing DataFrames to CSV files and numpy arrays to .npy files.
* Read functions handle reading CSV files regularly or in chunks, specifying dtypes to reduce memory usage if needed.
"""


## Write functions
def write_sessions_to_npy(file_name: str, data: np.ndarray) -> None:
    """
    Writes a numpy array to a .npy file. Used in script 02 when saving to numpy format.

    Parameters:
        file_name (str): The name of the file to write to.
        data (np.ndarray): The numpy array to write.
    """
    np.save(file_name, data)
    print(f"Numpy array written to {file_name}")



## Read functions
def read_raw_session_chunks(file_name: str, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """
    Reads a CSV file in chunks and yields each chunk as a DataFrame. Used when processing large raw session data files in script 01 to save on memory.
    Parameters:
        file_name (str): The name of the CSV file to read.
        chunksize (int): The number of rows per chunk. Default is 500,000.
    Yields:
        pd.DataFrame: A chunk of the DataFrame read from the CSV file.
    """
    dtype_map = {
        'id': 'int32',
        'patient_id': 'int32',
        'task_type_id': 'int16',
        'task_level': 'int16',
        'domain_ids': 'string',
        'domain_scores': 'string'
    }

    for chunk in pd.read_csv(file_name, dtype=dtype_map, low_memory=False, chunksize=chunksize): # type: ignore
        chunk['start_time'] = pd.to_datetime(chunk['start_time'], errors='coerce')
        yield chunk


## take in a string of file name of a CSV file and returns a dataframe
def read_preprocessed_session_file(file_name: str) -> pd.DataFrame:
    """
    Reads a preprocessed session CSV file into a DataFrame, with specified dtypes to reduce memory usage. Used in script 02 to read preprocessed session data.
    Parameters:
        file_name (str): The name of the CSV file to read.
    Returns:
        pd.DataFrame: The DataFrame containing the session data.
    """
    dtype_map = {
    'patient_id': 'int32',
    **{f'domain {i} encoding': 'int8' for i in range(1, 15)},
    **{f'domain {i} score': 'float32' for i in range(1, 15)},
    **{f'domain {i} target': 'float32' for i in range(1, 15)},
    'time_stamp': 'int64',  # assuming it's UNIX time
    }
    return pd.read_csv(file_name, dtype=dtype_map, low_memory=False) # type: ignore