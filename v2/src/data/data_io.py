### File I/O utilities for reading and writing data files
import pandas as pd

## take in a string of file name of a CSV file and returns a dataframe
def read_raw_session_file(file_name : str) -> pd.DataFrame:
    # Use low_memory=False and custom dtype_map to avoid dtype warning and ensure proper reading of mixed types
    dtype_map = {'domain_ids': 'string', 'domain_scores': 'string'}
    return pd.read_csv(file_name, dtype=dtype_map, low_memory=False)

## take in a string of file name of a CSV file and a dataframe, writes the dataframe to the CSV file
def write_sessions_to_csv(file_name : str, df : pd.DataFrame) -> None:
    df.to_csv(file_name, index=False)
    print(f"Dataframe written to {file_name}")


## take in a string of file name of a CSV file and returns a dataframe
def read_preprocessed_session_file(file_name : str) -> pd.DataFrame:
    # Use low_memory=False to avoid dtype warning and ensure proper reading of mixed types
    return pd.read_csv(file_name, low_memory=False)