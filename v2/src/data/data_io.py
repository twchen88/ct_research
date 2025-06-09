### File I/O utilities for reading and writing data files
import pandas as pd

def read_raw_session_chunks(file_name: str, chunksize: int = 500_000):
    dtype_map = {
        'id': 'int32',
        'patient_id': 'int32',
        'task_type_id': 'int16',
        'task_level': 'int16',
        'domain_ids': 'string',
        'domain_scores': 'string'
    }

    for chunk in pd.read_csv(file_name, dtype=dtype_map, low_memory=False, chunksize=chunksize):
        chunk['start_time'] = pd.to_datetime(chunk['start_time'], errors='coerce')
        yield chunk

def read_raw_session_file(file_name: str, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Reads a large CSV file in chunks and applies preprocessing to reduce memory usage.

    Parameters:
        file_name (str): Path to the raw CSV file.
        chunksize (int): Number of rows per chunk to process.

    Returns:
        pd.DataFrame: concatenated DataFrame containing all processed chunks.
    """
        
    dtype_map = {
        'id': 'int32',
        'patient_id': 'int32',
        'task_type_id': 'int16',
        'task_level': 'int16',
        'domain_ids': 'string',
        'domain_scores': 'string'
    }

    def preprocess_chunk(chunk):
        chunk['start_time'] = pd.to_datetime(chunk['start_time'], errors='coerce')
        return chunk

    chunks = []
    for chunk in pd.read_csv(file_name, dtype=dtype_map, low_memory=False, chunksize=chunksize):
        chunks.append(preprocess_chunk(chunk))

    return pd.concat(chunks, ignore_index=True)


## take in a string of file name of a CSV file and a dataframe, writes the dataframe to the CSV file
def write_sessions_to_csv(file_name : str, df : pd.DataFrame) -> None:
    df.to_csv(file_name, index=False)
    print(f"Dataframe written to {file_name}")


## take in a string of file name of a CSV file and returns a dataframe
def read_preprocessed_session_file(file_name : str) -> pd.DataFrame:
    # Use low_memory=False to avoid dtype warning and ensure proper reading of mixed types
    return pd.read_csv(file_name, low_memory=False)