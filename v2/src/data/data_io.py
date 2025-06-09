### File I/O utilities for reading and writing data files
import pandas as pd


## take in a string of file name of a CSV file and a dataframe, writes the dataframe to the CSV file
def write_sessions_to_csv(file_name : str, df : pd.DataFrame) -> None:
    df.to_csv(file_name, index=False)
    print(f"Dataframe written to {file_name}")


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


## take in a string of file name of a CSV file and returns a dataframe
def read_preprocessed_session_file(file_name : str) -> pd.DataFrame:
    dtype_map = {
    'patient_id': 'int32',
    **{f'domain {i} encoding': 'int8' for i in range(1, 15)},
    **{f'domain {i} score': 'float32' for i in range(1, 15)},
    **{f'domain {i} target': 'float32' for i in range(1, 15)},
    'time_stamp': 'int64',  # assuming it's UNIX time
    }
    return pd.read_csv(file_name, dtype=dtype_map, low_memory=False)