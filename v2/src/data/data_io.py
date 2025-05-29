### File I/O utilities for reading and writing data files
import pandas as pd

## take in a string of file name of a CSV file and returns a dataframe
def read_session_file(file_name):
    return pd.read_csv(file_name)

## take in a string of file name of a CSV file and a dataframe, writes the dataframe to the CSV file
def write_sessions_to_csv(file_name, df):
    df.to_csv(file_name, index=False)
    print(f"Dataframe written to {file_name}")