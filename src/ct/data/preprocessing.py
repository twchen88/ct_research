### File contains functions for preprocessing data
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from typing import Dict, Any, Tuple, List

"""
src/data/preprocessing.py
-----------------
This module contains functions for preprocessing data, including dropping duplicates, finding usage frequency and time for filtering purposes,
and filtering outliers in datetime data.
* drop_duplicates: Drops duplicate rows based on specified columns.
* sort_by_start_time: Converts start time to datetime format and sorts the DataFrame by start time.
* find_usage_frequency: Calculates usage frequency for each patient based on unique days and session count.
* extract_session_data: Processes session data to extract domain scores and encodings in vectorized form.
* filter_datetime_outliers: Filters outliers in datetime data using DBSCAN clustering.
* save_metadata: Saves metadata about the output file and configuration used to generate it.
"""


def drop_duplicates(df: pd.DataFrame, based_on: List[str]) -> pd.DataFrame:
    """
    Drop duplicate rows in a DataFrame based on specified columns, keeping the first appearance.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        based_on (list): List of column names to consider for identifying duplicates.

    Returns:
        pd.DataFrame: DataFrame with duplicates dropped, keeping the first occurrence.
    """
    df = df.drop_duplicates(subset=based_on, keep='first')
    return df


def sort_by_start_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, sort by start time, return the sorted dataframe.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
    
    Returns:
        pd.DataFrame: DataFrame sorted by start time.
    """
    df = df.copy()
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df


def find_usage_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of all considered sessions, for each patient, find usage frequency in terms of unique days, 
    return a dataframe with patient_id, usage_time, and unique_days.

    Parameters:
        df (pd.DataFrame): The DataFrame containing session data with 'patient_id' and 'start_time' columns.

    Returns:
        pd.DataFrame: DataFrame with 'patient_id', 'usage_time', 'unique_days', and 'usage_freq'.
    """
    df = sort_by_start_time(df).copy()          # keep original intact
    df["start_date"] = df["start_time"].dt.date # strip time for distinct-day count
    # 1) distinct active days per patient
    unique_days = df.groupby("patient_id")["start_date"].nunique().rename("unique_days")
    # 2) span from first to last session (inclusive)
    span = (df.groupby("patient_id")["start_time"].max() - df.groupby("patient_id")["start_time"].min()).dt.days.add(1).rename("usage_time")
    # 3) combine and compute frequency
    usage = pd.concat([unique_days, span], axis=1).reset_index()
    usage["usage_freq"] = usage["unique_days"] / usage["usage_time"]
    return usage


def process_row(row: pd.Series) -> Tuple[List[int], List[float]]:
    """
    Given a session, take domain_ids and domain_scores, which are in string format separated by ",", and replace with a list of the values.
    This function is a helper function for extract_session_data()

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'domain_ids' and 'domain_scores'.
    
    Returns:
        tuple: A tuple containing two lists:
            - values_a: List of integers representing domain IDs.
            - values_b: List of floats representing domain scores.
    """
    values_a = [int(x.strip()) for x in str(row['domain_ids']).split(',')]
    values_b = [float(x.strip()) for x in str(row['domain_scores']).split(',')]
    return values_a, values_b



def extract_session_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe that contains sessions for a single patient, for each session/row, find most updated domain scores for all 14 domains,
    record previous scores, and encode domains as a binary vector. At the end, return a dataframe with these information.

    Parameters:
        data (pd.DataFrame): The DataFrame containing session data with 'patient_id', 'domain_ids', 'domain_scores', and 'start_time' columns.
    
    Returns:
        pd.DataFrame: A DataFrame with patient_id, domain encodings, previous scores, current scores, and start time."""
    # Initialize variables
    session_row = [] # contents of a row (patient id, encoding, cur score, prev score...)
    overall = [] # aggregate of everything (n sessions x 45)

    cur_score = np.zeros((14)) # score for each session
    cur_score.fill(np.nan)
    prev_score = np.zeros((14)) # score for previous session
    prev_score.fill(np.nan)

    patient_id = data["patient_id"].iloc[0] # store patient id for the dataframe

    # sort by start time
    data = sort_by_start_time(data)

    # iterate through each row
    for _, row in data.iterrows():
        domains, domain_scores = process_row(row)  # returns a list of domains : int and of domain_scores : float

        # iterate through domains practiced in this session and update current scores
        for j, domain in enumerate(domains):
            cur_score[domain - 1] = domain_scores[j] # domain - 1 because domain ids are 1-indexed, but numpy is 0-indexed

        # Encode domains for this session
        domain_encoding = np.isin(np.arange(1, 15), domains, assume_unique=True).astype(int)
        
        # append everything to the row list
        session_row.append(patient_id) # patient id
        session_row.extend(domain_encoding.copy().tolist()) # encoding columns
        session_row.extend(prev_score.copy().tolist()) # socre columns
        session_row.extend(cur_score.copy().tolist()) # target columns
        session_row.append(row["start_time"])
        session_row.append(row["start_time"].timestamp())

        # append row to overall, reset
        overall.append(session_row)
        session_row = []
        prev_score = cur_score.copy()
    
    # Create column names
    column_names = (
        ["patient_id"]
        + [f"domain {i} encoding" for i in range(1, 15)]
        + [f"domain {i} score" for i in range(1, 15)]
        + [f"domain {i} target" for i in range(1, 15)]
        + ["start_time"]
        + ["time_stamp"]
    )

    # Create dataframe
    scores_df = pd.DataFrame(overall, columns=column_names)
    scores_df.reset_index(drop=True, inplace=True)
    return scores_df



# (copied from commit 9f8d808)
def filter_datetime_outliers(data: pd.DataFrame, eps_days: int, min_samples: int) -> pd.DataFrame:
    """
    Uses DBSCAN clustering to filter outliers in datetime data based on the start_time column.

    Parameters:
        data (pd.DataFrame): The DataFrame containing session data with 'start_time' column.
        eps_days (int): The maximum distance between two samples for one to be considered as in the neighborhood of the other, in days.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed, containing the original columns except 'timestamp' and 'cluster'.
    """
    df = data.copy()

    # Convert dates to numerical timestamps
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df["timestamp"] = df['start_time'].astype(np.int64) // 10**9  # Convert to seconds

    # Apply DBSCAN clustering
    eps_seconds = eps_days * 24 * 60 * 60  # Convert days to seconds
    clustering = DBSCAN(eps=eps_seconds, min_samples=min_samples).fit(df[["timestamp"]])
    
    # Assign cluster labels
    df["cluster"] = clustering.labels_

    # Remove outliers (DBSCAN labels outliers as -1)
    filtered_df = df[df["cluster"] != -1].drop(columns=["timestamp", "cluster"])

    return filtered_df


def convert_to_percentile(df: pd.DataFrame, columns: List[str] = [f"domain {i} score" for i in range(1, 15)] + [f"domain {i} target" for i in range(1, 15)]) -> pd.DataFrame:
    """
    Convert specified columns in a DataFrame to percentiles.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to convert to percentiles.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to percentiles.
    """
    for col in columns:
        df[col] = df[col].rank(pct=True)
    return df