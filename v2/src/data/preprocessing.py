### File contains functions for preprocessing data
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN

# drop duplicates based on specific columns (default to None)
def drop_duplicates(df, based_on=None):
    df = df.drop_duplicates(subset=based_on, keep='first')
    return df

# given a dataframe, turn start time to a datetime format and sort by start time, return the sorted dataframe
def sort_by_start_time(df):
    df = df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.sort_values(by='start_time').reset_index(drop=True)
    return df

# given a dataframe of all considered sessions, for each patient, find usage frequency in terms of unique days,
# return a dataframe with patient_id, session count, and unique days
def find_usage_frequency(df):
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


# given a session, take domain_ids and domain_scores, which are in string format separated by ",", 
# and replace with a list of the values
# used with extract_session_data()
def process_row(row):
    values_a = [int(x.strip()) for x in str(row['domain_ids']).split(',')]
    values_b = [float(x.strip()) for x in str(row['domain_scores']).split(',')]
    return values_a, values_b

# Given a dataframe that contains sessions for a single patient, for each session/row, find most updated domain scores for all 14 domains,
# keep previous scores, and encode domains as a binary vector. At the end, return a dataframe with these information.
def extract_session_data(data: pd.DataFrame):
    # Initialize variables
    session_row = [] # contents of a row (patient id, encoding, cur score, prev score...)
    overall = [] # aggregate of everything (n sessions x 45)

    cur_score = np.zeros((14)) # score for each session
    cur_score.fill(np.nan)
    prev_score = np.zeros((14)) # score for previous session
    prev_score.fill(np.nan)

    patient_id = data["patient_id"].iloc[0] # store patient id for the dataframe

    # iterate through each row
    for _, row in data.iterrows():
        domains, domain_scores = process_row(row)  # returns a list of domains : int and of domain_scores : float

        # iterate through domains practiced in this session and update current scores
        for j, domain in enumerate(domains):
            cur_score[domain - 1] = domain_scores[j] # domain - 1 because domain ids are 1-indexed, but numpy is 0-indexed

        # Encode domains for this session
        domain_encoding = np.isin(np.arange(1, 15), domains, assume_unique=True).astype(int)
        
        # append everything to the row list
        session_row.append(patient_id)
        session_row.extend(domain_encoding.copy().tolist())
        session_row.extend(prev_score.copy().tolist())
        session_row.extend(cur_score.copy().tolist())
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

# filter out session gaps
def filter_datetime_outliers(data, column="start_time", eps_days=14, min_samples=5):

    df = data.copy()

    # Convert datetime column to date only (removing time)
    df["date"] = df[column].dt.date

    # Get unique days as a DataFrame
    unique_dates = pd.DataFrame(df["date"].unique(), columns=["date"])

    # Convert unique dates to numerical timestamps (days since epoch)
    unique_dates["timestamp"] = pd.to_datetime(unique_dates["date"]).astype(np.int64) // 10**9

    # Apply DBSCAN clustering on unique dates
    eps_seconds = eps_days * 24 * 60 * 60  # Convert days to seconds
    clustering = DBSCAN(eps=eps_seconds, min_samples=min_samples).fit(unique_dates[["timestamp"]])

    # Assign cluster labels
    unique_dates["cluster"] = clustering.labels_

    # Keep only clustered dates
    clustered_dates = unique_dates[unique_dates["cluster"] != -1]["date"]

    # Filter original DataFrame to keep only records from clustered dates
    filtered_df = df[df["date"].isin(clustered_dates)].drop(columns=["date"])

    return filtered_df

# modified from https://stackoverflow.com/questions/56750841/how-to-trim-outliers-in-dates-in-python
def datetime_outlier(data):
    qa = data["start_time"].quantile(0.2) #lower 10%
    qb = data["start_time"] #higher 10%
    #remove outliers
    xf = data[(data.start_time >= qa) & (data.start_time <= qb)]
    return xf

# save metadata about the output file and configuration used to generate it
def save_metadata(input_path, output_path, config_path, config, stats):
    metadata = {
        "input_file": input_path,
        "output_file": output_path,
        "config_file": config_path,
        "timestamp": datetime.now().isoformat(),
        "params": config["filter_params"],
        "output_file_stats": stats
    }
    meta_path = output_path.replace(".csv", ".meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)