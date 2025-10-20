# @ file: v2/src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import ct.data.preprocessing as preprocessing

def test_drop_duplicates():
    # Create a sample DataFrame
    data = {
        'id': [1, 2, 2, 3],
        'name': ['Alice', 'Bob', 'Brent', 'Charlie'],
        'age': [25, 30, 30, 35]
    }
    df = pd.DataFrame(data)
    # Brent is a duplicate of Bob, but with a different name so we can test that the drop is based on column 'id'

    # Drop duplicates based on the 'id' colum
    result_df = preprocessing.drop_duplicates(df, based_on=['id'])

    # Check if duplicates are dropped correctly
    assert len(result_df) == 3
    assert result_df['id'].tolist() == [1, 2, 3]

def test_sort_by_start_time():
    # Create a sample DataFrame
    data = {
        'id': [1, 2, 3],
        'start_time': ['2023-01-02 10:00:00', '2023-01-01 09:00:00', '2023-01-01 11:00:00']
    }
    df = pd.DataFrame(data)
    # Convert start_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])

    # Sort by start_time
    sorted_df = preprocessing.sort_by_start_time(df)

    # Check if the DataFrame is sorted correctly
    assert sorted_df['start_time'].iloc[0] == pd.to_datetime('2023-01-01 09:00:00')
    assert sorted_df['start_time'].iloc[1] == pd.to_datetime('2023-01-01 11:00:00')
    assert sorted_df['start_time'].iloc[2] == pd.to_datetime('2023-01-02 10:00:00')


def test_find_usage_frequency():
    # Create a sample DataFrame
    data = {
        'id': [1, 2, 3, 4, 5],
        'patient_id': [1, 1, 2, 2, 2],
        'start_time': ['2023-01-01 09:00:00', '2023-01-05 09:00:00', '2023-01-02 09:00:00', '2023-01-03 09:00:00', '2023-01-03 10:00:00'],
    }

    df = pd.DataFrame(data)
    # Convert start_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])

    # Find usage days
    usage_freq_df = preprocessing.find_usage_frequency(df)

    # Check if the usage days are calculated correctly
    assert len(usage_freq_df) == 2 ## for two patients
    assert usage_freq_df['patient_id'].tolist() == [1, 2]
    assert usage_freq_df['unique_days'].tolist() == [2, 2]  # 2 days for each patient
    assert usage_freq_df['usage_time'].tolist() == [5, 2]
    assert usage_freq_df['usage_freq'].tolist() == [0.4, 1]
    assert usage_freq_df.columns.tolist() == ['patient_id', 'unique_days', 'usage_time', 'usage_freq']

def test_process_row():
    # Create a sample row with domain_ids and domain_scores
    data = {
        'domain_ids': ['1,2,3'],
        'domain_scores': ['0.5,0.6,0.7']
    }

    df = pd.DataFrame(data)

    processed_row = preprocessing.process_row(df.iloc[0])
    assert processed_row[0] == [1, 2, 3]  # domain_ids
    assert processed_row[1] == [0.5, 0.6, 0.7]  # domain_scores

def test_extract_session_data():
    # Create a sample DataFrame of a patient's session history
    data = sample_sessions = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'patient_id': [1, 1, 1, 1, 1, 2, 1, 2, 1, 1,
                   1, 2, 1, 2, 1, 1, 2, 1, 1, 2],
    'start_time': [
        '2025-05-01 09:00:00', '2025-05-01 13:00:00', '2025-05-01 17:00:00',
        '2025-05-01 21:00:00', '2025-05-02 01:00:00', '2025-05-02 05:00:00',
        '2025-05-10 09:00:00', '2025-05-07 13:00:00', '2025-05-02 17:00:00',
        '2025-05-02 21:00:00', '2025-05-03 01:00:00', '2025-05-03 05:00:00',
        '2025-05-03 09:00:00', '2025-05-03 13:00:00', '2025-05-03 17:00:00',
        '2025-05-03 15:00:00', '2025-05-05 01:00:00', '2025-05-04 09:00:00',
        '2025-05-04 05:00:00', '2025-05-04 13:00:00'
    ],
    'domain_ids': [
        '6', '3', '2', '1', '5',
        '2,5,6',  # multi
        '1', '3,6,4',  # multi
        '4', '5',
        '2', '1,2,7',  # multi
        '6', '3,5,4',  # multi
        '3', '7',
        '4',  # patient 2 multi shown above keeps balance
        '5', '6',  # single-domain rows
        '7,4,2'   # multi
    ],
    'domain_scores': [
        '0.64', '0.79', '0.98', '0.36', '0.62',
        '0.92,0.37,0.69',
        '0.82', '0.34,0.35,0.64',
        '0.58', '0.86',
        '0.92', '0.50,0.73,0.96',
        '0.35', '0.61,0.70,0.66',
        '0.57', '0.93',
        '0.48',
        '0.66', '0.72',
        '0.81,0.54,0.42'
    ]}

    df = pd.DataFrame(data)
    df = preprocessing.sort_by_start_time(df)
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

    session_data = df.groupby("patient_id")[df.columns].apply(preprocessing.extract_session_data).reset_index(drop=True)
    print(session_data)
    # Check if the session data is extracted correctly
    assert isinstance(session_data, pd.DataFrame)
    expected_cols = (
    ["patient_id"]
    + ["domain %d encoding" % i for i in range(1, 15)]
    + ["domain %d score" % i for i in range(1, 15)]
    + ["domain %d target" % i for i in range(1, 15)]
    + ["start_time"]
    + ["time_stamp"]
    )
    assert session_data.shape == (20, 45)  # 20 rows, 45 columns
    assert session_data.columns.tolist() == expected_cols

    ## check some values
    assert session_data.iloc[0]["domain 6 encoding"] == 1
    assert np.isnan(session_data.iloc[0]["domain 6 score"])
    assert session_data.iloc[0]["domain 6 target"] == 0.64
    assert session_data.iloc[1]["domain 6 score"] == 0.64
    assert session_data.iloc[1]["domain 6 encoding"] == 0

    assert session_data.iloc[1]["domain 3 encoding"] == 1
    assert session_data.iloc[2]["domain 3 score"] == 0.79
    assert session_data.iloc[2]["domain 3 encoding"] == 0
    assert session_data.iloc[2]["domain 2 encoding"] == 1

    assert session_data.iloc[13]["domain 1 encoding"] == 1
    assert session_data.iloc[13]["domain 1 target"] == 0.82
    assert session_data.iloc[13]["domain 1 score"] == 0.36

    assert np.isnan(session_data.iloc[14]["domain 2 score"])
    assert np.isnan(session_data.iloc[14]["domain 5 score"])
    assert np.isnan(session_data.iloc[14]["domain 6 score"])

    assert session_data.iloc[15]["domain 2 score"] == 0.92
    assert session_data.iloc[15]["domain 5 score"] == 0.37
    assert session_data.iloc[15]["domain 6 score"] == 0.69
    assert session_data.iloc[14]["start_time"] == pd.to_datetime('2025-05-02 05:00:00')

    # check that patient_id is preserved and that the returned dataframe is in order of patient_id
    assert session_data["patient_id"].iloc[0] == 1
    assert session_data["patient_id"].iloc[14] == 2



def test_filter_datetime_outliers():
    pass

def test_datetime_outlier():
    pass