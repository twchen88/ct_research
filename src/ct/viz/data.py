import matplotlib.pyplot as plt
import pandas as pd

"""
src/viz/data.py
----------------------------
This module provides plotting functions that are specific to the src/data/ related sections of the prediction project.
"""

def plot_filtering_visualization(df: pd.DataFrame, sessions_filter_df: pd.DataFrame, pid: str):
    """
    Visualize the filtering process of sessions for a specific patient. Use in a notebook for data visualization and exploration purposes.

    Parameters:
        df (pd.DataFrame): Original dataframe containing all sessions.
        sessions_filter_df (pd.DataFrame): Filtered dataframe after applying session filtering.
        pid (str): Patient ID to visualize.
    """
    df1 = df[df.patient_id == pid]
    df2 = sessions_filter_df[sessions_filter_df.patient_id == pid]

    # Convert start_time to just the date (drop time details)
    df1['date'] = df1['start_time'].dt.date
    df2['date'] = df2['start_time'].dt.date

    # Find missing dates
    missing_dates = set(df1['date']) - set(df2['date'])

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot all dates from df1
    ax.scatter(df1['date'], [1] * len(df1), label='df1 (Original)', color='blue', marker='o')

    # Plot retained dates from df2
    ax.scatter(df2['date'], [2] * len(df2), label='df2 (Filtered)', color='green', marker='o')

    # Highlight missing dates
    for missing in missing_dates:
        ax.scatter(missing, 1, color='red', marker='x', label="Removed" if missing == list(missing_dates)[0] else "")

    # Formatting
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Original", "Filtered"])
    ax.set_xticklabels(df1['date'], rotation=45, ha='right')
    plt.xlabel("Date")
    plt.ylabel("Timeline")
    plt.title("Visualization of Session Filtering")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()
    plt.close()