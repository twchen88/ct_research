import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def plot_random_user_weekly_global(meta, Y, M_target, min_weeks=4):
    users = meta["users"]
    weeks_per_user = meta["weeks_per_user"]

    # Build a list of candidate users with enough weeks and at least one observed week
    candidate_idxs = []
    for u_idx, weeks in enumerate(weeks_per_user):
        T_u = len(weeks)
        if T_u < min_weeks:
            continue

        y_u = Y[u_idx, :T_u, :]
        m_u = M_target[u_idx, :T_u, :]
        m_sum = m_u.sum(axis=1)
        g_series = np.where(m_sum > 0, (y_u * m_u).sum(axis=1) / m_sum, np.nan)

        if not np.all(np.isnan(g_series)):
            candidate_idxs.append(u_idx)

    if not candidate_idxs:
        raise ValueError(f"No users found with at least {min_weeks} weeks and non-NaN scores.")

    # Pick one at random
    u_idx = int(np.random.choice(candidate_idxs))
    weeks = weeks_per_user[u_idx]
    T_u = len(weeks)

    y_u = Y[u_idx, :T_u, :]
    m_u = M_target[u_idx, :T_u, :]
    m_sum = m_u.sum(axis=1)
    g_series = np.where(m_sum > 0, (y_u * m_u).sum(axis=1) / m_sum, np.nan)

    print(f"Chosen user index: {u_idx}, user id: {users[u_idx]}, weeks: {T_u}")

    plt.figure(figsize=(8, 4))
    plt.plot(weeks, g_series, marker="o")
    plt.title(f"User {users[u_idx]} — Weekly global score")
    plt.xlabel("Week")
    plt.ylabel("Mean of observed domain scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_random_patient_domain_avg(
    df: pd.DataFrame,
    patient_key: str = "patient_id",   # can be a column or an index level name
    week_key: str = "week_number",     # can be a column or an index level name
    seed: int | None = None,
):
    """
    Choose a random patient and plot, by week, the per-week mean of domain_{i}_avg
    values, excluding domain i for a row iff (domain_{i}_avg==0 and domain_{i}_inv==0).

    Works when patient_key/week_key are columns OR index levels (including MultiIndex).
    Returns (patient_id, weekly_df).
    """
    d0 = df.copy()

    # --- Ensure patient/week are columns (handle MultiIndex / index levels) ---
    need_reset = False
    for key in (patient_key, week_key):
        if key not in d0.columns:
            if (d0.index.name == key) or (hasattr(d0.index, "names") and key in d0.index.names):
                need_reset = True
            else:
                raise KeyError(f"'{key}' not found as a column or index level.")
    if need_reset:
        d0 = d0.reset_index()

    # --- Find domain avg/inv pairs ---
    avg_cols = [c for c in d0.columns if re.fullmatch(r"domain_\d+_avg", str(c))]
    if not avg_cols:
        raise ValueError("No columns matching 'domain_{i}_avg' found.")

    pairs = []
    for avg in avg_cols:
        inv = re.sub(r"_avg$", "_inv", avg)
        if inv not in d0.columns:
            raise ValueError(f"Missing matching inv column for {avg} (expected {inv}).")
        pairs.append((avg, inv))

    # --- Choose random patient ---
    rng = np.random.default_rng(seed)
    patients = d0[patient_key].dropna().unique()
    if len(patients) == 0:
        raise ValueError("No patients found.")
    pid = rng.choice(patients)

    d = d0.loc[d0[patient_key] == pid].copy()

    # --- Compute per-row mean across included domains ---
    included_vals = []
    for avg, inv in pairs:
        include = ~((d[avg] == 0) & (d[inv] == 0))
        included_vals.append(d[avg].where(include, np.nan))

    d["domain_avg_mean"] = pd.concat(included_vals, axis=1).mean(axis=1, skipna=True)

    # --- Aggregate by week and plot ---
    weekly = (
        d.groupby(week_key, as_index=False)["domain_avg_mean"]
         .mean()
         .sort_values(week_key)
         .dropna(subset=["domain_avg_mean"])
    )

    plt.figure()
    plt.plot(weekly[week_key], weekly["domain_avg_mean"], marker="o")
    plt.title(f"Patient {pid}: mean(domain_i_avg) by week (excluding avg=inv=0)")
    plt.xlabel(week_key)
    plt.ylabel("Mean of included domain avg values")
    plt.grid(True)
    plt.show()

    return pid, weekly