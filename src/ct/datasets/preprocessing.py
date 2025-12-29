### File contains functions for preprocessing data
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from typing import Dict, Any, Tuple, List, Optional

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
    df = df.sort_values(by='start_time', inplace=False).reset_index(drop=True)
    return df


def find_usage_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of all considered sessions for a patient, find usage frequency in terms of unique days,
    return a dataframe with patient_id, usage_time, usage_freq, and unique_days.

    Usage time is defined as the span from the first to last session (inclusive).
    Unique days is defined as the number of distinct days the patient has practiced.
    Usage frequency is defined as unique days divided by usage time.

    Parameters:
        df (pd.DataFrame): The DataFrame containing session data with 'patient_id' and 'start_time' columns.

    Returns:
        pd.DataFrame: DataFrame with 'patient_id', 'usage_time', 'unique_days', and 'usage_freq'.
    """
    df = sort_by_start_time(df)
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
    assert 'domain_ids' in row and 'domain_scores' in row, "Row must contain 'domain_ids' and 'domain_scores' columns."
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
        pd.DataFrame: A DataFrame with patient_id, domain encodings, previous scores, current scores, and start time. Information are recorded in this order for a total of 45 columns."""
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

        assert len(session_row) == 45, f"Expected 45 columns, got {len(session_row)}"

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
    Convert specified columns in a DataFrame to percentiles. Specified columns are usually domain scores and targets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to convert to percentiles. Default to domain scores and targets.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to percentiles.
    """
    for col in columns:
        df[col] = df[col].rank(pct=True)
    return df

class WeeklyHistoryPairEncoder:
    """
    Weekly version of the preprocessor:
      - Groups by (user, week_start, domain), aggregates by mean (configurable)
      - Builds dense weekly timelines per user (first_active_week .. last_active_week)
      - Encodes per-domain scores as pairs: (present, 1 - x), missing as (0, 0)
      - Weekly covariates: week_index, weeks_since_last_active, n_sessions_week, n_domains_week, week_of_year
    """

    def __init__(self, config: Dict[str, Any]):
        time_cfg = config["time"]
        self.user_col = "patient_id"
        self.time_col = "start_time"
        self.domain_ids_col = "domain_ids"
        self.domain_scores_col = "domain_scores"

        self.week_freq = time_cfg["week_freq"]     # e.g., 'W-MON'
        self.agg = time_cfg["agg"]
        self.missing_code = time_cfg["missing_code"]
        assert self.missing_code in {"00", "11"}

        self.users_: Optional[List[Any]] = None
        self.user_to_idx_: Optional[Dict[Any, int]] = None
        self.domains_: Optional[List[int]] = None
        self.domain_to_idx_: Optional[Dict[int, int]] = None

        self.covariate_names_ = config["features"]["covariates"]

    # --- helpers ---

    def _add_week_start(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        # Convert to period by week, then to week's start timestamp
        df["week_start"] = df[self.time_col].dt.to_period(self.week_freq).dt.start_time
        return df

    def _parse_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def split_clean(s: Any) -> List[str]:
            if pd.isna(s):
                return []
            return [tok.strip() for tok in str(s).split(",") if tok.strip() != ""]

        ids = df[self.domain_ids_col].apply(split_clean)
        scs = df[self.domain_scores_col].apply(split_clean)
        valid_len = ids.str.len() == scs.str.len()
        df = df[valid_len].copy()
        ids = ids[valid_len]
        scs = scs[valid_len]

        df["__id_list"] = ids
        df["__score_list"] = scs

        df = df.explode(["__id_list", "__score_list"]).reset_index(drop=True)
        if df.empty:
            df["domain_id"] = pd.Series(dtype=int)
            df["domain_score"] = pd.Series(dtype=float)
            return df

        df["domain_id"] = df["__id_list"].astype(int)
        df["domain_score"] = df["__score_list"].astype(float)
        df = df.drop(columns=["__id_list", "__score_list"])
        df["domain_score"] = df["domain_score"].clip(0.0, 1.0)
        return df

    def _aggregate(self, df_parsed: pd.DataFrame) -> pd.DataFrame:
        g = df_parsed.groupby([self.user_col, "week_start", "domain_id"])["domain_score"]
        if self.agg == "mean": out = g.mean()
        elif self.agg == "last": out = g.last()
        elif self.agg == "max": out = g.max()
        elif self.agg == "median": out = g.median()
        else: raise ValueError(f"Unsupported agg='{self.agg}'")
        return out.reset_index()

    # --- public API ---

    def fit(self, df: pd.DataFrame):
        df = self._add_week_start(df)
        df = self._parse_rows(df)

        self.users_ = sorted(df[self.user_col].unique().tolist())
        self.user_to_idx_ = {u: i for i, u in enumerate(self.users_)}

        self.domains_ = sorted(df["domain_id"].unique().tolist())
        self.domain_to_idx_ = {d: i for i, d in enumerate(self.domains_)}

        return self

    def transform(self, df: pd.DataFrame):
        """
        Consolidate sessions per week and create pair-encoded arrays + covariates.

        Returns:
        E_pairs:      [U, T_max, K, 2]
        E_pairs_flat: [U, T_max, 2K]
        Y:            [U, T_max, K]
        M_target:     [U, T_max, K]
        X_week:       [U, T_max, D_x]
        meta:         dict with users, domains, weeks_per_user, valid_lengths, feature names
        """
        assert self.users_ is not None and self.domains_ is not None, "Call fit() first."

        users, domains = self.users_, self.domains_
        K = len(domains)

        df = self._add_week_start(df)
        dfp = self._parse_rows(df)
        agg = self._aggregate(dfp)  # columns: [patient_id, week_start, domain_id, domain_score]

        # Weekly counts for covariates
        session_counts = (
            dfp.groupby([self.user_col, "week_start"])
            .size()
            .rename("n_sessions_week")
            .reset_index()
        )
        domain_counts = (
            agg.groupby([self.user_col, "week_start"])["domain_id"]
            .nunique()
            .rename("n_domains_week")
            .reset_index()
        )
        counts = pd.merge(session_counts, domain_counts,
                        on=[self.user_col, "week_start"], how="outer")

        # Per-user dense weekly timelines (min..max week_start in agg)
        user_weeks: Dict[Any, pd.DatetimeIndex] = {}
        max_T = 0
        for u in users:
            g_u = agg[agg[self.user_col] == u]
            if g_u.empty:
                user_weeks[u] = pd.DatetimeIndex([])
                continue
            w_min, w_max = g_u["week_start"].min(), g_u["week_start"].max()
            weeks = pd.date_range(w_min, w_max, freq="7D")
            user_weeks[u] = weeks
            max_T = max(max_T, len(weeks))

        U = len(users)
        # allocate arrays
        E_pairs = np.zeros((U, max_T, K, 2), dtype=np.float32)
        Y = np.full((U, max_T, K), np.nan, dtype=np.float32)
        M_target = np.zeros((U, max_T, K), dtype=np.float32)

        D_x = len(self.covariate_names_)
        X_week = np.zeros((U, max_T, D_x), dtype=np.float32)

        miss_pair = (
            np.array([0.0, 0.0], dtype=np.float32)
            if self.missing_code == "00"
            else np.array([1.0, 1.0], dtype=np.float32)
        )

        counts_idx = counts.set_index([self.user_col, "week_start"])

        # ---------- FILL PER USER ----------
        for u_idx, u in enumerate(users):
            weeks = user_weeks[u]
            T_u = len(weeks)
            if T_u == 0:
                continue

            # Start with all missing
            E_pairs[u_idx, :T_u, :, :] = miss_pair

            # Slice aggregated data for this user
            g_u = agg[agg[self.user_col] == u]

            if not g_u.empty:
                # Pivot to [week_start x domain_id] table
                pivot = g_u.pivot(index="week_start", columns="domain_id", values="domain_score")

                # Reindex rows to full weekly grid, columns to full domain list
                pivot = pivot.reindex(index=weeks, columns=domains)

                # Extract values as array [T_u, K]
                arr = pivot.to_numpy(dtype=float)  # NaN if no observation
                # Fill Y and M_target
                Y[u_idx, :T_u, :] = arr
                M = ~np.isnan(arr)
                M_target[u_idx, :T_u, :] = M.astype(np.float32)

                # For observed entries, set pair encoding (present=1, inv=1-x)
                pres = M.astype(np.float32)
                # For inv: 1 - x, but only where observed; where missing, will be overwritten by miss_pair
                x_filled = np.nan_to_num(arr, nan=0.0)
                inv = 1.0 - x_filled
                E_pairs[u_idx, :T_u, :, 0] = pres
                E_pairs[u_idx, :T_u, :, 1] = inv
                # For missing entries, override back to miss_pair
                # (optional: but pres==0 already encodes missing; miss_pair keeps it consistent)
                missing_mask = ~M
                E_pairs[u_idx, :T_u, :, 0][missing_mask] = miss_pair[0]
                E_pairs[u_idx, :T_u, :, 1][missing_mask] = miss_pair[1]

            # Weekly covariates
            last_active_week: Optional[pd.Timestamp] = None
            for t_idx, w in enumerate(weeks):
                # defaults
                week_index = float(t_idx)
                iso_week = int(pd.Timestamp(w).isocalendar().week)

                if (u, w) in counts_idx.index:
                    row = counts_idx.loc[(u, w)]
                    n_sessions_week = float(row["n_sessions_week"]) if "n_sessions_week" in row else 0.0
                    n_domains_week = float(row["n_domains_week"]) if "n_domains_week" in row else 0.0
                else:
                    n_sessions_week, n_domains_week = 0.0, 0.0

                if n_sessions_week > 0:
                    weeks_since_last_active = 0.0 if last_active_week is None else float((w - last_active_week).days // 7)
                    last_active_week = w
                else:
                    weeks_since_last_active = (
                        np.nan if last_active_week is None else float((w - last_active_week).days // 7)
                    )

                X_week[u_idx, t_idx, :] = np.array(
                    [
                        week_index,
                        0.0 if np.isnan(weeks_since_last_active) else weeks_since_last_active,
                        n_sessions_week,
                        n_domains_week,
                        float(iso_week),
                    ],
                    dtype=np.float32,
                )

        E_pairs_flat = E_pairs.reshape(U, max_T, 2 * K)

        meta = {
            "users": users,
            "domains": domains,
            "weeks_per_user": [user_weeks[u] for u in users],
            "valid_lengths": [len(user_weeks[u]) for u in users],
            "x_week_feature_names": self.covariate_names_,
            "week_freq": self.week_freq,
        }
        return E_pairs, E_pairs_flat, Y, M_target, X_week, meta

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)