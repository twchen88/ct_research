import pandas as pd
import numpy as np
from typing import Optional


def _ensure_history_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df is indexed by (patient_id, week_number) and sorted.
    """
    if isinstance(df.index, pd.MultiIndex) and df.index.names[:2] == ["patient_id", "week_number"]:
        out = df.copy()
    elif {"patient_id", "week_number"}.issubset(df.columns):
        out = df.set_index(["patient_id", "week_number"]).copy()
    else:
        raise ValueError("Expected MultiIndex (patient_id, week_number) or columns patient_id, week_number.")
    return out.sort_index()

def densify_week_grid_ffill_scores(
    df: pd.DataFrame,
    freq_cols: list[str],
    avg_cols: list[str],
    inv_cols: list[str],
) -> pd.DataFrame:
    """
    Densify each patient to full week grid [min_week..max_week].

    Gap weeks:
      - freq := 0
      - avg/inv := forward-filled (stable)
      - domain_i_obs := 1 if that domain had a real value that week, else 0

    Note:
      - We define "observed" BEFORE ffill: if avg/inv existed in original data for that week.
      - For leading missing values (before first observation), we set avg/inv=0 and obs=0.
    """
    df = _ensure_history_index(df).copy()

    # Ensure float for NaN handling
    df[freq_cols] = df[freq_cols].astype(float)
    df[avg_cols]  = df[avg_cols].astype(float)
    df[inv_cols]  = df[inv_cols].astype(float)

    obs_cols = [c.replace("_avg", "_obs") for c in avg_cols]
    out_parts = []

    for pid, sub in df.groupby(level=0, sort=False):
        sub = sub.droplevel(0).sort_index()  # index: week_number

        wmin, wmax = int(sub.index.min()), int(sub.index.max())
        full_weeks = pd.Index(range(wmin, wmax + 1), name="week_number")

        dense = sub.reindex(full_weeks)  # introduces NaNs for gap weeks

        # Observed mask BEFORE ffill: observed if avg or inv not NaN that week (per domain)
        obs = (~dense[avg_cols].isna()) | (~dense[inv_cols].isna())   # [T,K] bool
        dense[obs_cols] = obs.astype(np.float32)

        # Fill gaps: freq=0
        dense[freq_cols] = dense[freq_cols].fillna(0.0)

        # Forward-fill scores (stable through gaps)
        dense[avg_cols] = dense[avg_cols].ffill()
        dense[inv_cols] = dense[inv_cols].ffill()

        # Leading NaNs (before first observation) -> 0, and obs already 0 there
        dense[avg_cols] = dense[avg_cols].fillna(0.0)
        dense[inv_cols] = dense[inv_cols].fillna(0.0)

        dense["patient_id"] = pid
        dense = dense.reset_index().set_index(["patient_id", "week_number"])
        out_parts.append(dense)

    out = pd.concat(out_parts, axis=0).sort_index()
    return out

def filter_users_by_usage(
    weekly_df: pd.DataFrame,
    min_sessions_per_week: int = 1,
    min_weeks: int = 4,
    freq_cols: list[str] | None = None,
    require_consecutive: bool = True,
) -> pd.DataFrame:
    """
    Filter HistoryEncoder.transform() output by:
      1) usage frequency: user must have >= min_sessions_per_week in every qualifying week
      2) usage time: user must have records for >= min_weeks weeks

    Assumptions:
      - weekly_df is the wide weekly output: one row per (patient_id, week_number)
      - session count per week is approximated as sum of domain_*_freq columns
        (this matches the encoder's definition: freq counts domain occurrences, not necessarily sessions).
        If you want "sessions" exactly, include a session_id in raw data and encode it separately.

    Params:
      freq_cols: optionally provide which columns to use as frequency columns.
                Default: all columns ending in "_freq".
      require_consecutive:
        - True (default): checks the criteria over a consecutive block of weeks of length min_weeks,
          starting at week 0 up to the user's max observed week (missing weeks break the streak).
        - False: checks criteria over the user's observed weeks only (ignores gaps).

    Returns:
      Filtered weekly_df containing only users who pass, with original indexing preserved.
    """
    df = _ensure_history_index(weekly_df)

    if freq_cols is None:
        freq_cols = [c for c in df.columns if c.endswith("_freq")]
    if not freq_cols:
        raise ValueError("No frequency columns found. Pass freq_cols explicitly.")

    # total "usage count" per week (see note in docstring)
    usage_per_week = df[freq_cols].sum(axis=1)

    # Build per-user masks
    keep_users = []

    for pid, g in usage_per_week.groupby(level=0, sort=False):
        s = g.droplevel(0).sort_index()  # index = week_number

        if require_consecutive:
            # Reindex to full consecutive weeks from 0..max_week (missing => 0 usage)
            full_weeks = pd.RangeIndex(0, int(s.index.max()) + 1 if len(s) else 0)
            s_full = s.reindex(full_weeks, fill_value=0)

            # Need at least min_weeks overall span
            if len(s_full) < min_weeks:
                continue

            # Find any consecutive window of length min_weeks where all weeks meet min_sessions_per_week
            meets = (s_full >= min_sessions_per_week).astype(int)
            # rolling sum == window size means all True in that window
            ok_any_window = (meets.rolling(min_weeks).sum() == min_weeks).any()
            if not ok_any_window:
                continue

            keep_users.append(pid)

        else:
            # Only consider observed weeks (gaps ignored)
            if len(s) < min_weeks:
                continue
            if (s >= min_sessions_per_week).sum() < min_weeks:
                continue
            # Optional stricter: require every observed week meets threshold
            if (s < min_sessions_per_week).any():
                continue

            keep_users.append(pid)

    filtered = df.loc[df.index.get_level_values(0).isin(keep_users)].copy()
    return filtered


def filter_patients_allow_gaps_with_cap(
    df: pd.DataFrame,
    obs_cols: list[str],
    lookback_weeks: int = 12,
    min_observed_target_weeks: int = 8,
    max_gap_weeks: int = 8,
) -> pd.DataFrame:
    """
    Keep patients who:
      - have at least lookback+1 weeks after densify
      - have at least min_observed_target_weeks where ANY domain is observed
      - do not have a consecutive gap run longer than max_gap_weeks
    """
    df = _ensure_history_index(df)

    keep = []
    for pid, sub in df.groupby(level=0, sort=False):
        sub = sub.droplevel(0).sort_index()
        T = len(sub)
        if T < lookback_weeks + 1:
            continue

        # week observed if any domain observed
        week_obs = (sub[obs_cols].to_numpy(np.float32).sum(axis=1) > 0)  # [T] bool
        if int(week_obs.sum()) < min_observed_target_weeks:
            continue

        # max consecutive gaps
        gaps = (~week_obs).astype(np.int32)
        max_run = 0
        run = 0
        for g in gaps:
            if g == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > max_gap_weeks:
            continue

        keep.append(pid)

    return df.loc[df.index.get_level_values(0).isin(keep)].copy()