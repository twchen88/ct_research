import pandas as pd


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


# --- Example usage ---
# weekly_df = HistoryEncoder(raw_data).transform()
# filtered = filter_users_by_usage(weekly_df, min_sessions_per_week=2, min_weeks=6, require_consecutive=True)