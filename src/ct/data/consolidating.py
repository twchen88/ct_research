from __future__ import annotations

import ast
import numpy as np
import pandas as pd
from typing import Any, List

"""
src/ct/data/history.py
-----------------------------
This module provides a HistoryEncoder class that takes raw data from SQL database and transforms it into a structured format as described below:
1. sessions consolidated into weekly records
2. domains practiced encoded by frequency in that week
3. performance metrics averaged over the week
4. domains that were not practiced represented with (0, 0) for performance metrics; domains practiced represented with (avg_score, 1 - avg_score)
5. weekly records sorted by timestamp
6. add week number column, week numbers from each user's first week starting at 0 (does not use a global week number)
7. output as a pandas DataFrame with multi-index (user_id, week_number)??
"""

def _to_list(x: Any) -> List[Any]:
    """
    Best-effort conversion of a cell into a Python list.
    Handles:
      - already-a-list/tuple/np.ndarray
      - stringified list like "[1, 2]" or "['a','b']"
      - comma-separated strings like "1,2,3"
      - nulls -> []
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Try literal_eval for JSON-ish / python list strings
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray)):
                return list(v)
            # single item like "3"
            return [v]
        except Exception:
            # fallback: comma-separated
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return parts
    # scalar -> [scalar]
    return [x]


def _coerce_numeric_list(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            # drop unparseable entries
            continue
    return out

def _latest_domain_score_per_week(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input long_df columns (from consolidate):
      patient_id, week_number, week_start_ts, start_ts, domain_id, domain_score

    Output:
      patient_id, week_number, week_start_ts, domain_id, latest_score
    """
    # sort so "last" is the most recent within the week
    tmp = long_df.sort_values(["patient_id", "week_number", "domain_id", "start_ts"])
    latest = (
        tmp.groupby(["patient_id", "week_number", "week_start_ts", "domain_id"], as_index=False)
           .tail(1)
           .rename(columns={"domain_score": "latest_score"})
    )
    return latest[["patient_id", "week_number", "week_start_ts", "domain_id", "latest_score"]]


def _avg_domain_score_per_week(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input long_df columns:
      patient_id, week_number, week_start_ts, domain_id, domain_score

    Output:
      patient_id, week_number, week_start_ts, domain_id, week_score
    where week_score is the average score for that domain within that week.
    """
    avg = (
        long_df.groupby(["patient_id", "week_number", "week_start_ts", "domain_id"], as_index=False)
              .agg(week_score=("domain_score", "mean"))
    )
    return avg

def _forward_fill_history(avg_long: pd.DataFrame, all_weeks: pd.DataFrame) -> pd.DataFrame:
    """
    avg_long:
      patient_id, week_number, week_start_ts, domain_id, week_score  (avg within week)

    all_weeks:
      patient_id, week_number, week_start_ts  (includes missing weeks)
    """
    domains = pd.DataFrame({"domain_id": avg_long["domain_id"].unique()})
    patients = all_weeks[["patient_id"]].drop_duplicates()

    # patient x domain grid
    grid = patients.merge(domains, how="cross")
    grid = grid.merge(all_weeks, on="patient_id", how="left")

    # join weekly averaged scores
    grid = grid.merge(
        avg_long,
        on=["patient_id", "week_number", "week_start_ts", "domain_id"],
        how="left",
    )

    grid = grid.sort_values(["patient_id", "domain_id", "week_number"])

    # forward-fill last known weekly score
    grid["score_ffill"] = grid.groupby(["patient_id", "domain_id"], sort=False)["week_score"].ffill()

    # Typed-safe replacement for aggfunc="last"
    def _last(s: pd.Series) -> float:
        # pivot_table passes a Series for each group
        return s.iloc[-1]


    avg_wide = grid.pivot_table(
        index=["patient_id", "week_number"],
        columns="domain_id",
        values="score_ffill",
        aggfunc=_last
    ).sort_index(axis=1)

    inv_wide = 1.0 - avg_wide

    avg_wide.columns = [f"domain_{d}_avg" for d in avg_wide.columns.astype(str)]
    inv_wide.columns = [f"domain_{d}_inv" for d in inv_wide.columns.astype(str)]

    week_ts = all_weeks.set_index(["patient_id", "week_number"])[["week_start_ts"]]

    return week_ts.join(avg_wide, how="left").join(inv_wide, how="left")

def _make_all_weeks(long_df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "week_number", "week_start_ts"}
    if not required.issubset(long_df.columns):
        raise ValueError(f"long_df must contain {required}, got {set(long_df.columns)}")

    # 1) per-patient max week (guaranteed column name)
    max_weeks = (
        long_df.groupby("patient_id", as_index=False)
              .agg(max_week=("week_number", "max"))
    )

    # 2) expand to all weeks 0..max_week
    parts = []
    for pid, mx in zip(max_weeks["patient_id"].tolist(), max_weeks["max_week"].tolist()):
        mx = int(mx)
        parts.append(pd.DataFrame({"patient_id": pid, "week_number": np.arange(mx + 1, dtype=int)}))
    all_weeks = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["patient_id", "week_number"])

    # 3) get each patient's week-0 anchor timestamp with an explicit named agg
    first_week_start = (
        long_df.groupby("patient_id", as_index=False)
              .agg(first_week_start_ts=("week_start_ts", "min"))
    )

    # 4) merge and compute week_start_ts for every week_number
    all_weeks = all_weeks.merge(first_week_start, on="patient_id", how="left")

    all_weeks["week_start_ts"] = all_weeks["first_week_start_ts"] + pd.to_timedelta(
        all_weeks["week_number"] * 7, unit="D"
    )

    # keep it clean
    all_weeks = all_weeks.drop(columns=["first_week_start_ts"])

    return all_weeks.sort_values(["patient_id", "week_number"]).reset_index(drop=True)


def _align_freq_to_all_weeks(freq_df: pd.DataFrame, all_weeks: pd.DataFrame) -> pd.DataFrame:
    """
    freq_df: output of encode_domains(long_df)
      index: (patient_id, week_number)
      columns: week_start_ts + domain_*_freq

    all_weeks:
      columns: patient_id, week_number, week_start_ts  (includes missing weeks)

    Returns:
      A frequency df with one row per (patient_id, week_number) in all_weeks
      Missing weeks filled with 0s for *_freq.
    """
    # Base index from all_weeks
    base = all_weeks.set_index(["patient_id", "week_number"])[["week_start_ts"]].copy()

    # Join freq columns; missing weeks become NaN -> fill with 0
    freq_cols = [c for c in freq_df.columns if c.endswith("_freq")]

    # Make sure we don't duplicate week_start_ts on join
    freq_only = freq_df.drop(columns=["week_start_ts"], errors="ignore")

    out = base.join(freq_only, how="left")

    if freq_cols:
        out[freq_cols] = out[freq_cols].fillna(0).astype(int)
    else:
        # no freq cols found -> still return base
        pass

    return out


class HistoryEncoder:
    """
    Transforms raw SQL session rows into per-user weekly records:

    1) sessions consolidated into weekly records (7-day windows from each user's first session)
    2) domain practice encoded by frequency per week (wide columns)
    3) domain performance averaged per week
    4) per-domain performance encoded as (avg_score, 1-avg_score); missing domains => (0,0)
    5) weekly records sorted by timestamp
    6) week_number starts at 0 for each user (relative to their first session)
    7) output DataFrame indexed by (patient_id, week_number)
    """

    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data.copy()

    def consolidate(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["patient_id", "domain_ids", "domain_scores", "start_time"]
        base = df.loc[:, [c for c in needed if c in df.columns]].copy()

        missing = [c for c in needed if c not in base.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse timestamps (your start_time is datetime-like strings)
        base["start_ts"] = pd.to_datetime(base["start_time"], utc=True, errors="coerce")
        base = base.dropna(subset=["start_ts"])

        # Normalize lists
        base["domain_ids_list"] = base["domain_ids"].apply(_to_list)
        base["domain_scores_list"] = base["domain_scores"].apply(_to_list)
        base["domain_scores_list"] = base["domain_scores_list"].apply(_coerce_numeric_list)

        # Align lengths per row
        def _align(row: pd.Series):
            ids = row["domain_ids_list"]
            scores = row["domain_scores_list"]
            m = min(len(ids), len(scores))
            return (ids[:m], scores[:m]) if m > 0 else ([], [])

        aligned = base.apply(_align, axis=1, result_type="expand")
        base["domain_ids_list"] = aligned[0]
        base["domain_scores_list"] = aligned[1]
        base = base[base["domain_ids_list"].map(len) > 0].copy()

        # Per-user relative week_number
        base = base.sort_values(["patient_id", "start_ts"])
        first_ts = base.groupby("patient_id", sort=False)["start_ts"].transform("min")
        delta_days = (base["start_ts"] - first_ts) / pd.Timedelta(days=1)
        base["week_number"] = np.floor(delta_days / 7.0).astype(int)
        base["week_start_ts"] = first_ts + pd.to_timedelta(base["week_number"] * 7, unit="D")

        # Explode into long format AND KEEP start_ts
        long_df = base[
            ["patient_id", "week_number", "week_start_ts", "start_ts", "domain_ids_list", "domain_scores_list"]
        ].copy()

        long_df = long_df.explode(["domain_ids_list", "domain_scores_list"], ignore_index=True)
        long_df = long_df.rename(columns={"domain_ids_list": "domain_id", "domain_scores_list": "domain_score"})

        # domain_id is integer; keep as int if possible, else coerce
        long_df["domain_id"] = pd.to_numeric(long_df["domain_id"], errors="coerce").astype("Int64")
        long_df["domain_score"] = pd.to_numeric(long_df["domain_score"], errors="coerce")
        long_df = long_df.dropna(subset=["domain_id", "domain_score"])
        long_df["domain_id"] = long_df["domain_id"].astype(int)

        return long_df

    def encode_domains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes long df from consolidate() and returns weekly wide frequency columns:
          index: (patient_id, week_number)
          columns: week_start_ts, domain_<id>_freq ...
        """
        # count occurrences per domain per week
        counts = (
            df.groupby(["patient_id", "week_number", "week_start_ts", "domain_id"], as_index=False)
              .size()
              .rename(columns={"size": "freq"})
        )

        freq_wide = (
            counts.pivot_table(
                index=["patient_id", "week_number"],
                columns="domain_id",
                values="freq",
                fill_value=0,
                aggfunc="sum",
            )
            .sort_index(axis=1)
        )

        # rename domain columns
        freq_wide.columns = [f"domain_{d}_freq" for d in freq_wide.columns.astype(str)]

        # attach week_start_ts (unique per patient/week by construction)
        week_ts = (
            df.groupby(["patient_id", "week_number"], as_index=False)["week_start_ts"]
              .min()
              .set_index(["patient_id", "week_number"])
        )

        out = week_ts.join(freq_wide, how="left").fillna(0)
        return out

    def average_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes long df from consolidate() and returns weekly wide performance columns:
          index: (patient_id, week_number)
          columns: week_start_ts, domain_<id>_avg, domain_<id>_inv ...
        Missing domains are represented as (0,0) after join/fill in transform().
        """
        means = (
            df.groupby(["patient_id", "week_number", "week_start_ts", "domain_id"], as_index=False)["domain_score"]
              .mean()
              .reset_index()
              .rename(columns={"domain_score": "avg_score"})
        )


        avg_wide = (
            means.pivot_table(
                index=["patient_id", "week_number"],
                columns="domain_id",
                values="avg_score",
                aggfunc="mean",
            )
            .sort_index(axis=1)
        )

        inv_wide = 1.0 - avg_wide

        # rename columns into separate blocks
        avg_wide.columns = [f"domain_{d}_avg" for d in avg_wide.columns.astype(str)]
        inv_wide.columns = [f"domain_{d}_inv" for d in inv_wide.columns.astype(str)]

        week_ts = (
            df.groupby(["patient_id", "week_number"], as_index=False)["week_start_ts"]
              .min()
              .set_index(["patient_id", "week_number"])
        )

        out = week_ts.join(avg_wide, how="left").join(inv_wide, how="left")
        return out

    def transform(self) -> pd.DataFrame:
        long_df = self.consolidate(self.raw_data)

        all_weeks = _make_all_weeks(long_df)

        # Frequency (fill missing weeks with 0)
        freq_df_observed = self.encode_domains(long_df)
        freq_df = _align_freq_to_all_weeks(freq_df_observed, all_weeks)

        # Performance: average within week, then carry forward across weeks
        avg_long = _avg_domain_score_per_week(long_df)
        perf_hist_df = _forward_fill_history(avg_long, all_weeks)

        out = freq_df.drop(columns=["week_start_ts"], errors="ignore").join(perf_hist_df, how="outer")

        freq_cols = [c for c in out.columns if c.endswith("_freq")]
        out[freq_cols] = out[freq_cols].fillna(0).astype(int)

        perf_cols = [c for c in out.columns if c.endswith("_avg") or c.endswith("_inv")]
        out[perf_cols] = out[perf_cols].fillna(0.0)

        out = out.reset_index().sort_values(["patient_id", "week_start_ts", "week_number"])
        return out.set_index(["patient_id", "week_number"])