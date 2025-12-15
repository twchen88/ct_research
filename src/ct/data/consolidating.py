from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def _parse_start_time_min(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, utc=True, errors="coerce")


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
        """
        Returns a *long* dataframe with one row per (patient_id, week_number, domain_id occurrence):
          patient_id, week_number, week_start_ts, domain_id, domain_score
        """
        needed = ["patient_id", "domain_ids", "domain_scores", "start_time_min"]
        base = df.loc[:, [c for c in needed if c in df.columns]].copy()

        missing = [c for c in needed if c not in base.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse timestamps
        base["start_ts"] = _parse_start_time_min(base["start_time_min"])
        base = base.dropna(subset=["start_ts"])

        # Normalize domains into aligned lists
        base["domain_ids_list"] = base["domain_ids"].apply(_to_list)
        base["domain_scores_list"] = base["domain_scores"].apply(_to_list)

        # Ensure scores are numeric
        base["domain_scores_list"] = base["domain_scores_list"].apply(_coerce_numeric_list)

        # Align lengths per row (truncate to min length to keep id-score pairing)
        def _align(row: pd.Series) -> Tuple[List[Any], List[float]]:
            ids = row["domain_ids_list"]
            scores = row["domain_scores_list"]
            m = min(len(ids), len(scores))
            if m <= 0:
                return [], []
            return ids[:m], scores[:m]

        aligned = base.apply(_align, axis=1, result_type="expand")
        base["domain_ids_list"] = aligned[0]
        base["domain_scores_list"] = aligned[1]

        # Drop sessions with no valid domain entries
        base = base[base["domain_ids_list"].map(len) > 0].copy()

        # Compute per-user week_number as 7-day windows from first session
        base = base.sort_values(["patient_id", "start_ts"])
        first_ts = base.groupby("patient_id", sort=False)["start_ts"].transform("min")
        delta_days = (base["start_ts"] - first_ts).dt.total_seconds() / (24 * 3600)
        base["week_number"] = np.floor(delta_days / 7.0).astype(int)

        # A representative timestamp for the weekly record (week start anchored on first_ts)
        base["week_start_ts"] = first_ts + pd.to_timedelta(base["week_number"] * 7, unit="D")

        # Explode into long format
        long_df = base[["patient_id", "week_number", "week_start_ts", "domain_ids_list", "domain_scores_list"]].copy()
        long_df = long_df.explode(["domain_ids_list", "domain_scores_list"], ignore_index=True)
        long_df = long_df.rename(columns={"domain_ids_list": "domain_id", "domain_scores_list": "domain_score"})

        # Coerce domain_id to string to avoid mixed int/str column naming issues
        long_df["domain_id"] = long_df["domain_id"].astype(str)
        long_df["domain_score"] = pd.to_numeric(long_df["domain_score"], errors="coerce")
        long_df = long_df.dropna(subset=["domain_score"])

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

        freq_df = self.encode_domains(long_df)          # week_start_ts + domain_*_freq
        perf_df = self.average_performance(long_df)     # week_start_ts + domain_*_avg + domain_*_inv

        # Merge (keeping one week_start_ts)
        out = freq_df.drop(columns=["week_start_ts"], errors="ignore").join(
            perf_df, how="outer"
        )

        # For domains not practiced: performance metrics -> (0,0)
        perf_cols = [c for c in out.columns if c.endswith("_avg") or c.endswith("_inv")]
        out[perf_cols] = out[perf_cols].fillna(0.0)

        # For domains never seen in a given week: freq already 0 from pivot; ensure any NAs are 0
        freq_cols = [c for c in out.columns if c.endswith("_freq")]
        out[freq_cols] = out[freq_cols].fillna(0).astype(int)

        # Sort by timestamp within patient
        out = out.reset_index()
        out = out.sort_values(["patient_id", "week_start_ts", "week_number"])
        out = out.set_index(["patient_id", "week_number"])

        return out