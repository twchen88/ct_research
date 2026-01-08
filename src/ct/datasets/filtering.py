"""
ct.datasets.filtering
---------------------
Filter + densify time-binned history output (from ct.datasets.history).

Input:
  - history_df: DataFrame indexed by (patient_id, step_index) (or with those columns)
  - filtering config section:
      max_gap_windows
      min_history_windows
      min_active_windows
      time_bin_col (optional; default "step_index")

Output:
  - filtered (optionally densified) DataFrame indexed by (patient_id, step_index)

Notes:
  - This module computes per-domain *_obs indicators BEFORE forward-filling scores
    so you can filter on true observed activity while still producing dense grids.
  - Uses ct.datasets.history naming:
      step_start_ts
      freq_domain_<id>
      score_domain_<id>
      inv_domain_<id>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from ct.utils.logger import get_logger
logger = get_logger(__name__)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class FilteringConfig:
    max_gap_windows: int = 8
    min_history_windows: int = 12
    min_active_windows: int = 8

    # code-only knobs (not required in YAML)
    densify: bool = True
    keep_obs_cols: bool = True

    # injected by pipeline (not in YAML)
    time_bin_col: str = "step_index"


def _load_filtering_config(filtering_cfg: Optional[Dict[str, Any]]) -> FilteringConfig:
    filtering_cfg = filtering_cfg or {}
    return FilteringConfig(**filtering_cfg)


# -----------------------------
# Utilities
# -----------------------------

def _ensure_history_index(df: pd.DataFrame, time_bin_col: str) -> pd.DataFrame:
    """
    Ensure df is indexed by (patient_id, time_bin_col) and sorted.
    """
    if (
        isinstance(df.index, pd.MultiIndex)
        and df.index.names[:2] == ["patient_id", time_bin_col]
    ):
        out = df.copy()
    elif {"patient_id", time_bin_col}.issubset(df.columns):
        out = df.set_index(["patient_id", time_bin_col]).copy()
    else:
        logger.error(f"DataFrame index: {df.index.names}, columns: {df.columns.tolist()}")
        raise ValueError(
            f"Expected MultiIndex (patient_id, {time_bin_col}) or columns patient_id, {time_bin_col}."
        )
    return out.sort_index()


def _infer_history_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    cols = list(df.columns)
    freq_cols = [c for c in cols if c.startswith("freq_domain_")]
    score_cols = [c for c in cols if c.startswith("score_domain_")]
    inv_cols = [c for c in cols if c.startswith("inv_domain_")]
    obs_cols = [c for c in cols if c.startswith("obs_domain_")]
    return freq_cols, score_cols, inv_cols, obs_cols


def _compute_domain_obs_from_scores(
    dense: pd.DataFrame, score_cols: List[str], inv_cols: List[str]
) -> pd.DataFrame:
    """
    Compute per-domain observed mask for the current bin BEFORE ffill.
    Output columns are "obs_domain_<id>" aligned to score_cols.
    """
    if not score_cols:
        logger.warning("No score_domain_ columns found; cannot compute obs_domain_.")
        return pd.DataFrame(index=dense.index)

    # score_domain_123 -> obs_domain_123
    obs_cols = [c.replace("score_domain_", "obs_domain_") for c in score_cols]

    score_present = ~dense[score_cols].isna()
    if inv_cols:
        inv_present = ~dense[inv_cols].isna()
        inv_present.columns = score_cols  # align domain-wise
        obs = (score_present | inv_present).astype(np.float32)
    else:
        obs = score_present.astype(np.float32)

    return pd.DataFrame(obs.to_numpy(np.float32), index=dense.index, columns=obs_cols)


def _max_consecutive_false(mask: np.ndarray) -> int:
    max_run = 0
    run = 0
    for ok in mask:
        if not ok:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _bin_observed_any_domain(
    sub: pd.DataFrame,
    *,
    obs_cols: List[str],
    freq_cols: List[str],
) -> np.ndarray:
    """
    Bin is observed if any domain is observed.
    Preference order:
      1) obs_domain_* columns if present (true observation before ffill)
      2) sum of freq_domain_* columns > 0
    """
    if obs_cols:
        return (sub[obs_cols].to_numpy(np.float32).sum(axis=1) > 0)
    if freq_cols:
        return (sub[freq_cols].to_numpy(np.float32).sum(axis=1) > 0)
    return np.zeros(len(sub), dtype=bool)


# -----------------------------
# Densify
# -----------------------------

def densify_bin_grid_ffill_scores(df: pd.DataFrame, time_bin_col: str) -> pd.DataFrame:
    """
    Densify each patient to full bin grid [min_bin..max_bin].

    Gap bins:
      - freq := 0
      - score/inv := forward-filled (stable)
      - obs_domain_* := 1 if that domain had a real value that bin, else 0
        (computed BEFORE forward fill)
    """
    df = _ensure_history_index(df, time_bin_col).copy()
    freq_cols, score_cols, inv_cols, _ = _infer_history_columns(df)

    # ensure float for NaN/ffill behavior
    if freq_cols:
        df[freq_cols] = df[freq_cols].astype(float)
    if score_cols:
        df[score_cols] = df[score_cols].astype(float)
    if inv_cols:
        df[inv_cols] = df[inv_cols].astype(float)

    out_parts: List[pd.DataFrame] = []

    for pid, sub in df.groupby(level=0, sort=False):
        sub = sub.droplevel(0).sort_index()
        if len(sub) == 0:
            continue

        bmin, bmax = int(sub.index.min()), int(sub.index.max())
        full_bins = pd.Index(range(bmin, bmax + 1), name=time_bin_col)

        dense = sub.reindex(full_bins)  # NaNs on gaps

        # observed mask BEFORE ffill
        obs_df = _compute_domain_obs_from_scores(dense, score_cols, inv_cols)
        if not obs_df.empty:
            dense = dense.join(obs_df, how="left")

        # gaps => freq=0
        if freq_cols:
            dense[freq_cols] = dense[freq_cols].fillna(0.0)

        # forward-fill scores through gaps
        if score_cols:
            dense[score_cols] = dense[score_cols].ffill().fillna(0.0)
        if inv_cols:
            dense[inv_cols] = dense[inv_cols].ffill().fillna(0.0)

        dense["patient_id"] = pid
        dense = dense.reset_index().set_index(["patient_id", time_bin_col])
        out_parts.append(dense)

    return pd.concat(out_parts, axis=0).sort_index() if out_parts else df.iloc[0:0].copy()


# -----------------------------
# Filtering
# -----------------------------

def filter_patients_by_config(df: pd.DataFrame, cfg: FilteringConfig) -> pd.DataFrame:
    """
    Apply filtering rules:
      - min_history_windows: require at least this many bins in dense grid
      - min_active_windows: require at least this many observed bins (any domain observed)
      - max_gap_windows: disallow any consecutive gap longer than this (gaps = not observed)
    """
    df = _ensure_history_index(df, cfg.time_bin_col)
    freq_cols, _, _, obs_cols = _infer_history_columns(df)

    keep: List[Any] = []

    for pid, sub in df.groupby(level=0, sort=False):
        sub = sub.droplevel(0).sort_index()
        T = len(sub)

        if T < int(cfg.min_history_windows):
            continue

        bin_obs = _bin_observed_any_domain(sub, obs_cols=obs_cols, freq_cols=freq_cols)

        if int(bin_obs.sum()) < int(cfg.min_active_windows):
            continue

        if _max_consecutive_false(bin_obs) > int(cfg.max_gap_windows):
            continue

        keep.append(pid)

    if not keep:
        return df.iloc[0:0].copy()

    return df.loc[df.index.get_level_values(0).isin(keep)].copy()


# -----------------------------
# Public API
# -----------------------------

class FilterHistory:
    """
    Public API wrapper, symmetric with ct.datasets.history.BuildHistory.

    Usage:
      fh = FilterHistory(history_df, filtering_cfg=config["filtering"])
      filtered_df = fh.run()
    """

    def __init__(self, history_df: pd.DataFrame, filtering_cfg: Optional[Dict[str, Any]] = None):
        self.history_df = history_df.copy()
        self.cfg = _load_filtering_config(filtering_cfg)

    def run(self) -> pd.DataFrame:
        df = _ensure_history_index(self.history_df, self.cfg.time_bin_col)

        if self.cfg.densify:
            df = densify_bin_grid_ffill_scores(df, self.cfg.time_bin_col)

        df = filter_patients_by_config(df, self.cfg)

        if not self.cfg.keep_obs_cols:
            obs_cols = [c for c in df.columns if c.startswith("obs_domain_")]
            if obs_cols:
                df = df.drop(columns=obs_cols)

        return df