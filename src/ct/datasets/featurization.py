"""
ct.datasets.featurization
-------------------------
Turn time-binned history (from ct.datasets.history, optionally filtered) into model-ready features.

Config (featurization section):
  scaling_method: "standard" | "minmax" | "none" | "percentile"
  scaling_scope:  "global" | "per_patient"
  time_bin_col:   (optional; default "step_index")

Input:
  - DataFrame indexed by (patient_id, step_index) OR columns patient_id, step_index
  - History columns from ct.datasets.history:
      step_start_ts
      freq_domain_<id>
      score_domain_<id>
      inv_domain_<id>
    Optionally from ct.datasets.filtering densify:
      obs_domain_<id>

Output:
  - DataFrame indexed by (patient_id, step_index) with scaled feature columns.
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
class FeaturizationConfig:
    scaling_method: str = "standard"
    scaling_scope: str = "global"

    # code-only knobs (not required in YAML)
    include_freq: bool = True
    include_score: bool = True
    include_inv: bool = True
    include_obs: bool = False
    passthrough_cols: Tuple[str, ...] = ("step_start_ts",)
    features_only: bool = True
    float32: bool = True

    # injected by pipeline (not in YAML)
    time_bin_col: str = "step_index"


def _load_featurization_config(featurization_cfg: Optional[Dict[str, Any]]) -> FeaturizationConfig:
    featurization_cfg = featurization_cfg or {}
    if "passthrough_cols" in featurization_cfg and isinstance(featurization_cfg["passthrough_cols"], list):
        featurization_cfg["passthrough_cols"] = tuple(featurization_cfg["passthrough_cols"])
    return FeaturizationConfig(**featurization_cfg)


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


def _infer_feature_cols(df: pd.DataFrame, cfg: FeaturizationConfig) -> List[str]:
    cols = list(df.columns)
    out: List[str] = []

    if cfg.include_freq:
        out += [c for c in cols if c.startswith("freq_domain_")]
    if cfg.include_score:
        out += [c for c in cols if c.startswith("score_domain_")]
    if cfg.include_inv:
        out += [c for c in cols if c.startswith("inv_domain_")]
    if cfg.include_obs:
        out += [c for c in cols if c.startswith("obs_domain_")]

    # stable order
    out = sorted(set(out))
    return out


def _safe_standardize(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)
    z = (x - mean) / std_safe
    return z, mean.squeeze(), std.squeeze()


def _safe_minmax(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mn = np.nanmin(x, axis=axis, keepdims=True)
    mx = np.nanmax(x, axis=axis, keepdims=True)
    denom = mx - mn
    denom_safe = np.where(denom == 0, 1.0, denom)
    y = (x - mn) / denom_safe
    return y, mn.squeeze(), mx.squeeze()


def _percentile_rank_1d(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    mask = ~np.isnan(a)
    v = a[mask]
    if v.size == 0:
        return out

    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    ranks = np.empty_like(order, dtype=float)

    n = v_sorted.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v_sorted[j + 1] == v_sorted[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    pct = np.zeros_like(ranks) if n == 1 else (ranks - 1.0) / (n - 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    out[mask] = pct[inv_order]
    return out


def _percentile_rank_matrix(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        y[:, j] = _percentile_rank_1d(x[:, j].astype(float))
    return y


# -----------------------------
# Scaling
# -----------------------------

def scale_features_global(
    df: pd.DataFrame, feature_cols: List[str], cfg: FeaturizationConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = df[feature_cols].to_numpy(dtype=float, copy=True)
    params: Dict[str, Any] = {"method": cfg.scaling_method, "scope": "global"}

    if cfg.scaling_method == "none":
        Xs = X
    elif cfg.scaling_method == "standard":
        Xs, mean, std = _safe_standardize(X, axis=0)
        params["mean"] = mean
        params["std"] = std
    elif cfg.scaling_method == "minmax":
        Xs, mn, mx = _safe_minmax(X, axis=0)
        params["min"] = mn
        params["max"] = mx
    elif cfg.scaling_method == "percentile":
        Xs = _percentile_rank_matrix(X)
        params["note"] = "percentile ranks computed from current data"
    else:
        logger.error(f"Unknown scaling_method={cfg.scaling_method!r}; expected standard|minmax|none|percentile.")
        raise ValueError(f"Unknown scaling_method={cfg.scaling_method!r}; expected standard|minmax|none|percentile.")

    out = df.copy()
    out[feature_cols] = Xs.astype(np.float32) if cfg.float32 else Xs
    return out, params


def scale_features_per_patient(
    df: pd.DataFrame, feature_cols: List[str], cfg: FeaturizationConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    params: Dict[str, Any] = {"method": cfg.scaling_method, "scope": "per_patient"}
    out = df.copy()

    for pid, sub_idx in out.groupby(level=0, sort=False).groups.items():
        sub = out.loc[sub_idx, feature_cols]
        X = sub.to_numpy(dtype=float, copy=True)

        if cfg.scaling_method == "none":
            Xs = X
        elif cfg.scaling_method == "standard":
            Xs, _, _ = _safe_standardize(X, axis=0)
        elif cfg.scaling_method == "minmax":
            Xs, _, _ = _safe_minmax(X, axis=0)
        elif cfg.scaling_method == "percentile":
            Xs = _percentile_rank_matrix(X)
        else:
            logger.error(f"Unknown scaling_method={cfg.scaling_method!r}; expected standard|minmax|none|percentile.")
            raise ValueError(f"Unknown scaling_method={cfg.scaling_method!r}; expected standard|minmax|none|percentile.")

        out.loc[sub_idx, feature_cols] = Xs.astype(np.float32) if cfg.float32 else Xs

    return out, params


# -----------------------------
# Public API
# -----------------------------

class FeaturizeHistory:
    """
    Symmetric with ct.datasets.history.BuildHistory and ct.datasets.filtering.FilterHistory.

    Usage:
      fz = FeaturizeHistory(filtered_history_df, featurization_cfg=config["featurization"])
      X = fz.run()
      # optionally inspect fz.params_ for reproducibility metadata
    """

    def __init__(self, history_df: pd.DataFrame, featurization_cfg: Optional[Dict[str, Any]] = None):
        self.history_df = history_df.copy()
        self.cfg = _load_featurization_config(featurization_cfg)
        self.params_: Dict[str, Any] = {}

    def run(self) -> pd.DataFrame:
        df = _ensure_history_index(self.history_df, self.cfg.time_bin_col)

        feature_cols = _infer_feature_cols(df, self.cfg)
        if not feature_cols:
            logger.error("No feature columns found. Expected columns with prefixes like "
                         "freq_domain_, score_domain_, inv_domain_ (and optionally obs_domain_). "
                         "Check include_* flags in featurization config.")
            raise ValueError(
                "No feature columns found. Expected columns with prefixes like "
                "freq_domain_, score_domain_, inv_domain_ (and optionally obs_domain_). "
                "Check include_* flags in featurization config."
            )

        # ensure numeric (coerce errors to NaN)
        df = df.copy()
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        # scale
        if self.cfg.scaling_scope == "global":
            df_scaled, params = scale_features_global(df, feature_cols, self.cfg)
        elif self.cfg.scaling_scope == "per_patient":
            df_scaled, params = scale_features_per_patient(df, feature_cols, self.cfg)
        else:
            logger.error(f"Unknown scaling_scope={self.cfg.scaling_scope!r}; expected global|per_patient.")
            raise ValueError(f"Unknown scaling_scope={self.cfg.scaling_scope!r}; expected global|per_patient.")

        # store params for inspection/debugging
        params["feature_cols"] = feature_cols
        self.params_ = params

        # output selection
        if self.cfg.features_only:
            X = df_scaled[feature_cols].copy()
        else:
            keep = list(feature_cols)
            for c in self.cfg.passthrough_cols:
                if c in df_scaled.columns:
                    keep.append(c)
            X = df_scaled[keep].copy()

        return X.sort_index()