from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from ct.predictor.GRU_MLP import TemporalEncoderGRU

"""
src/data/encoding.py
-----------------
This module contains functions for encoding data.
Make preprocessed data numerically and structurally compatible with the input expectations of prediction neural network.
* filter_nonzero_rows: Filters rows in a DataFrame that have at most a specified number of zeros.
* create_missing_indicator: Encodes a numpy array by replacing each value with a pair indicating whether it is NaN or not.
"""


def weekly_df_to_gru_batch(
    weekly_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]:
    """
    Convert HistoryEncoder output (filtered) into GRU inputs.

    Expects:
      - index is MultiIndex (patient_id, week_number)
      - one row per patient-week
      - numeric feature columns for GRU inputs

    Returns:
      X: (B, T_max, D) padded tensor
      lengths: (B,) actual lengths (number of weeks per patient)
      patient_ids: list of patient_ids aligned to batch rows
      feature_cols: used feature columns
    """
    if not (isinstance(weekly_df.index, pd.MultiIndex) and weekly_df.index.names[:2] == ["patient_id", "week_number"]):
        if {"patient_id", "week_number"}.issubset(weekly_df.columns):
            df = weekly_df.set_index(["patient_id", "week_number"]).copy()
        else:
            raise ValueError("weekly_df must have MultiIndex (patient_id, week_number) or columns patient_id, week_number.")
    else:
        df = weekly_df.copy()

    df = df.sort_index()

    # Pick feature columns (default: all numeric columns except timestamps)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found. Pass feature_cols explicitly.")

    patient_ids: List[int] = []
    seq_tensors: List[torch.Tensor] = []
    lengths: List[int] = []

    # group by patient_id (level 0)
    for pid, g in df.groupby(level=0, sort=False):
        g = g.droplevel(0).sort_index()  # index now week_number
        arr = g[feature_cols].to_numpy(dtype=np.float32)  # (T_i, D)
        if arr.shape[0] == 0:
            continue
        patient_ids.append(int(pid[0]) if isinstance(pid, tuple) else int(pid))
        lengths.append(arr.shape[0])
        seq_tensors.append(torch.tensor(arr, dtype=dtype))

    if not seq_tensors:
        raise ValueError("No sequences found after grouping. Is weekly_df empty?")

    B = len(seq_tensors)
    T_max = max(lengths)
    D = seq_tensors[0].shape[1]

    X = torch.zeros((B, T_max, D), dtype=dtype, device=device)
    for i, seq in enumerate(seq_tensors):
        t = seq.shape[0]
        X[i, :t, :] = seq.to(device)

    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    return X, lengths_t, patient_ids, feature_cols


@torch.no_grad()
def encode_weekly_df_for_mlp(
    model: TemporalEncoderGRU,
    weekly_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Produces:
      embeddings: (B, d_hidden) tensor for MLP input
      patient_ids: aligned list of patient_ids
      feature_cols: used feature cols
      lengths: sequence lengths
    """
    model = model.to(device)
    model.eval()

    X, lengths, patient_ids, used_cols = weekly_df_to_gru_batch(
        weekly_df, feature_cols=feature_cols, device=device
    )

    emb = model(X, lengths)  # (B, d_hidden)
    return {
        "embeddings": emb,
        "patient_ids": patient_ids,
        "feature_cols": used_cols,
        "lengths": lengths,
    }

def filter_nonzero_rows(df: pd.DataFrame, max_zeros: int) -> pd.DataFrame:
    """
    Returns a copy of the DataFrame containing only rows with at most `max_zeros` number of zeros.

    Parameters:
        df (pd.DataFrame): Input DataFrame to filter.
        max_zeros (int): Maximum allowed number of zeros per row.

    Returns:
        pd.DataFrame: Filtered DataFrame with qualifying rows.
    """
    zero_counts = (df == 0).sum(axis=1)
    filtered_df = df[zero_counts <= max_zeros].copy()
    return filtered_df


def create_missing_indicator(data: np.ndarray, rand_seed: int = 42) -> np.ndarray:
    """
    Encodes a numpy array by replacing each value with a pair:
    - If the value is NaN: [0,0] or [1,1] chosen randomly.
    - If the value is not NaN: [value, 1 - value].

    Parameters:
        data (np.ndarray): 2D array of floats with possible NaNs.
        rand_seed (int): Seed for reproducibility.

    Returns:
        np.ndarray: 2D array with encoded values.
    """
    np.random.seed(rand_seed)
    rows, cols = data.shape
    encoded = np.empty((rows, cols * 2), dtype=data.dtype)

    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            if np.isnan(val):
                choice = np.random.choice([0, 1])
                encoded[i, j * 2] = choice
                encoded[i, j * 2 + 1] = choice
            else:
                encoded[i, j * 2] = val
                encoded[i, j * 2 + 1] = 1 - val

    return encoded.copy()


def encode_target_data(target: np.ndarray, encoded_domains: np.ndarray) -> np.ndarray:
    """
    Encodes target data by replacing non-target values with 0s and target values with target score.
    - if encoding is 0, then target is 0
    - if encoding is 1, then target is target score

    Parameters:
        target (np.ndarray): 2D array of floats with possible NaNs.
        encoded_domains (np.ndarray): 2D array of encoded domain values (0 or 1).

    Returns:
        np.ndarray: 2D array with encoded values.
    """
    product = np.multiply(target, encoded_domains)
    return np.nan_to_num(product, nan=0)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple


class PairEncodedWindowDatasetWeek(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        E_pairs: np.ndarray,     # [U, T, K, 2]
        X_week: np.ndarray,      # [U, T, D_x]
        Y: np.ndarray,           # [U, T, K]
        M_target: np.ndarray,    # [U, T, K]
        meta: Dict[str, Any],
        actions: Optional[np.ndarray] = None,   # [U, T, Da] (optional)
        statics: Optional[np.ndarray] = None,   # [U, Ds]   (optional)
        user_time_slice: Optional[Dict[int, Tuple[int, int]]] = None,
        drop_no_target: Optional[bool] = None,
    ):
        super().__init__()
        self.cfg = config
        W = config["time"]["lookback_weeks"]
        self.window = W
        self.horizon = 1

        assert E_pairs.ndim == 4 and E_pairs.shape[-1] == 2
        U, T, K, _ = E_pairs.shape
        assert X_week.shape[:2] == (U, T)
        assert Y.shape == M_target.shape == (U, T, K)

        self.U, self.T, self.K = U, T, K
        self.meta = meta

        self.Dx = X_week.shape[2]
        self.Dp = 2 * K
        self.E_flat = E_pairs.reshape(U, T, self.Dp).astype(np.float32)
        self.X_week = X_week.astype(np.float32)

        pres = E_pairs[..., 0].astype(np.float32)
        invv = E_pairs[..., 1].astype(np.float32)
        self.Y_recon = pres * (1.0 - invv)   # [U,T,K]

        self.Y = Y.astype(np.float32)
        self.M = M_target.astype(np.float32)

        self.actions = actions.astype(np.float32) if actions is not None else None
        self.statics = statics.astype(np.float32) if statics is not None else None
        self.Da = 0 if self.actions is None else self.actions.shape[2]
        self.Ds = 0 if self.statics is None else self.statics.shape[1]

        # Build anchor indices
        self.indices: List[Tuple[int, int]] = []
        weeks_per_user: List[pd.DatetimeIndex] = meta.get("weeks_per_user", [pd.DatetimeIndex([]) for _ in range(U)])

        drop_no_target = self.cfg["train"]["drop_no_target"] if drop_no_target is None else drop_no_target
        for u in range(U):
            T_u = len(weeks_per_user[u])
            if T_u == 0: continue
            lo, hi = (0, T_u) if user_time_slice is None else user_time_slice.get(u, (0, T_u))
            lo = max(lo, self.window)
            hi = min(hi, T_u - self.horizon)
            for t in range(lo, hi):
                if drop_no_target and self.M[u, t, :].sum() < 1.0:
                    continue
                self.indices.append((u, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u, t = self.indices[idx]
        t0, t1 = t - self.window, t

        y_hist = self.Y_recon[u, t0:t1, :]                                  # [W,K]
        x_hist = np.concatenate([self.E_flat[u, t0:t1, :], self.X_week[u, t0:t1, :]], axis=-1)  # [W, 2K+D_x]

        a_now = self.actions[u, t, :] if self.actions is not None else np.zeros((0,), dtype=np.float32)
        s_static = self.statics[u, :]   if self.statics is not None else np.zeros((0,), dtype=np.float32)

        y_next = np.nan_to_num(self.Y[u, t, :].copy(), nan=0.0)             # [K]
        m_next = self.M[u, t, :].copy()

        return {
            "y_hist": torch.from_numpy(y_hist),
            "x_hist": torch.from_numpy(x_hist),
            "a_now": torch.from_numpy(a_now),
            "s_static": torch.from_numpy(s_static),
            "user_idx": torch.tensor(u, dtype=torch.long),
            "y_next": torch.from_numpy(y_next),
            "m_next": torch.from_numpy(m_next),
        }

    @property
    def dims(self) -> Dict[str, int]:
        return {
            "U": self.U,
            "K": self.K,
            "Dx_total": self.Dp + self.Dx,
            "Da": self.Da,
            "Ds": self.Ds,
            "W": self.window,
        }


def build_time_slices_weekly(meta: Dict[str, Any], config: Dict[str, Any]) -> Dict[int, Dict[str, Tuple[int, int]]]:
    train_frac = config["train"]["train_frac"]
    val_frac = config["train"]["val_frac"]
    slices: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for u, weeks in enumerate(meta["weeks_per_user"]):
        T_u = len(weeks)
        if T_u == 0:
            slices[u] = {"train": (0, 0), "val": (0, 0), "test": (0, 0)}
            continue
        i1 = int(np.floor(train_frac * T_u))
        i2 = int(np.floor((train_frac + val_frac) * T_u))
        slices[u] = {"train": (0, i1), "val": (i1, i2), "test": (i2, T_u)}
    return slices


def make_weekly_dataloaders(config: Dict[str, Any],
                            E_pairs: np.ndarray, X_week: np.ndarray, Y: np.ndarray, M_target: np.ndarray, meta: Dict[str, Any],
                            actions: Optional[np.ndarray] = None, statics: Optional[np.ndarray] = None):
    slices = build_time_slices_weekly(meta, config)
    ds_train = PairEncodedWindowDatasetWeek(config, E_pairs, X_week, Y, M_target, meta,
                                            actions=actions, statics=statics,
                                            user_time_slice={u: s["train"] for u, s in slices.items()})
    ds_val   = PairEncodedWindowDatasetWeek(config, E_pairs, X_week, Y, M_target, meta,
                                            actions=actions, statics=statics,
                                            user_time_slice={u: s["val"] for u, s in slices.items()})
    ds_test  = PairEncodedWindowDatasetWeek(config, E_pairs, X_week, Y, M_target, meta,
                                            actions=actions, statics=statics,
                                            user_time_slice={u: s["test"] for u, s in slices.items()})
    bs = config["train"]["batch_size"]
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=bs, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=bs, shuffle=False)
    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test

def advance_weekly_covariates(last_x: np.ndarray) -> np.ndarray:
    """
    last_x: (D_x,) ordered as [week_index, weeks_since_last_active, n_sessions_week, n_domains_week, week_of_year]
    MVP policy: keep volume/breadth fixed; increment week_index + week_of_year; update gap accordingly.
    """
    week_index, gap, nsess, ndoms, week_of_year = last_x.tolist()
    week_index = float(week_index) + 1.0
    week_of_year = int(week_of_year) + 1
    if week_of_year > 53: week_of_year = 1
    gap = 0.0 if nsess > 0 else float(gap) + 1.0
    return np.array([week_index, gap, nsess, ndoms, float(week_of_year)], dtype=np.float32)

def global_score(y_vec: np.ndarray, mask_vec: Optional[np.ndarray] = None) -> float:
    """
    y_vec: (K,) per-domain scores in [0,1]
    mask_vec: (K,) optional 1/0 mask of which domains are observed
    """
    if mask_vec is None:
        return float(np.mean(y_vec))
    m = np.asarray(mask_vec, dtype=float)
    s = m.sum()
    return 0.0 if s <= 1e-8 else float((y_vec * m).sum() / s)