"""
ct.datasets.history
------------------
Build per-patient time-binned history from raw session rows.

Output:
  pandas DataFrame indexed by (patient_id, step_index) containing:
    - step_start_ts (timestamp of start of time bin)
    - freq_domain_<id> (count or percent)
    - score_domain_<id>  (performance)
    - inv_domain_<id>  (1 - score)

Missing domains:
  - If missing_encoding == "00": score=0 and inv=0 (and freq=0)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ct.utils.logger import get_logger
logger = get_logger(__name__)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class HistoryConfig:
    # Binning
    aggregate_window: str = "1W"          # e.g. "1W", "7D"
    time_bin_alignment: str = "floor"  # floor|period_start|period_end
    time_index_mode: str = "per_patient"  # per_patient|global

    # Aggregation / encoding
    aggregation_method: str = "average"    # latest|average|max
    forward_fill: bool = True # whether to carry forward last known value in gaps
    frequency_calculation: str = "count"  # count|percent
    missing_encoding: str = "00"          # 00|mean|zero (00 and zero behave the same here)

    # Output indexing
    time_bin_col: str = "step_index"     
    step_index_base: int = 0              # 0|1 (applies to produced bin numbers)

@dataclass(frozen=True)
class PairedListSpec:
    category_col: str = "domain_ids"
    value_col: str = "domain_scores"

    category_dtype: str = "int32"
    value_dtype: str = "float32"

    sep: str = ","

@dataclass(frozen=True)
class InputSchemaConfig:
    id_col: str = "patient_id"
    time_col: str = "start_time"
    paired_lists: List[PairedListSpec] = field(default_factory=lambda: [PairedListSpec()])


def _load_history_config(history_cfg: Optional[Dict[str, Any]]) -> HistoryConfig:
    history_cfg = history_cfg or {}
    return HistoryConfig(**history_cfg)


def _load_input_schema_config(input_schema_cfg: Optional[Dict[str, Any]]) -> InputSchemaConfig:
    if input_schema_cfg is None:
        logger.warning("No input_schema config provided; using defaults.")
        return InputSchemaConfig()
    paired = [
        PairedListSpec(**spec)
        for spec in input_schema_cfg.get("paired_lists", [])
    ]
    return InputSchemaConfig(
        id_col=input_schema_cfg["id_col"],
        time_col=input_schema_cfg["time_col"],
        paired_lists=paired,
    )

# -----------------------------
# Parsing helpers
# -----------------------------

def _to_list(x: Any, sep: str) -> List[Any]:
    """
    Separate string to list by sep, or parse literal list.
    """
    if isinstance(x, list):
        logger.debug("Input is already a list; returning as is.")
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        logger.debug("Input is None or NaN; returning empty list.")
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        logger.debug("Input is an array-like; converting to list.")
        return list(x)
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except Exception:
                logger.debug(f"Failed to parse list literal: {x!r}; falling back to split by {sep!r}.")
        parts = [p.strip() for p in x.split(sep)]
        return [p for p in parts if p != ""]
    return [x]


def _coerce_numeric_list(xs: List[Any]) -> List[float]:
    out: List[float] = []
    drop_count = 0
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            drop_count += 1
            continue
    if drop_count > 0:
        logger.warning(f"Failed to coerce value to float; dropping {drop_count} unparseable entries.")
    return out


# -----------------------------
# Time binning
# -----------------------------

def _parse_window(window: str) -> pd.Timedelta:
    """
    Convert config aggregate_window like "1W" or "7D" into a Timedelta.
    Pandas supports "W" as weeks; "1W" -> 7 days.
    """
    try:
        return pd.to_timedelta(window)
    except Exception as e:
        logger.error(f"Unsupported aggregate_window={window!r}. Try '7D' or '1W'.")
        raise ValueError(f"Unsupported aggregate_window={window!r}. Try '7D' or '1W'.") from e


def _bin_number(delta: pd.Series, window: pd.Timedelta) -> pd.Series:
    window_ns = window.value
    delta_ns = delta.astype("timedelta64[ns]").astype("int64")
    return (delta_ns // window_ns).astype(int)


def _align_bin_start(
    anchor_ts: pd.Series,
    bin_num: pd.Series,
    window: pd.Timedelta,
    alignment: str,
) -> pd.Series:
    """
    Compute the representative timestamp for the bin.
    alignment:
      - period_start: anchor + bin*window
      - period_end: anchor + (bin+1)*window
      - floor: alias for period_start (kept for config compatibility)
    """
    if alignment == "floor":
        alignment = "period_start"

    if alignment == "period_start":
        return anchor_ts + (bin_num * window)

    if alignment == "period_end":
        return anchor_ts + ((bin_num + 1) * window)

    logger.error(f"Unknown time_bin_alignment={alignment!r}; expected floor|period_start|period_end.")
    raise ValueError(
        f"Unknown time_bin_alignment={alignment!r}; expected floor|period_start|period_end."
    )


# -----------------------------
# Core transforms
# -----------------------------

def consolidate_sessions(raw: pd.DataFrame, cfg: HistoryConfig, schema: InputSchemaConfig) -> pd.DataFrame:
    """
    Produce long dataframe with one row per (session, domain) and computed time bins.

    Output columns:
      id_col, <cfg.time_bin_col>, step_start_ts, start_ts, domain_id, domain_score
    """
    if len(schema.paired_lists) != 1:
        logger.error("Expected exactly one paired_lists entry for history building.")
        raise ValueError("Expected exactly one paired_lists entry for history building.")
    spec = schema.paired_lists[0]
    needed = [schema.id_col] + [spec.category_col] + [spec.value_col] + [schema.time_col]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    base = raw.loc[:, needed].copy()

    base["start_ts"] = pd.to_datetime(base[schema.time_col], utc=True, errors="coerce")
    base = base.dropna(subset=["start_ts"])

    # normalize lists
    base["domain_ids_list"] = base[spec.category_col].apply(lambda v: _to_list(v, sep=spec.sep))
    base["domain_scores_list"] = base[spec.value_col].apply(lambda v: _to_list(v, sep=spec.sep)).apply(_coerce_numeric_list)

    # align list lengths
    def _align_lists(row: pd.Series) -> Tuple[List[Any], List[float]]:
        ids = row["domain_ids_list"]
        scores = row["domain_scores_list"]
        m = min(len(ids), len(scores))
        return (ids[:m], scores[:m]) if m > 0 else ([], [])

    aligned = base.apply(_align_lists, axis=1, result_type="expand")
    base["domain_ids_list"] = aligned[0]
    base["domain_scores_list"] = aligned[1]
    base = base[base["domain_ids_list"].map(len) > 0].copy()

    window = _parse_window(cfg.aggregate_window)

    # binning
    base = base.sort_values([schema.id_col, "start_ts"])

    if cfg.time_index_mode == "global":
        global_anchor = pd.to_datetime(base["start_ts"].min(), utc=True)
        anchor = pd.Series([global_anchor] * len(base), index=base.index)
    elif cfg.time_index_mode == "per_patient":
        anchor = base.groupby(schema.id_col, sort=False)["start_ts"].transform("min")
    else:
        logger.error(f"Unknown time_index_mode={cfg.time_index_mode!r}")
        raise ValueError(f"Unknown time_index_mode={cfg.time_index_mode!r}")

    bin_col = cfg.time_bin_col
    bin_ts_col = "step_start_ts"
    delta = base["start_ts"] - anchor
    base[bin_col] = _bin_number(delta, window) + int(cfg.step_index_base)
    base[bin_ts_col] = _align_bin_start(anchor, base[bin_col] - int(cfg.step_index_base), window, cfg.time_bin_alignment)

    # explode to long
    long_df = base[[schema.id_col, bin_col, bin_ts_col, "start_ts", "domain_ids_list", "domain_scores_list"]].copy()
    long_df = long_df.explode(["domain_ids_list", "domain_scores_list"], ignore_index=True)
    long_df = long_df.rename(columns={"domain_ids_list": "domain_id",
                                      "domain_scores_list": "domain_score",
                                      schema.id_col: "patient_id"})

    long_df["domain_id"] = pd.to_numeric(long_df["domain_id"], errors="coerce")
    long_df["domain_score"] = pd.to_numeric(long_df["domain_score"], errors="coerce")

    long_df = long_df.dropna(subset=["domain_id", "domain_score"])

    long_df["domain_id"] = long_df["domain_id"].astype(np.dtype(spec.category_dtype))
    long_df["domain_score"] = long_df["domain_score"].astype(np.dtype(spec.value_dtype))

    return long_df


def make_all_bins(long_df: pd.DataFrame, time_bin_col: str) -> pd.DataFrame:
    """
    Create full (patient_id, step_index) coverage from min..max observed bin per patient.
    """
    required = {"patient_id", time_bin_col, "step_start_ts"}
    if not required.issubset(long_df.columns):
        logger.error(f"long_df must contain {required}, got {set(long_df.columns)}")
        raise ValueError(f"long_df must contain {required}, got {set(long_df.columns)}")

    max_bins = long_df.groupby("patient_id", as_index=False).agg(max_step=(time_bin_col, "max"))
    min_bins = long_df.groupby("patient_id", as_index=False).agg(min_step=(time_bin_col, "min"))

    mm = min_bins.merge(max_bins, on="patient_id", how="inner")

    parts = []
    for pid, mn, mx in zip(mm["patient_id"].tolist(), mm["min_step"].tolist(), mm["max_step"].tolist()):
        mn = int(mn); mx = int(mx)
        parts.append(pd.DataFrame({"patient_id": pid, time_bin_col: np.arange(mn, mx + 1, dtype=int)}))

    all_bins = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["patient_id", time_bin_col])
    # anchor step_start_ts by joining observed starts (bin start should be stable by construction)
    starts = (
        long_df.groupby(["patient_id", time_bin_col], as_index=False)["step_start_ts"]
        .min()
    )
    all_bins = all_bins.merge(starts, on=["patient_id", time_bin_col], how="left")

    # for any missing step_start_ts (should be rare), forward fill within patient by step_index ordering
    all_bins = all_bins.sort_values(["patient_id", time_bin_col])
    all_bins["step_start_ts"] = all_bins.groupby("patient_id", sort=False)["step_start_ts"].ffill()

    return all_bins.reset_index(drop=True)


def encode_domain_frequency(long_df: pd.DataFrame, all_bins: pd.DataFrame, cfg: HistoryConfig) -> pd.DataFrame:
    """
    Returns df indexed by (patient_id, step_index):
      step_start_ts + freq_domain_<id> columns
    Missing bins filled with 0.
    """
    counts = (
        long_df.groupby(["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id"], as_index=False)
        .size()
        .rename(columns={"size": "freq"})
    )

    wide = (
        counts.pivot_table(
            index=["patient_id", cfg.time_bin_col],
            columns="domain_id",
            values="freq",
            fill_value=0,
            aggfunc="sum",
        )
        .sort_index(axis=1)
    )

    wide.columns = [f"freq_domain_{d}" for d in wide.columns.astype(str)]

    base = all_bins.set_index(["patient_id", cfg.time_bin_col])[["step_start_ts"]]
    out = base.join(wide, how="left").fillna(0)

    freq_cols = [c for c in out.columns if c.startswith("freq_domain_")]
    if cfg.frequency_calculation == "count":
        out[freq_cols] = out[freq_cols].astype(int)
        return out

    if cfg.frequency_calculation == "percent":
        # percent of sessions in that bin that are of each domain
        denom = out[freq_cols].sum(axis=1).replace({0: np.nan})
        out[freq_cols] = (out[freq_cols].div(denom, axis=0) * 100.0).fillna(0.0)
        return out

    logger.error(f"Unknown frequency_calculation={cfg.frequency_calculation!r}; expected 'count' or 'percent'.")
    raise ValueError(f"Unknown frequency_calculation={cfg.frequency_calculation!r}; expected 'count' or 'percent'.")


def aggregate_domain_performance(long_df: pd.DataFrame, cfg: HistoryConfig) -> pd.DataFrame:
    """
    Returns long df:
      patient_id, step_index, step_start_ts, domain_id, step_score
    where step_score is aggregated within a bin.
    """
    method = cfg.aggregation_method

    if method in ("average", "mean"):
        agg = (
            long_df.groupby(["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id"], as_index=False)
            .agg(step_score=("domain_score", "mean"))
        )
        return agg

    if method == "max":
        agg = (
            long_df.groupby(["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id"], as_index=False)
            .agg(step_score=("domain_score", "max"))
        )
        return agg

    if method == "latest":
        tmp = long_df.sort_values(["patient_id", cfg.time_bin_col, "domain_id", "start_ts"])
        latest = (
            tmp.groupby(["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id"], as_index=False)
            .tail(1)
            .rename(columns={"domain_score": "step_score"})
        )
        return latest[["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id", "step_score"]]

    logger.error(f"Unknown aggregation_method={method!r}; expected latest|average|max.")
    raise ValueError(f"Unknown aggregation_method={method!r}; expected latest|average|max.")


def build_performance_history(
    perf_long: pd.DataFrame,
    all_bins: pd.DataFrame,
    cfg: HistoryConfig,
) -> pd.DataFrame:
    """
    Create wide history columns (avg and inv) aligned to all bins.
    If cfg.forward_fill=True, carry forward last known step_score per patient/domain.
    """
    domains = pd.DataFrame({"domain_id": perf_long["domain_id"].unique()})
    patients = all_bins[["patient_id"]].drop_duplicates()

    grid = patients.merge(domains, how="cross").merge(all_bins, on="patient_id", how="left")

    grid = grid.merge(
        perf_long,
        on=["patient_id", cfg.time_bin_col, "step_start_ts", "domain_id"],
        how="left",
    ).sort_values(["patient_id", "domain_id", cfg.time_bin_col])

    if cfg.forward_fill:
        logger.debug("Applying forward fill to performance scores.")
        grid["score_hist"] = grid.groupby(["patient_id", "domain_id"], sort=False)["step_score"].ffill()
    else:
        grid["score_hist"] = grid["step_score"]
    
    def _last(s: pd.Series):
        return s.iloc[-1]

    wide = grid.pivot_table(
        index=["patient_id", cfg.time_bin_col],
        columns="domain_id",
        values="score_hist",
        aggfunc=_last,
    ).sort_index(axis=1)

    inv = 1.0 - wide

    wide.columns = [f"score_domain_{d}" for d in wide.columns.astype(str)]
    inv.columns = [f"inv_domain_{d}" for d in inv.columns.astype(str)]

    base_ts = all_bins.set_index(["patient_id", cfg.time_bin_col])[["step_start_ts"]]
    out = base_ts.join(wide, how="left").join(inv, how="left")

    return out


def apply_missing_encoding(out: pd.DataFrame, cfg: HistoryConfig) -> pd.DataFrame:
    """
    Ensure missing domain cells are encoded consistently.
    - "00" or "zero": score=0, inv=0, freq=0
    - "mean": (optional future) could fill avg with per-domain mean; not implemented by default
    """
    freq_cols = [c for c in out.columns if c.startswith("freq_domain_")]
    avg_cols = [c for c in out.columns if c.startswith("score_domain_")]
    inv_cols = [c for c in out.columns if c.startswith("inv_domain_")]

    if freq_cols:
        # frequencies always 0 when missing
        if cfg.frequency_calculation == "count":
            out[freq_cols] = out[freq_cols].fillna(0).astype(int)
        else:
            out[freq_cols] = out[freq_cols].fillna(0.0)

    enc = cfg.missing_encoding

    if enc in ("00", "zero"):
        if avg_cols:
            out[avg_cols] = out[avg_cols].fillna(0.0)
        if inv_cols:
            out[inv_cols] = out[inv_cols].fillna(0.0)
        return out

    if enc == "mean":
        logger.error("missing_encoding='mean' not implemented.")
        raise NotImplementedError("missing_encoding='mean' not implemented (would require per-domain mean fill).")

    logger.error(f"Unknown missing_encoding={enc!r}; expected '00'|'zero'|'mean'.")
    raise ValueError(f"Unknown missing_encoding={enc!r}; expected '00'|'zero'|'mean'.")


# -----------------------------
# Public API
# -----------------------------

class BuildHistory:
    """
    Public API wrapper around pure functions.
    """

    def __init__(self, raw_data: pd.DataFrame, history_cfg: Optional[Dict[str, Any]] = None, input_schema_cfg: Optional[Dict[str, Any]] = None):
        self.raw_data = raw_data.copy()
        self.cfg = _load_history_config(history_cfg)
        self.schema = _load_input_schema_config(input_schema_cfg)

    def run(self) -> pd.DataFrame:
        long_df = consolidate_sessions(self.raw_data, self.cfg, self.schema)

        all_bins = make_all_bins(long_df, self.cfg.time_bin_col)

        freq_df = encode_domain_frequency(long_df, all_bins, self.cfg)
        perf_long = aggregate_domain_performance(long_df, self.cfg)
        perf_df = build_performance_history(perf_long, all_bins, self.cfg)

        # join (avoid duplicating step_start_ts)
        out = freq_df.drop(columns=["step_start_ts"], errors="ignore").join(perf_df, how="outer")
        out = apply_missing_encoding(out, self.cfg)

        # stable ordering + index
        out = out.reset_index().sort_values(["patient_id", "step_start_ts", self.cfg.time_bin_col])
        return out.set_index(["patient_id", self.cfg.time_bin_col])