# test/unit/test_history.py

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.api.types import is_integer_dtype, is_float_dtype

import ct.datasets.history as h


@pytest.fixture
def raw_df():
    # Two patients, multiple sessions across 7D bins, with paired lists as strings.
    # Include one bad timestamp row and one row with mismatched list lengths.
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2, 2, 1],
            "start_time": [
                "2026-01-01T10:00:00Z",  # p1 bin0
                "2026-01-03T10:00:00Z",  # p1 bin0
                "2026-01-10T10:00:00Z",  # p1 bin1
                "2026-01-02T10:00:00Z",  # p2 bin0
                "2026-01-09T10:00:00Z",  # p2 bin1
                "bad-ts",                # drop
                "2026-01-04T10:00:00Z",  # p1 bin0, mismatched lists -> aligned to min length
            ],
            "domain_ids": [
                "1,2",
                "1",
                "2",
                "1,2",
                "2",
                "1",
                "3,4,5",
            ],
            "domain_scores": [
                "0.2,0.8",
                "0.6",
                "0.1",
                "0.9,0.1",
                "0.4",
                "0.7",
                "0.5,not-a-number",  # only "0.5" survives coercion; aligned => ids ["3"]
            ],
        }
    )


@pytest.fixture
def cfg_base():
    return h.HistoryConfig(
        aggregate_window="7D",
        time_bin_alignment="period_start",
        time_index_mode="per_patient",
        aggregation_method="average",
        forward_fill=True,
        frequency_calculation="count",
        missing_encoding="00",
        time_bin_col="step_index",
        step_index_base=0,
    )


@pytest.fixture
def schema_base():
    return h.InputSchemaConfig(
        id_col="patient_id",
        time_col="start_time",
        paired_lists=[h.PairedListSpec(category_col="domain_ids", value_col="domain_scores", sep=",")],
    )


# -----------------------------
# Parsing helpers
# -----------------------------

def test__to_list_behaviors():
    assert h._to_list(["a", "b"], sep=",") == ["a", "b"]
    assert h._to_list(None, sep=",") == []
    assert h._to_list(np.nan, sep=",") == []
    assert h._to_list(("1", "2"), sep=",") == ["1", "2"]
    assert h._to_list("1, 2, ,3", sep=",") == ["1", "2", "3"]

    # Literal list parsing
    assert h._to_list("[1, 2, 3]", sep=",") == [1, 2, 3]

    # If literal parsing fails, it should fall back to split (no exception)
    out = h._to_list("[1, two, 3]", sep=",")
    assert isinstance(out, list)
    assert len(out) > 0


def test__coerce_numeric_list_drops_unparseable():
    assert h._coerce_numeric_list(["1", "2.5", 3]) == [1.0, 2.5, 3.0]
    assert h._coerce_numeric_list(["x", "1", "y"]) == [1.0]


# -----------------------------
# Time binning
# -----------------------------

def test__parse_window_valid_and_invalid():
    assert h._parse_window("7D") == pd.to_timedelta("7D")
    assert h._parse_window("1W") == pd.to_timedelta("7D")

    with pytest.raises(ValueError):
        h._parse_window("not-a-window")


def test__align_bin_start_modes():
    anchor = pd.Series(pd.to_datetime(["2026-01-01T00:00:00Z"] * 3, utc=True))
    bin_num = pd.Series([0, 1, 2], dtype=int)
    window = pd.to_timedelta("7D")

    start = h._align_bin_start(anchor, bin_num, window, "period_start")
    end = h._align_bin_start(anchor, bin_num, window, "period_end")
    floor = h._align_bin_start(anchor, bin_num, window, "floor")

    assert_series_equal(floor, start)
    assert (end - start).iloc[0] == window

    with pytest.raises(ValueError):
        h._align_bin_start(anchor, bin_num, window, "weird")


# -----------------------------
# consolidate_sessions
# -----------------------------

def test_consolidate_sessions_happy_path(raw_df, cfg_base, schema_base):
    long_df = h.consolidate_sessions(raw_df, cfg_base, schema_base)

    # Required columns
    assert {"patient_id", "step_index", "step_start_ts", "start_ts", "domain_id", "domain_score"}.issubset(
        long_df.columns
    )

    # bad-ts row should be dropped
    assert (long_df["start_ts"].isna()).sum() == 0

    # domain_id and domain_score dtypes
    assert str(long_df["domain_id"].dtype) in ("int32", "int64")  # depends on platform; spec default is int32
    assert str(long_df["domain_score"].dtype) in ("float32", "float64")  # spec default is float32

    # mismatched list row should contribute only one domain (id "3") because only score "0.5" survives
    assert (long_df["domain_id"] == 3).any()
    assert not (long_df["domain_id"] == 4).any()
    assert not (long_df["domain_id"] == 5).any()


def test_consolidate_sessions_missing_columns_raises(cfg_base, schema_base):
    df = pd.DataFrame({"patient_id": [1], "start_time": ["2026-01-01T00:00:00Z"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        h.consolidate_sessions(df, cfg_base, schema_base)


def test_consolidate_sessions_time_index_mode_global_vs_per_patient(raw_df, schema_base):
    cfg_per = h.HistoryConfig(aggregate_window="7D", time_index_mode="per_patient")
    cfg_glob = h.HistoryConfig(aggregate_window="7D", time_index_mode="global")

    long_per = h.consolidate_sessions(raw_df, cfg_per, schema_base)
    long_glob = h.consolidate_sessions(raw_df, cfg_glob, schema_base)

    # Step indices for patient 2 differ between modes, because anchor changes.
    # per_patient anchor for p2 is its min time (2026-01-02), global anchor is 2026-01-01.
    p2_per = long_per[long_per["patient_id"] == 2]["step_index"].min()
    p2_glob = long_glob[long_glob["patient_id"] == 2]["step_index"].min()
    assert p2_per == 0
    assert p2_glob == 0  # still 0 here because 1 day offset within 7D window

    # But the computed step_start_ts should differ (anchor-based)
    per_ts = (
        long_per[long_per["patient_id"] == 2]
        .sort_values(["step_index", "domain_id"])
        .iloc[0]["step_start_ts"]
    )
    glob_ts = (
        long_glob[long_glob["patient_id"] == 2]
        .sort_values(["step_index", "domain_id"])
        .iloc[0]["step_start_ts"]
    )
    assert per_ts != glob_ts


def test_consolidate_sessions_unknown_time_index_mode_raises(raw_df, schema_base):
    cfg = h.HistoryConfig(aggregate_window="7D", time_index_mode="nope")
    with pytest.raises(ValueError, match="Unknown time_index_mode"):
        h.consolidate_sessions(raw_df, cfg, schema_base)


# -----------------------------
# make_all_bins
# -----------------------------

def test_make_all_bins_builds_full_coverage(raw_df, cfg_base, schema_base):
    long_df = h.consolidate_sessions(raw_df, cfg_base, schema_base)
    all_bins = h.make_all_bins(long_df, cfg_base.time_bin_col)

    assert {"patient_id", "step_index", "step_start_ts"}.issubset(all_bins.columns)

    # p1 has bins 0..1; p2 has bins 0..1
    p1 = all_bins[all_bins["patient_id"] == 1]["step_index"].tolist()
    p2 = all_bins[all_bins["patient_id"] == 2]["step_index"].tolist()
    assert p1 == [0, 1]
    assert p2 == [0, 1]


def test_make_all_bins_requires_columns():
    df = pd.DataFrame({"patient_id": [1], "step_index": [0]})
    with pytest.raises(ValueError, match="long_df must contain"):
        h.make_all_bins(df, "step_index")


# -----------------------------
# encode_domain_frequency
# -----------------------------

def test_encode_domain_frequency_count(raw_df, cfg_base, schema_base):
    long_df = h.consolidate_sessions(raw_df, cfg_base, schema_base)
    all_bins = h.make_all_bins(long_df, cfg_base.time_bin_col)
    out = h.encode_domain_frequency(long_df, all_bins, cfg_base)

    # Indexed by (patient_id, step_index)
    assert out.index.names == ["patient_id", "step_index"]
    assert "step_start_ts" in out.columns

    freq_cols = [c for c in out.columns if c.startswith("freq_domain_")]
    assert freq_cols

    # Counts should be integers in count mode
    assert all(is_integer_dtype(out[c]) for c in freq_cols)

    # Sanity: patient 1 bin0 should have some domain counts > 0
    assert bool((out.loc[(1, 0), freq_cols] > 0).any())


def test_encode_domain_frequency_percent(raw_df, cfg_base, schema_base):
    cfg = cfg_base.__class__(**{**cfg_base.__dict__, "frequency_calculation": "percent"})
    long_df = h.consolidate_sessions(raw_df, cfg, schema_base)
    all_bins = h.make_all_bins(long_df, cfg.time_bin_col)
    out = h.encode_domain_frequency(long_df, all_bins, cfg)

    freq_cols = [c for c in out.columns if c.startswith("freq_domain_")]
    # Each row sums to 100 (or 0 if no sessions; but here bins exist due to observed coverage)
    row_sums = out[freq_cols].sum(axis=1)
    assert np.all((row_sums.round(6) == 100.0) | (row_sums.round(6) == 0.0))


def test_encode_domain_frequency_unknown_mode_raises(raw_df, cfg_base, schema_base):
    cfg = cfg_base.__class__(**{**cfg_base.__dict__, "frequency_calculation": "wat"})
    long_df = h.consolidate_sessions(raw_df, cfg, schema_base)
    all_bins = h.make_all_bins(long_df, cfg.time_bin_col)
    with pytest.raises(ValueError, match="Unknown frequency_calculation"):
        h.encode_domain_frequency(long_df, all_bins, cfg)


# -----------------------------
# aggregate_domain_performance
# -----------------------------

@pytest.mark.parametrize(
    "method",
    ["average", "max", "latest"],
)
def test_aggregate_domain_performance_methods(raw_df, cfg_base, schema_base, method):
    cfg = cfg_base.__class__(**{**cfg_base.__dict__, "aggregation_method": method})
    long_df = h.consolidate_sessions(raw_df, cfg, schema_base)
    perf_long = h.aggregate_domain_performance(long_df, cfg)

    assert {"patient_id", cfg.time_bin_col, "step_start_ts", "domain_id", "step_score"}.issubset(perf_long.columns)
    assert perf_long["step_score"].notna().all()


def test_aggregate_domain_performance_unknown_method_raises(raw_df, cfg_base, schema_base):
    cfg = cfg_base.__class__(**{**cfg_base.__dict__, "aggregation_method": "nope"})
    long_df = h.consolidate_sessions(raw_df, cfg, schema_base)
    with pytest.raises(ValueError, match="Unknown aggregation_method"):
        h.aggregate_domain_performance(long_df, cfg)


# -----------------------------
# build_performance_history
# -----------------------------

def test_build_performance_history_forward_fill_effect(raw_df, cfg_base, schema_base):
    # Create scenario where a domain is present in bin0 but missing in bin1;
    # forward_fill=True should carry it into bin1.
    cfg = cfg_base.__class__(**{**cfg_base.__dict__, "aggregation_method": "latest", "forward_fill": True})
    long_df = h.consolidate_sessions(raw_df, cfg, schema_base)
    all_bins = h.make_all_bins(long_df, cfg.time_bin_col)
    perf_long = h.aggregate_domain_performance(long_df, cfg)
    perf_df_ff = h.build_performance_history(perf_long, all_bins, cfg)

    cfg_noff = cfg.__class__(**{**cfg.__dict__, "forward_fill": False})
    perf_df_noff = h.build_performance_history(perf_long, all_bins, cfg_noff)

    # For patient 2, domain 1 appears in bin0 (from "1,2" row) but not in bin1 ("2" only).
    col = "score_domain_1"
    assert col in perf_df_ff.columns
    # bin0 should be non-null; bin1 should be carried forward in ff version
    assert pd.notna(perf_df_ff.loc[(2, 0), col])
    assert pd.notna(perf_df_ff.loc[(2, 1), col])
    # without ff, bin1 likely NaN
    assert pd.isna(perf_df_noff.loc[(2, 1), col])


# -----------------------------
# apply_missing_encoding
# -----------------------------

def test_apply_missing_encoding_00_sets_missing_to_zero():
    out = pd.DataFrame(
        {
            "step_start_ts": [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-08T00:00:00Z")],
            "freq_domain_1": [1, np.nan],
            "score_domain_1": [0.5, np.nan],
            "inv_domain_1": [0.5, np.nan],
        },
        index=pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["patient_id", "step_index"]),
    )
    cfg = h.HistoryConfig(missing_encoding="00", frequency_calculation="count")
    enc = h.apply_missing_encoding(out.copy(), cfg)

    assert enc.loc[(1, 1), "freq_domain_1"] == 0
    assert enc.loc[(1, 1), "score_domain_1"] == 0.0
    assert enc.loc[(1, 1), "inv_domain_1"] == 0.0
    assert np.issubdtype(str(enc["freq_domain_1"].dtype), np.integer)


def test_apply_missing_encoding_mean_not_implemented_raises():
    out = pd.DataFrame({"score_domain_1": [np.nan]}, index=pd.MultiIndex.from_tuples([(1, 0)], names=["patient_id", "step_index"]))
    cfg = h.HistoryConfig(missing_encoding="mean")
    with pytest.raises(NotImplementedError, match="not implemented"):
        h.apply_missing_encoding(out, cfg)


def test_apply_missing_encoding_unknown_raises():
    out = pd.DataFrame({"score_domain_1": [np.nan]}, index=pd.MultiIndex.from_tuples([(1, 0)], names=["patient_id", "step_index"]))
    cfg = h.HistoryConfig(missing_encoding="???")
    with pytest.raises(ValueError, match="Unknown missing_encoding"):
        h.apply_missing_encoding(out, cfg)


# -----------------------------
# BuildHistory integration
# -----------------------------

def test_build_history_run_end_to_end(raw_df):
    builder = h.BuildHistory(
        raw_data=raw_df,
        history_cfg={
            "aggregate_window": "7D",
            "time_index_mode": "per_patient",
            "time_bin_alignment": "period_start",
            "aggregation_method": "average",
            "forward_fill": True,
            "frequency_calculation": "count",
            "missing_encoding": "00",
            "time_bin_col": "step_index",
            "step_index_base": 0,
        },
        input_schema_cfg={
            "id_col": "patient_id",
            "time_col": "start_time",
            "paired_lists": [
                {"category_col": "domain_ids", "value_col": "domain_scores", "sep": ","}
            ],
        },
    )

    out = builder.run()

    # Indexing contract
    assert out.index.names == ["patient_id", "step_index"]
    assert "step_start_ts" in out.columns

    # Should include frequency and performance columns for observed domains
    assert any(c.startswith("freq_domain_") for c in out.columns)
    assert any(c.startswith("score_domain_") for c in out.columns)
    assert any(c.startswith("inv_domain_") for c in out.columns)

    # Missing encoding contract: no NaNs in score/inv after "00"
    score_cols = [c for c in out.columns if c.startswith("score_domain_")]
    inv_cols = [c for c in out.columns if c.startswith("inv_domain_")]
    assert out[score_cols].isna().sum().sum() == 0
    assert out[inv_cols].isna().sum().sum() == 0


def test_build_history_schema_defaults_warn_and_work(raw_df):
    # If input_schema_cfg is None, defaults are used:
    # id_col="patient_id", time_col="start_time", paired_lists domain_ids/domain_scores
    builder = h.BuildHistory(raw_data=raw_df, history_cfg={"aggregate_window": "7D"}, input_schema_cfg=None)
    out = builder.run()
    assert out.index.names == ["patient_id", "step_index"]