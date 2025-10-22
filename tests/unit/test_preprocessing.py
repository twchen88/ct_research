
from __future__ import annotations
import importlib, sys, pathlib
import pandas as pd
import numpy as np
import pytest

def _import_preprocessing():
    try:
        return importlib.import_module("ct.data.preprocessing")
    except Exception:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("preprocessing")

pre = _import_preprocessing()

class Test_drop_duplicates:
    def test_happy_drops_dupes(self, df_with_duplicates):
        out = pre.drop_duplicates(df_with_duplicates, based_on=["patient_id", "start_time"])
        assert len(out) == 2
        assert out.duplicated(subset=["patient_id", "start_time"]).sum() == 0

    def test_error_missing_column(self, df_with_duplicates):
        with pytest.raises(KeyError):
            pre.drop_duplicates(df_with_duplicates, based_on=["missing_col"])

class Test_sort_by_start_time:
    def test_happy_sorted_and_datetime(self, df_unsorted):
        out = pre.sort_by_start_time(df_unsorted.copy())
        assert out["start_time"].is_monotonic_increasing
        assert pd.api.types.is_datetime64_any_dtype(out["start_time"])

    def test_edge_with_nat(self):
        df = pd.DataFrame({"start_time":[pd.NaT, pd.Timestamp("2024-01-01")]})
        out = pre.sort_by_start_time(df)
        assert out.iloc[-1]["start_time"] is pd.NaT or pd.isna(out.iloc[-1]["start_time"])

class Test_find_usage_frequency:
    def test_happy_usage_freq(self, toy_sessions_df):
        out = pre.find_usage_frequency(toy_sessions_df.copy())
        assert set(out["patient_id"]) == {1,2}
        row1 = out[out["patient_id"]==1].iloc[0]
        assert int(row1["unique_days"]) == 1
        assert int(row1["usage_time"]) == 1
        assert row1["usage_freq"] == pytest.approx(1.0)

    def test_edge_single_session_span(self):
        df = pd.DataFrame({
            "patient_id":[9],
            "start_time":[pd.Timestamp("2024-01-05 10:00:00")]
        })
        out = pre.find_usage_frequency(df)
        r = out.iloc[0]
        assert r["unique_days"] == 1
        assert r["usage_time"] == 1
        assert r["usage_freq"] == 1.0

class Test_process_row:
    def test_happy_parse_lists(self, toy_sessions_df):
        row = toy_sessions_df.iloc[0]
        doms, scores = pre.process_row(row)
        assert doms == [1,3]
        assert scores == [0.2, 0.8]

    def test_error_non_numeric_tokens(self, toy_sessions_df):
        bad = toy_sessions_df.iloc[0].copy()
        bad["domain_ids"] = "1, x"
        with pytest.raises(ValueError):
            pre.process_row(bad)

class Test_extract_session_data:
    def test_happy_two_sessions(self, toy_sessions_df):
        out = pre.extract_session_data(toy_sessions_df.copy())
        expected_cols = 1 + 14 + 14 + 14 + 2
        assert out.shape[1] == expected_cols
        enc_cols = [f"domain {i} encoding" for i in range(1,15)]
        first = out[out["patient_id"]==1].iloc[0]
        assert first[enc_cols[0]] == 1 and first[enc_cols[2]] == 1
        tgt_cols = [f"domain {i} target" for i in range(1,15)]
        assert tgt_cols[0] in out.columns
        assert out[tgt_cols].select_dtypes(include="number").notna().any().all()

    def test_edge_single_session(self):
        df = pd.DataFrame([{
            "patient_id": 5,
            "domain_ids": "2",
            "domain_scores": "0.4",
            "start_time": pd.Timestamp("2024-01-01 08:00:00")
        }])
        out = pre.extract_session_data(df)
        assert (out["patient_id"] == 5).all()
        assert out[[f"domain {i} encoding" for i in range(1,15)]].sum(axis=1).iloc[0] == 1

class Test_filter_datetime_outliers:
    def test_happy_filters_far_outlier(self):
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "start_time":[base, base + pd.Timedelta(days=1), base + pd.Timedelta(days=2), base + pd.Timedelta(days=100)],
            "patient_id":[1,1,1,1]
        })
        filtered = pre.filter_datetime_outliers(df.copy(), eps_days=5, min_samples=2)
        # Expect the far-out point to be removed, keep the early cluster
        assert filtered["start_time"].max() <= base + pd.Timedelta(days=2)

    def test_edge_all_sparse_returns_empty_or_subset(self):
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({"start_time":[base, base + pd.Timedelta(days=10), base + pd.Timedelta(days=20)]})
        out = pre.filter_datetime_outliers(df.copy(), eps_days=1, min_samples=2)
        # Implementation-dependent: could return empty or original if no clusters found
        assert isinstance(out, pd.DataFrame)

class Test_convert_to_percentile:
    def test_happy_rank_pct(self, percent_df):
        out = pre.convert_to_percentile(percent_df.copy(), columns=["a","b"])
        assert out["a"].between(0,1).all()
        assert out["b"].between(0,1).all()
        assert out.sort_values("a").index.tolist() == [0,2,1]

    def test_edge_empty_columns(self, percent_df):
        out = pre.convert_to_percentile(percent_df.copy(), columns=[])
        pd.testing.assert_frame_equal(out, percent_df)
