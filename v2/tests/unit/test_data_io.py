
"""
This file groups all test classes targeting functions in data_io.py:
- Test_read_preprocessed_session_file
- Test_read_raw_session_chunks
- Test_write_sessions_to_csv
- Test_write_sessions_to_npy
"""
from __future__ import annotations
import importlib
import pandas as pd
import numpy as np
import pytest

# Helper to import the module from package name or local path
def _import_data_io():
    try:
        return importlib.import_module("ct.data.data_io")
    except Exception:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("data_io")

data_io = _import_data_io()

# -------- read_preprocessed_session_file -------------------------------------

class Test_read_preprocessed_session_file:
    def test_happy_loads_with_expected_dtypes(self, tmp_csv_preprocessed):
        df = data_io.read_preprocessed_session_file(str(tmp_csv_preprocessed))
        # Assert required columns and dtypes (spot check a few; contract is the dtype map)
        assert "patient_id" in df.columns
        assert str(df["patient_id"].dtype) in {"int32", "Int32"}  # pandas may show nullable dtype
        assert "domain 1 encoding" in df.columns
        assert str(df["domain 1 encoding"].dtype) in {"int8", "Int8"}
        assert "domain 1 score" in df.columns
        # Float32 can surface as float32
        assert str(df["domain 1 score"].dtype) in {"float32", "Float32"}
        assert "time_stamp" in df.columns
        assert str(df["time_stamp"].dtype) in {"int64", "Int64"}

    def test_edge_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            data_io.read_preprocessed_session_file(str(tmp_path / "nope.csv"))

# -------- read_raw_session_chunks (xfail skeleton) ---------------------------

class Test_read_raw_session_chunks:
    def test_happy_iterates_in_chunks(self, tmp_csv_raw):
        chunksize = 20
        chunks = list(data_io.read_raw_session_chunks(str(tmp_csv_raw), chunksize=chunksize))
        # yields multiple DataFrames
        assert len(chunks) >= 2
        assert all(isinstance(c, pd.DataFrame) for c in chunks)

        # each chunk has 1..chunksize rows; last chunk may be smaller
        assert all(1 <= len(c) <= chunksize for c in chunks)

        # concatenated equals full file row count
        total = sum(len(c) for c in chunks)
        full = pd.read_csv(tmp_csv_raw)
        assert total == len(full)

        # start_time exists and is datetime after coercion
        first = chunks[0]
        assert "start_time" in first.columns
        assert pd.api.types.is_datetime64_any_dtype(first["start_time"])

    def test_edge_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            next(data_io.read_raw_session_chunks(str(tmp_path / "nope.csv")))  # consume to trigger read

    def test_edge_large_chunksize_returns_single_chunk(self, tmp_csv_raw):
        chunks = list(data_io.read_raw_session_chunks(str(tmp_csv_raw), chunksize=10_000))
        assert len(chunks) == 1

    def test_edge_invalid_dates_become_nat(self, tmp_path):
        # file with a bad start_time to exercise errors='coerce'
        import pandas as pd
        p = tmp_path / "bad_dates.csv"
        pd.DataFrame({
            "id":[1], "patient_id":[1], "task_type_id":[1], "task_level":[1],
            "domain_ids":["1"], "domain_scores":["0.1"], "start_time":["not-a-date"],
        }).to_csv(p, index=False)

        chunks = list(data_io.read_raw_session_chunks(str(p), chunksize=10))
        st = chunks[0]["start_time"].iloc[0]
        assert pd.isna(st)  # coerced to NaT

# -------- write_sessions_to_csv (xfail skeleton) -----------------------------

class Test_write_sessions_to_csv:
    def test_happy_writes_dataframe_to_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1,2], "b": ["x","y"]})
        out = tmp_path / "sessions.csv"
        data_io.write_sessions_to_csv(str(out), df)  # adjust signature if different
        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ["a","b"]
        assert len(loaded) == 2

    def test_edge_existing_file_behavior(self, tmp_path):
        out = tmp_path / "sessions.csv"
        out.write_text("", encoding="utf-8")
        df = pd.DataFrame({"a":[1]})
        # Expect either overwrite or FileExistsError depending on API
        data_io.write_sessions_to_csv(str(out), df)  # or with overwrite=True

# -------- write_sessions_to_npy (xfail skeleton) -----------------------------

class Test_write_sessions_to_npy:
    def test_happy_writes_npy(self, tmp_path):
        arr = np.arange(6, dtype=np.float32).reshape(3,2)
        out = tmp_path / "sessions.npy"
        data_io.write_sessions_to_npy(str(out), arr)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (3,2)
        assert loaded.dtype == np.float32
