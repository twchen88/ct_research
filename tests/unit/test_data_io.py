
from __future__ import annotations
import importlib, sys, pathlib, json, os, stat
import numpy as np
import pandas as pd
import pytest

def _import_data_io():
    try:
        return importlib.import_module("ct.data.data_io")
    except Exception:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("data_io")

dio = _import_data_io()

class Test_write_sessions_to_csv:
    def test_happy_writes(self, tmp_path):
        df = pd.DataFrame({"a":[1,2], "b":["x","y"]})
        out = tmp_path / "sessions.csv"
        dio.write_sessions_to_csv(str(out), df)
        loaded = pd.read_csv(out)
        pd.testing.assert_frame_equal(loaded, df)

    def test_edge_overwrite_existing(self, tmp_path):
        out = tmp_path / "sessions.csv"
        out.write_text("", encoding="utf-8")
        df = pd.DataFrame({"a":[1]})
        # If function overwrites, this passes; if it raises, assert the exception:
        try:
            dio.write_sessions_to_csv(str(out), df)
            loaded = pd.read_csv(out)
            assert loaded.shape[0] == 1
        except Exception as e:
            assert isinstance(e, (FileExistsError, PermissionError))

class Test_write_sessions_to_npy:
    def test_happy_writes_npy(self, tmp_path):
        arr = np.arange(6, dtype=np.float32).reshape(3,2)
        out = tmp_path / "sessions.npy"
        dio.write_sessions_to_npy(str(out), arr)
        loaded = np.load(out)
        assert loaded.shape == (3,2)
        assert loaded.dtype == np.float32

class Test_read_raw_session_chunks:
    def test_happy_iterates_chunks(self, tmp_path):
        # create CSV with required columns
        df = pd.DataFrame({
            "id":[1,2,3,4,5],
            "patient_id":[1,1,1,1,1],
            "task_type_id":[1,1,1,1,1],
            "task_level":[1,1,1,1,1],
            "domain_ids":["1","1","1","1","1"],
            "domain_scores":["0.1","0.2","0.3","0.4","0.5"],
            "start_time":[pd.Timestamp("2024-01-01")+pd.Timedelta(minutes=i) for i in range(5)]
        })
        path = tmp_path / "raw.csv"
        df.to_csv(path, index=False)
        chunks = list(dio.read_raw_session_chunks(str(path), chunksize=2))
        assert len(chunks) >= 2
        assert "start_time" in chunks[0].columns
        assert pd.api.types.is_datetime64_any_dtype(chunks[0]["start_time"])

    def test_error_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            next(dio.read_raw_session_chunks(str(tmp_path / "missing.csv"), chunksize=2))

class Test_read_preprocessed_session_file:
    def test_happy_dtype_map(self, tmp_path):
        # Construct minimal valid CSV with expected columns
        cols = ["patient_id"] \
            + [f"domain {i} encoding" for i in range(1,15)] \
            + [f"domain {i} score" for i in range(1,15)] \
            + [f"domain {i} target" for i in range(1,15)] \
            + ["time_stamp"]
        row = [123] + [1]*14 + [0.5]*14 + [0.9]*14 + [1700000000]
        p = tmp_path / "pre.csv"
        pd.DataFrame([row], columns=cols).to_csv(p, index=False)
        out = dio.read_preprocessed_session_file(str(p))
        assert "patient_id" in out.columns and "time_stamp" in out.columns
        assert str(out["patient_id"].dtype) in {"int32","Int32"}
        assert str(out["domain 1 encoding"].dtype).lower().startswith("int")
        assert str(out["domain 1 score"].dtype).lower().startswith("float")

    def test_error_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _ = dio.read_preprocessed_session_file(str(tmp_path / "nope.csv"))
