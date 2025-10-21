
"""
Tests for ct.data.db_utils
"""
from __future__ import annotations
import importlib
from pathlib import Path
import pandas as pd
import pytest

def _import_db_utils():
    try:
        return importlib.import_module("ct.data.db_utils")
    except Exception:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("db_utils")

db_utils = _import_db_utils()

class DummyConn:
    pass

class Test_load_sql:
    def test_happy_query_string_calls_pandas_read_sql(self, monkeypatch, capture_calls):
        sentinel_df = pd.DataFrame({"x": [1]})
        capture_calls.return_value = sentinel_df
        monkeypatch.setattr(pd, "read_sql", capture_calls, raising=True)

        out = db_utils.load_sql("SELECT 1 AS x", DummyConn())
        assert isinstance(out, pd.DataFrame)
        # Ensure pandas.read_sql was called with the raw query string
        assert len(capture_calls.calls) == 1
        (args, kwargs) = capture_calls.calls[0]
        assert args[0].strip().lower().startswith("select")
        assert isinstance(args[1], DummyConn)

    def test_happy_sql_file_reads_contents_then_calls_pandas(self, tmp_path, monkeypatch, capture_calls):
        sql_path = tmp_path / "q.sql"
        sql_path.write_text("SELECT 42 AS answer", encoding="utf-8")
        sentinel_df = pd.DataFrame({"answer": [42]})
        capture_calls.return_value = sentinel_df
        monkeypatch.setattr(pd, "read_sql", capture_calls, raising=True)

        out = db_utils.load_sql(str(sql_path), DummyConn())
        assert isinstance(out, pd.DataFrame)
        assert len(capture_calls.calls) == 1
        (args, kwargs) = capture_calls.calls[0]
        # First arg should be the SQL string read from file
        assert "answer" in args[0]
        assert isinstance(args[1], DummyConn)

    def test_edge_non_sql_extension_treated_as_query(self, monkeypatch, capture_calls):
        """If the query doesn't end with .sql, it's treated as a query string even if it looks like a path."""
        capture_calls.return_value = pd.DataFrame({"x": [1]})
        monkeypatch.setattr(pd, "read_sql", capture_calls, raising=True)
        out = db_utils.load_sql("path/that/isnt/sql.txt", DummyConn())
        assert len(capture_calls.calls) == 1
