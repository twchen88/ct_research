import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest


def _ensure_ct_utils_package():
    """
    Ensure ct.utils exists as a *package* without shadowing the real ct package.
    If import fails, synthesize a minimal package-like module.
    """
    try:
        return importlib.import_module("ct.utils")
    except Exception:
        utils_pkg = sys.modules.get("ct.utils")
        if utils_pkg is None:
            utils_pkg = types.ModuleType("ct.utils")
            sys.modules["ct.utils"] = utils_pkg
        if not hasattr(utils_pkg, "__path__"):
            utils_pkg.__path__ = []  # type: ignore[attr-defined]
        return utils_pkg


def install_pull_raw_stubs(
    monkeypatch,
    *,
    load_sql_return_df: pd.DataFrame | None = None,
    load_sql_raises: Exception | None = None,
    connect_engine_raises: Exception | None = None,
):
    """
    Install minimal stubs so importing ct.io.pull_raw does NOT require the real
    ct.io.db_utils / ct.io.snapshots / ct.utils.logger stack.

    Returns a dict of call-recorders so tests can assert side effects.
    """
    # Ensure parent packages are importable (don’t create/override ct or ct.io)
    importlib.import_module("ct.io")

    # ---- ct.utils.logger ----
    _ensure_ct_utils_package()
    logger_mod = types.ModuleType("ct.utils.logger")
    logger_calls: list[tuple[str, str]] = []

    class _Logger:
        def info(self, msg, *args, **kwargs):
            logger_calls.append(("info", str(msg)))

        def error(self, msg, *args, **kwargs):
            logger_calls.append(("error", str(msg)))

        def warning(self, msg, *args, **kwargs):
            logger_calls.append(("warning", str(msg)))

    def get_logger(name: str):
        return _Logger()

    setattr(logger_mod, "get_logger", get_logger)
    sys.modules["ct.utils.logger"] = logger_mod

    # ---- ct.io.db_utils ----
    db_utils_mod = types.ModuleType("ct.io.db_utils")
    db_calls: dict[str, object] = {"connect_engine_args": [], "load_sql_args": []}

    @contextmanager
    def connect_engine(sql_params):
        db_calls["connect_engine_args"].append(sql_params) # type: ignore[arg-type]
        if connect_engine_raises is not None:
            raise connect_engine_raises
        yield object()

    def load_sql(sql_file_or_query, engine):
        db_calls["load_sql_args"].append((sql_file_or_query, engine)) # type: ignore[arg-type]
        if load_sql_raises is not None:
            raise load_sql_raises
        if load_sql_return_df is None:
            return pd.DataFrame()
        return load_sql_return_df

    setattr(db_utils_mod, "connect_engine", connect_engine)
    setattr(db_utils_mod, "load_sql", load_sql)
    sys.modules["ct.io.db_utils"] = db_utils_mod

    # ---- ct.io.snapshots ----
    snapshots_mod = types.ModuleType("ct.io.snapshots")
    snap_calls: dict[str, object] = {"write_snapshot_metadata": []}

    def write_snapshot_metadata(
        *,
        snapshot_dir: Path,
        snapshot_id: str,
        pull_cfg: dict,
        output_file: Path,
        row_count: int,
        config_path: str | None = None,
    ):
        snap_calls["write_snapshot_metadata"].append( # type: ignore[arg-type]
            {
                "snapshot_dir": snapshot_dir,
                "snapshot_id": snapshot_id,
                "pull_cfg": pull_cfg,
                "output_file": output_file,
                "row_count": row_count,
                "config_path": config_path,
            }
        )
        return snapshot_dir / "meta.yaml"

    def get_snapshot_id_from_path(path: Path) -> str:
        # Not used by pull_raw currently, but imported.
        return path.name

    setattr(snapshots_mod, "write_snapshot_metadata", write_snapshot_metadata)
    setattr(snapshots_mod, "get_snapshot_id_from_path", get_snapshot_id_from_path)
    sys.modules["ct.io.snapshots"] = snapshots_mod

    return {
        "logger_calls": logger_calls,
        "db_calls": db_calls,
        "snap_calls": snap_calls,
    }


def import_pull_raw(monkeypatch, **stub_kwargs):
    """
    Import ct.io.pull_raw after installing dependency stubs.

    Returns (module, call_recorders).
    """
    recorders = install_pull_raw_stubs(monkeypatch, **stub_kwargs)

    module_name = "ct.io.pull_raw"
    # Avoid half-imported leftovers
    if module_name in sys.modules:
        del sys.modules[module_name]

    mod = importlib.import_module(module_name)
    return mod, recorders


def test__write_sessions_to_csv_writes_csv_and_logs(monkeypatch, tmp_path):
    pull_raw_mod, rec = import_pull_raw(monkeypatch)

    out = tmp_path / "out.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    pull_raw_mod._write_sessions_to_csv(out, df)

    assert out.exists()
    text = out.read_text()
    # header should exist
    assert text.splitlines()[0].strip() == "a,b"
    # index should NOT be written
    assert not text.splitlines()[1].startswith("0,")

    assert any(
        lvl == "info" and "Query results saved to" in msg
        for (lvl, msg) in rec["logger_calls"]
    )


def test_pull_raw_happy_path_writes_output_and_metadata(monkeypatch, tmp_path):
    df = pd.DataFrame({"session_id": [101, 102], "val": [1.5, 2.5]})
    pull_raw_mod, rec = import_pull_raw(monkeypatch, load_sql_return_df=df)

    snapshot_dir = tmp_path / "snapshots" / "s1"
    pull_cfg = {
        "source": {
            # pull_raw passes this straight to connect_engine(...)
            "sql_params": str(tmp_path / "db_cfg.json"),
            "sql_file_path": "queries/pull.sql",
        },
        "output": {"filename": "raw.csv"},
    }

    out_path = pull_raw_mod.pull_raw(
        pull_cfg=pull_cfg,
        snapshot_dir=snapshot_dir,
        snapshot_id="sid_001",
        config_path="cfg.yaml",
    )

    # Output path returned correctly
    assert out_path == snapshot_dir / "raw.csv"
    assert out_path.exists()

    # Output contains expected header and rows (no index)
    content = out_path.read_text().splitlines()
    assert content[0].strip() == "session_id,val"
    assert content[1].strip() == "101,1.5"
    assert content[2].strip() == "102,2.5"

    # connect_engine called with sql_params
    assert rec["db_calls"]["connect_engine_args"] == [pull_cfg["source"]["sql_params"]]

    # load_sql called with sql_file_path and an engine object
    assert rec["db_calls"]["load_sql_args"][0][0] == pull_cfg["source"]["sql_file_path"]
    assert rec["db_calls"]["load_sql_args"][0][1] is not None

    # snapshot metadata written with correct row_count and file path
    calls = rec["snap_calls"]["write_snapshot_metadata"]
    assert len(calls) == 1
    call = calls[0]
    assert call["snapshot_dir"] == snapshot_dir
    assert call["snapshot_id"] == "sid_001"
    assert call["pull_cfg"] == pull_cfg
    assert call["output_file"] == out_path
    assert call["row_count"] == 2
    assert call["config_path"] == "cfg.yaml"

    # log messages exist
    assert any(lvl == "info" and "Database connection established" in msg for lvl, msg in rec["logger_calls"])
    assert any(lvl == "info" and "SQL query results loaded successfully" in msg for lvl, msg in rec["logger_calls"])


def test_pull_raw_wraps_failures_as_runtimeerror(monkeypatch, tmp_path):
    pull_raw_mod, rec = import_pull_raw(
        monkeypatch,
        load_sql_raises=ValueError("boom"),
    )

    snapshot_dir = tmp_path / "snapshots" / "s1"
    pull_cfg = {
        "source": {
            "sql_params": str(tmp_path / "db_cfg.json"),
            "sql_file_path": "queries/pull.sql",
        },
        "output": {"filename": "raw.csv"},
    }

    with pytest.raises(RuntimeError, match="Failed to load data from SQL"):
        pull_raw_mod.pull_raw(
            pull_cfg=pull_cfg,
            snapshot_dir=snapshot_dir,
            snapshot_id="sid_002",
        )

    # Should log an error before raising
    assert any(lvl == "error" and "Failed to load data from SQL" in msg for (lvl, msg) in rec["logger_calls"])

    # Should NOT attempt to write metadata on failure
    assert rec["snap_calls"]["write_snapshot_metadata"] == []