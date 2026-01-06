import importlib
import sys
import types
from datetime import datetime, timezone

import yaml

def install_ct_utils_stubs(monkeypatch, *, git_hash="deadbeef", cfg_hash="cfg_hash", hostname="host1"):
    """
    Install minimal stub modules for ct.utils.* so ct.io.snapshots can import.

    Key detail: ct.utils must look like a *package* (have __path__), otherwise
    importing ct.utils.config_io (or any ct.utils.* submodule) will fail with
    "'ct.utils' is not a package".
    """

    # Ensure ct.utils exists AND behaves like a package.
    utils_pkg = sys.modules.get("ct.utils")
    if utils_pkg is None:
        utils_pkg = types.ModuleType("ct.utils")
        sys.modules["ct.utils"] = utils_pkg

    # Mark as package so submodule imports work.
    if not hasattr(utils_pkg, "__path__"):
        utils_pkg.__path__ = []  # type: ignore[attr-defined]

    # --- ct.utils.hashing ---
    hashing = types.ModuleType("ct.utils.hashing")

    def hash_dict(d):
        return cfg_hash

    setattr(hashing, "hash_dict", hash_dict)
    sys.modules["ct.utils.hashing"] = hashing

    # --- ct.utils.metadata ---
    metadata = types.ModuleType("ct.utils.metadata")

    def get_git_commit_hash():
        return git_hash

    setattr(metadata, "get_git_commit_hash", get_git_commit_hash)
    sys.modules["ct.utils.metadata"] = metadata

    # --- ct.utils.logger ---
    logger_mod = types.ModuleType("ct.utils.logger")
    calls = []

    class _Logger:
        def info(self, msg, *args, **kwargs):
            calls.append(("info", msg, args, kwargs))

    def get_logger(name):
        return _Logger()

    setattr(logger_mod, "get_logger", get_logger)
    sys.modules["ct.utils.logger"] = logger_mod

    # --- ct.utils.config_io (imported somewhere in your ct package) ---
    config_io = types.ModuleType("ct.utils.config_io")

    def read_yaml(path):
        return yaml.safe_load(open(path, "r"))

    def write_yaml(path, data):
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    setattr(config_io, "read_yaml", read_yaml)
    setattr(config_io, "write_yaml", write_yaml)
    sys.modules["ct.utils.config_io"] = config_io

    # socket.gethostname is used in write_snapshot_metadata
    monkeypatch.setattr("socket.gethostname", lambda: hostname, raising=True)

    return calls


def import_snapshots(monkeypatch, **stub_kwargs):
    """
    Import (or reload) ct.io.snapshots after installing ct.utils stubs.
    Returns (module, logger_calls).
    """
    logger_calls = install_ct_utils_stubs(monkeypatch, **stub_kwargs)

    module_name = "ct.io.snapshots"
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)

    return mod, logger_calls


def test_make_snapshot_id_deterministic(monkeypatch):
    snapshots, _ = import_snapshots(monkeypatch)

    class _FakeUUID:
        hex = "a1b2c3d4e5f6"  # short should be a1b2c3

    monkeypatch.setattr(snapshots.uuid, "uuid4", lambda: _FakeUUID(), raising=True)

    fixed = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return fixed

        @staticmethod
        def strftime(dt, fmt):
            return datetime.strftime(dt, fmt)

    monkeypatch.setattr(snapshots, "datetime", _FakeDateTime, raising=True)

    sid = snapshots.make_snapshot_id(prefix="raw")
    assert sid == "20240102T030405Z_raw_a1b2c3"


def test_get_snapshot_id_from_path(monkeypatch, tmp_path):
    snapshots, _ = import_snapshots(monkeypatch)

    d = tmp_path / "20240102T030405Z_raw_a1b2c3"
    d.mkdir()

    assert snapshots.get_snapshot_id_from_path(d) == "20240102T030405Z"


def test_write_latest_and_read_latest_snapshot_roundtrip(monkeypatch, tmp_path):
    snapshots, _ = import_snapshots(monkeypatch)

    snapshots.write_latest_snapshot(tmp_path, "snap_123")
    assert (tmp_path / "latest").read_text() == "snap_123"
    assert snapshots.read_latest_snapshot(tmp_path) == "snap_123"


def test_read_latest_snapshot_strips_whitespace(monkeypatch, tmp_path):
    snapshots, _ = import_snapshots(monkeypatch)

    (tmp_path / "latest").write_text("snap_123\n")
    assert snapshots.read_latest_snapshot(tmp_path) == "snap_123"


def test_write_snapshot_metadata_writes_expected_yaml(monkeypatch, tmp_path):
    snapshots, logger_calls = import_snapshots(
        monkeypatch,
        git_hash="0123456789abcdef",
        cfg_hash="hash123",
        hostname="my-host",
    )

    fixed = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return fixed

        @staticmethod
        def strftime(dt, fmt):
            return datetime.strftime(dt, fmt)

    monkeypatch.setattr(snapshots, "datetime", _FakeDateTime, raising=True)

    snapshot_dir = tmp_path / "snapdir"
    snapshot_dir.mkdir()

    output_file = snapshot_dir / "data.parquet"
    output_file.write_text("dummy")

    pull_cfg = {
        "source": {
            "kind": "sql",
            "sql_file_path": "queries/my.sql",
            "sql_params": {"a": 1},
        },
        "metadata": {"team": "platform"},
    }

    meta_path = snapshots.write_snapshot_metadata(
        snapshot_dir,
        snapshot_id="sid_001",
        pull_cfg=pull_cfg,
        output_file=output_file,
        row_count=42,
        config_path="cfg.yaml",
    )

    assert meta_path == snapshot_dir / "meta.yaml"

    data = yaml.safe_load(meta_path.read_text())

    assert data["snapshot_id"] == "sid_001"
    assert data["created_at"] == fixed.isoformat()

    assert data["source"] == {
        "kind": "sql",
        "sql_file_path": "queries/my.sql",
        "sql_params": {"a": 1},
    }

    assert data["output"] == {
        "filename": "data.parquet",
        "format": "parquet",
        "row_count": 42,
    }

    assert data["provenance"] == {
        "config_file": "cfg.yaml",
        "config_hash": "hash123",
        "git_commit_hash": "0123456789abcdef",
        "hostname": "my-host",
    }

    assert data["user_metadata"] == {"team": "platform"}
    assert any(call[0] == "info" for call in logger_calls)


def test_write_snapshot_metadata_defaults(monkeypatch, tmp_path):
    snapshots, _ = import_snapshots(monkeypatch, cfg_hash="x", git_hash="y", hostname="z")

    fixed = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return fixed

        @staticmethod
        def strftime(dt, fmt):
            return datetime.strftime(dt, fmt)

    monkeypatch.setattr(snapshots, "datetime", _FakeDateTime, raising=True)

    snapshot_dir = tmp_path / "snapdir"
    snapshot_dir.mkdir()

    output_file = snapshot_dir / "out.csv"
    output_file.write_text("dummy")

    pull_cfg = {"source": {"sql_file_path": "q.sql", "sql_params": None}}

    meta_path = snapshots.write_snapshot_metadata(
        snapshot_dir,
        snapshot_id="sid_002",
        pull_cfg=pull_cfg,
        output_file=output_file,
        row_count=1,
    )

    data = yaml.safe_load(meta_path.read_text())

    assert data["source"]["kind"] == "sql"
    assert data["source"]["sql_file_path"] == "q.sql"
    assert data["source"]["sql_params"] is None
    assert data["user_metadata"] == {}