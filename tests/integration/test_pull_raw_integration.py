import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_json(tmp_path: Path, name: str, data: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


@pytest.mark.integration
def test_pull_raw_pipeline_direct_writes_csv_and_meta(monkeypatch, tmp_path):
    """
    Integration scope (module-level, script-like workflow):
      pull_raw.pull_raw -> db_utils.connect_engine/load_sql -> snapshots.write_snapshot_metadata

    Emulate the script's orchestration:
      - choose snapshot_id
      - compute snapshot_dir from snapshot.root
      - call pull_raw.pull_raw(...)
      - do NOT update latest in this test
    """
    from ct.io import pull_raw
    from ct.io import db_utils
    from ct.io import snapshots

    # ---- Deterministic provenance/time for snapshots metadata ----
    monkeypatch.setattr("socket.gethostname", lambda: "host-test", raising=True)
    monkeypatch.setattr(snapshots, "get_git_commit_hash", lambda: "deadbeef", raising=True)
    monkeypatch.setattr(snapshots, "hash_dict", lambda d: "cfg_hash_123", raising=True)

    fixed = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return fixed

        @staticmethod
        def strftime(dt, fmt):
            return datetime.strftime(dt, fmt)

    monkeypatch.setattr(snapshots, "datetime", _FakeDateTime, raising=True)

    # ---- External edge mock: DB engine + query ----
    fake_engine = object()

    def fake_create_engine(url, pool_pre_ping=False, **kwargs):
        # connect_engine should read sqlalchemy_url from db_cfg.json
        assert url == "sqlite:///tmp.db"
        assert pool_pre_ping is True
        return fake_engine

    monkeypatch.setattr(db_utils, "create_engine", fake_create_engine, raising=True)

    df = pd.DataFrame({"session_id": [1, 2], "val": [0.1, 0.2]})

    def fake_read_sql(sql, con, *args, **kwargs):
        # db_utils.load_sql should ultimately call pd.read_sql with the engine
        assert con is fake_engine
        return df

    monkeypatch.setattr(db_utils.pd, "read_sql", fake_read_sql, raising=True)

    # ---- Real config artifacts that the modules read ----
    db_cfg_path = _write_json(
        tmp_path,
        "db_cfg.json",
        {"connection_mode": "direct", "sqlalchemy_url": "sqlite:///tmp.db"},
    )
    sql_path = tmp_path / "q.sql"
    sql_path.write_text("SELECT 1;")

    # PullRawConfig schema
    pull_cfg = {
        "metadata": {"team": "platform"},
        "source": {"sql_params": db_cfg_path, "sql_file_path": str(sql_path)},
        "output": {"filename": "raw.csv"},
        "snapshot": {"root": str(tmp_path / "snapshots"), "update_latest": False},
        "logging": {},
    }

    snapshot_id = "SID_TEST_001"
    snapshot_root = Path(pull_cfg["snapshot"]["root"])
    snapshot_dir = snapshot_root / snapshot_id

    out_path = pull_raw.pull_raw(
        pull_cfg=pull_cfg,
        snapshot_dir=snapshot_dir,
        snapshot_id=snapshot_id,
        config_path="cfg.yaml",
    )

    # ---- Assert CSV ----
    assert out_path == snapshot_dir / "raw.csv"
    assert out_path.exists()

    lines = out_path.read_text().splitlines()
    assert lines[0] == "session_id,val"
    assert lines[1] == "1,0.1"
    assert lines[2] == "2,0.2"

    # ---- Assert metadata YAML ----
    meta_path = snapshot_dir / "meta.yaml"
    assert meta_path.exists()

    meta = yaml.safe_load(meta_path.read_text())
    assert meta["snapshot_id"] == snapshot_id
    assert meta["created_at"] == fixed.isoformat()
    assert meta["output"]["filename"] == "raw.csv"
    assert meta["output"]["format"] == "csv"
    assert meta["output"]["row_count"] == 2
    assert meta["provenance"]["config_file"] == "cfg.yaml"
    assert meta["provenance"]["config_hash"] == "cfg_hash_123"
    assert meta["provenance"]["git_commit_hash"] == "deadbeef"
    assert meta["provenance"]["hostname"] == "host-test"
    assert meta["user_metadata"] == {"team": "platform"}

    # ---- Assert latest was NOT written ----
    assert not (snapshot_root / "latest").exists()


@pytest.mark.integration
def test_pull_raw_pipeline_update_latest_writes_latest_pointer(monkeypatch, tmp_path):
    """
    Same pipeline, but simulates the script behavior of writing latest when enabled.
    """
    from ct.io import pull_raw
    from ct.io import db_utils
    from ct.io import snapshots

    # DB boundary
    fake_engine = object()
    monkeypatch.setattr(db_utils, "create_engine", lambda *a, **k: fake_engine, raising=True)
    monkeypatch.setattr(db_utils.pd, "read_sql", lambda *a, **k: pd.DataFrame({"x": [1]}), raising=True)

    db_cfg_path = _write_json(
        tmp_path,
        "db_cfg.json",
        {"connection_mode": "direct", "sqlalchemy_url": "sqlite:///tmp.db"},
    )
    sql_path = tmp_path / "q.sql"
    sql_path.write_text("SELECT 1;")

    pull_cfg = {
        "metadata": {},
        "source": {"sql_params": db_cfg_path, "sql_file_path": str(sql_path)},
        "output": {"filename": "raw.csv"},
        "snapshot": {"root": str(tmp_path / "snapshots"), "update_latest": True},
        "logging": {},
    }

    snapshot_id = "SID_TEST_002"
    snapshot_root = Path(pull_cfg["snapshot"]["root"])
    snapshot_dir = snapshot_root / snapshot_id

    pull_raw.pull_raw(
        pull_cfg=pull_cfg,
        snapshot_dir=snapshot_dir,
        snapshot_id=snapshot_id,
        config_path="cfg.yaml",
    )

    # Script-like behavior: update latest pointer if enabled
    if pull_cfg.get("snapshot", {}).get("update_latest", False):
        snapshots.write_latest_snapshot(snapshot_root, snapshot_id)

    assert (snapshot_root / "latest").read_text().strip() == snapshot_id