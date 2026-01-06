import importlib
import sys
import types
import pytest
import json

def _write_cfg(tmp_path, cfg: dict) -> str:
    p = tmp_path / "db_cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)

def _ensure_ct_utils_package():
    """
    Ensure ct.utils exists as a *package* without shadowing the real ct package.
    Prefer importing the real ct.utils; only synthesize ct.utils if import fails.
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


def install_ct_utils_stubs(monkeypatch, *, cfg=None, cfg_hash="cfg_hash", git_hash="deadbeef"):
    """
    Stub ct.utils.* deps required by ct.io.db_utils.

    IMPORTANT:
    - Do NOT create/override `ct` or `ct.io` (that can shadow the real package).
    - Do ensure `ct.utils` is package-like so `ct.utils.*` imports work.
    """
    if cfg is None:
        cfg = {}

    _ensure_ct_utils_package()

    # --- ct.utils.config_io ---
    config_io = types.ModuleType("ct.utils.config_io")

    def load_json_config(filename: str):
        return cfg

    setattr(config_io, "load_json_config", load_json_config)
    sys.modules["ct.utils.config_io"] = config_io

    # --- ct.utils.metadata ---
    metadata = types.ModuleType("ct.utils.metadata")

    def get_git_commit_hash():
        return git_hash

    setattr(metadata, "get_git_commit_hash", get_git_commit_hash)
    sys.modules["ct.utils.metadata"] = metadata

    # --- ct.utils.hashing ---
    hashing = types.ModuleType("ct.utils.hashing")

    def hash_dict(d):
        return cfg_hash

    setattr(hashing, "hash_dict", hash_dict)
    sys.modules["ct.utils.hashing"] = hashing

    # --- ct.utils.logger ---
    logger_mod = types.ModuleType("ct.utils.logger")
    calls = []

    class _Logger:
        def info(self, msg, *args, **kwargs):
            calls.append(("info", msg, args, kwargs))

        def warning(self, msg, *args, **kwargs):
            calls.append(("warning", msg, args, kwargs))

        def error(self, msg, *args, **kwargs):
            calls.append(("error", msg, args, kwargs))

    def get_logger(name: str):
        return _Logger()

    setattr(logger_mod, "get_logger", get_logger)
    sys.modules["ct.utils.logger"] = logger_mod

    return calls


def import_db_utils(monkeypatch, *, cfg=None):
    """
    Import ct.io.db_utils after installing ct.utils stubs.

    Robustness:
    - If a prior import failed, ct.io.db_utils may be partially in sys.modules.
      We delete it before importing to avoid "parent ct.io not in sys.modules".
    - Import ct.io (parent) first.
    """
    logger_calls = install_ct_utils_stubs(monkeypatch, cfg=cfg)

    module_name = "ct.io.db_utils"

    # Clear any half-imported leftovers
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Ensure parent is loaded
    importlib.import_module("ct.io")

    mod = importlib.import_module(module_name)
    return mod, logger_calls


def test_connect_engine_direct_mode(monkeypatch, tmp_path):
    cfg = {
        "connection_mode": "direct",
        "sqlalchemy_url": "sqlite:///tmp.db",
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    db_utils, logger_calls = import_db_utils(monkeypatch, cfg=cfg)

    created = {}

    def fake_create_engine(url, pool_pre_ping=False, **kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        created["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(db_utils, "create_engine", fake_create_engine, raising=True)

    with db_utils.connect_engine(cfg_path) as engine:
        assert engine is not None

    assert created["url"] == "sqlite:///tmp.db"
    assert created["pool_pre_ping"] is True
    assert any(c[0] == "info" for c in logger_calls)


def test_connect_engine_ssh_tunnel_mode_builds_url_and_stops_tunnel(monkeypatch, tmp_path):
    cfg = {
        "connection_mode": "ssh_tunnel",
        "ssh_address_or_host": "bastion.example.com",
        "ssh_user": "ec2-user",
        "key_path": "/tmp/fake.pem",
        "host_name": "db.internal",
        "port": 3306,
        "username": "dbuser",
        "password": "p@ss w/space",
        "database": "mydb",
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    db_utils, logger_calls = import_db_utils(monkeypatch, cfg=cfg)

    tunnel = {"started": False, "stopped": False, "kwargs": None}

    class FakeTunnel:
        local_bind_port = 5555

        def __init__(self, *args, **kwargs):
            tunnel["kwargs"] = kwargs

        def start(self):
            tunnel["started"] = True

        def stop(self):
            tunnel["stopped"] = True

    created = {}

    def fake_create_engine(url, pool_pre_ping=False, **kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        created["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(db_utils, "SSHTunnelForwarder", FakeTunnel, raising=True)
    monkeypatch.setattr(db_utils, "create_engine", fake_create_engine, raising=True)

    with db_utils.connect_engine(cfg_path) as engine:
        assert engine is not None
        assert tunnel["started"] is True
        assert tunnel["stopped"] is False

    assert tunnel["stopped"] is True
    assert created["url"].startswith(
        "mysql+pymysql://dbuser:p%40ss+w%2Fspace@127.0.0.1:5555"
    )
    assert created["url"].endswith("/mydb")
    assert created["pool_pre_ping"] is True
    assert any(c[0] == "info" for c in logger_calls)


def test_connect_engine_ssh_tunnel_without_database(monkeypatch, tmp_path):
    cfg = {
        "connection_mode": "ssh_tunnel",
        "ssh_address_or_host": "bastion.example.com",
        "ssh_user": "ec2-user",
        "key_path": "/tmp/fake.pem",
        "host_name": "db.internal",
        "username": "dbuser",
        "password": "pass",
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    db_utils, _ = import_db_utils(monkeypatch, cfg=cfg)

    class FakeTunnel:
        local_bind_port = 6000

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    created = {}

    def fake_create_engine(url, pool_pre_ping=False, **kwargs):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        return object()

    monkeypatch.setattr(db_utils, "SSHTunnelForwarder", FakeTunnel, raising=True)
    monkeypatch.setattr(db_utils, "create_engine", fake_create_engine, raising=True)

    with db_utils.connect_engine(cfg_path):
        pass

    assert created["url"] == "mysql+pymysql://dbuser:pass@127.0.0.1:6000"


def test_connect_engine_unknown_mode_raises(monkeypatch, tmp_path):
    cfg = {"connection_mode": "wat"}
    cfg_path = _write_cfg(tmp_path, cfg)

    db_utils, _ = import_db_utils(monkeypatch, cfg=cfg)

    with pytest.raises(ValueError):
        with db_utils.connect_engine(cfg_path):
            pass

def test_load_sql_runs_query_string(monkeypatch):
    cfg = {"connection_mode": "direct", "sqlalchemy_url": "sqlite://"}
    db_utils, _ = import_db_utils(monkeypatch, cfg=cfg)

    calls = {"sql": None, "con": None}

    def fake_read_sql(sql, con, *args, **kwargs):
        calls["sql"] = sql
        calls["con"] = con
        return "DF"

    monkeypatch.setattr(db_utils.pd, "read_sql", fake_read_sql, raising=True)

    con = object()
    out = db_utils.load_sql("SELECT 1", con)

    assert out == "DF"
    assert calls["sql"] == "SELECT 1"
    assert calls["con"] is con


def test_load_sql_reads_sql_file(monkeypatch, tmp_path):
    cfg = {"connection_mode": "direct", "sqlalchemy_url": "sqlite://"}
    db_utils, _ = import_db_utils(monkeypatch, cfg=cfg)

    sql_file = tmp_path / "q.sql"
    sql_file.write_text("SELECT 42;")

    calls = {"sql": None, "con": None}

    def fake_read_sql(sql, con, *args, **kwargs):
        calls["sql"] = sql
        calls["con"] = con
        return "DF"

    monkeypatch.setattr(db_utils.pd, "read_sql", fake_read_sql, raising=True)

    con = object()
    out = db_utils.load_sql(str(sql_file), con)

    assert out == "DF"
    assert calls["sql"] == "SELECT 42;"
    assert calls["con"] is con