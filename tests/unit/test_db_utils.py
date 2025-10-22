
from __future__ import annotations
import importlib, sys, pathlib
import pandas as pd
import json
import pytest
import sshtunnel
import pymysql

def _import_db_utils():
    try:
        return importlib.import_module("ct.data.db_utils")
    except Exception:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("db_utils")

dbu = _import_db_utils()

class DummyConn:
    pass

class Test_load_sql:
    def test_happy_query_string(self, monkeypatch):
        sentinel = pd.DataFrame({"x":[1]})
        calls = {}
        def fake_read_sql(q, con):
            calls["q"] = q; calls["con"] = con
            return sentinel
        monkeypatch.setattr(pd, "read_sql", fake_read_sql, raising=True)
        out = dbu.load_sql("SELECT 1 as x", DummyConn())
        assert out.equals(sentinel)
        assert calls["q"].lower().startswith("select")

    def test_happy_sql_file(self, tmp_path, monkeypatch):
        qpath = tmp_path / "q.sql"
        qpath.write_text("SELECT 42 AS answer", encoding="utf-8")
        sentinel = pd.DataFrame({"answer":[42]})
        monkeypatch.setattr(pd, "read_sql", lambda q, con: sentinel, raising=True)
        out = dbu.load_sql(str(qpath), DummyConn())
        assert out.equals(sentinel)

class Test_connect:
    def test_happy_reads_yaml_and_calls_pymysql(self, tmp_path, monkeypatch):

        # --- 1) Write config JSON using YOUR field names ---
        cfg = tmp_path / "db.json"
        config_dict = {
            "username": "u",
            "password": "p",
            "host_name": "localhost",   # your field
            "database": "d",            # include only if your code reads it
            "port": 3306,               # include only if your code reads it

            # SSH fields trigger the tunnel branch
            "ssh_address_or_host": "123",
            "ssh_user": "sshu",
            "key_path": "key.pem"
        }
        cfg.write_text(json.dumps(config_dict), encoding="utf-8")

        # --- 2) Fake tunnel that records the constructor args/kwargs ---
        tunnel_calls = []

        class DummyPKey:
            def get_fingerprint(self): return b"\x00" * 16
            def get_name(self): return "rsa"

        class FakeTunnel:
            def __init__(self, *args, **kwargs):
                # record invocation for assertions
                tunnel_calls.append({"args": args, "kwargs": kwargs})
                # typical attributes your code might read
                self.local_bind_port = 33306
                self.local_bind_host = "127.0.0.1"
            def start(self): pass
            def stop(self): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            @staticmethod
            def get_keys(*args, **kwargs):
                # (pkey, passphrase)
                return (DummyPKey(), None)

        # Patch BOTH symbols (module-level and db_utils import)
        monkeypatch.setattr(sshtunnel, "SSHTunnelForwarder", FakeTunnel, raising=True)
        monkeypatch.setattr(dbu, "SSHTunnelForwarder", FakeTunnel, raising=False)

        # --- 3) Fake pymysql.connect and capture kwargs ---
        captured = {}
        class DummyConn: pass

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return DummyConn()

        monkeypatch.setattr(pymysql, "connect", fake_connect, raising=True)

        # --- 4) Call the code under test ---
        conn = dbu.connect(str(cfg))
        assert isinstance(conn, DummyConn)

        # --- 5) Assert SSH address (and optionally port) passed to the tunnel ---
        # --- 5) Assert SSH address (and optionally port) passed to the tunnel ---
        import pprint

        assert tunnel_calls, "SSHTunnelForwarder was never constructed"
        call = tunnel_calls[0]
        expected_ssh_addr = config_dict["ssh_address_or_host"]

        # Try to extract (ssh_host, ssh_port) from the captured constructor call
        ssh_host = None
        ssh_port = None

        # Common pattern: first positional arg is (host, port)
        if call["args"]:
            first = call["args"][0]
            # e.g., ('123', 22)
            if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], str):
                ssh_host, ssh_port = first

        # Alternative pattern: kwargs
        if ssh_host is None:
            if "ssh_address_or_host" in call["kwargs"]:
                ssh_host = call["kwargs"]["ssh_address_or_host"]
                ssh_port = call["kwargs"].get("ssh_port", 22)
            elif "ssh_address" in call["kwargs"]:
                ssh_host = call["kwargs"]["ssh_address"]
                ssh_port = call["kwargs"].get("ssh_port", 22)

        # If still not found, fail with a helpful dump
        if ssh_host is None:
            raise AssertionError(
                "Could not find SSH address in SSHTunnelForwarder call.\n"
                f"args={call['args']}\nkwargs={pprint.pformat(call['kwargs'])}"
            )

        # Now assert host (and optionally port)
        assert ssh_host == expected_ssh_addr, f"Expected ssh host {expected_ssh_addr!r}, got {ssh_host!r}"
        # If you want to check port too:
        assert ssh_port in (22, config_dict.get("ssh_port", 22)), f"Unexpected ssh port: {ssh_port}"

        # --- 6) Assert DB connect kwargs after mapping (tunneled host) ---
        # If tunnel is used, host should be the local bind host (we set 127.0.0.1)
        assert captured["host"] in ("127.0.0.1", "localhost")
        # Your code should map username->user and host_name->host (before tunnel)
        assert captured["user"] == config_dict["username"]
        # Only assert these if your code forwards them to pymysql.connect:
        if "database" in config_dict:
            assert captured.get("database") == config_dict["database"]
        if "port" in config_dict:
            # Some code overwrites port to local_bind_port; both are acceptable depending on impl
            assert captured.get("port") in (config_dict["port"], 33306)