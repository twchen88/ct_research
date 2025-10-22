# tests/integration/test_integration__00_pull_data.py
from __future__ import annotations
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "00_pull_data.py"

@pytest.mark.integration
def test_happy_runs_with_minimal_config(tmp_path, monkeypatch):
    # ---- Build fake ct/ package with step markers ----
    ct_root = tmp_path / "ct"
    (ct_root / "data").mkdir(parents=True)
    (ct_root / "utils").mkdir(parents=True)
    for p in (ct_root, ct_root / "data", ct_root / "utils"):
        (p / "__init__.py").write_text("", encoding="utf-8")

    # ct/data/db_utils.py
    (ct_root / "data" / "db_utils.py").write_text(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "def connect(*args, **kwargs):\n"
        "    Path(Path.cwd() / 'connect_called.txt').write_text('1', encoding='utf-8')\n"
        "    class _Conn:\n"
        "        def close(self):\n"
        "            Path(Path.cwd() / 'close_called.txt').write_text('1', encoding='utf-8')\n"
        "    return _Conn()\n"
        "def load_sql(query_or_path, con):\n"
        "    Path(Path.cwd() / 'load_sql_called.txt').write_text('1', encoding='utf-8')\n"
        "    return pd.DataFrame({'x':[1]})\n",
        encoding="utf-8",
    )

    # ct/data/data_io.py
    (ct_root / "data" / "data_io.py").write_text(
        "from pathlib import Path\n"
        "def write_sessions_to_csv(path, df):\n"
        "    p = Path(path)\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    df.to_csv(p, index=False)\n"
        "    (Path.cwd() / '_written_path.txt').write_text(str(p), encoding='utf-8')\n",
        encoding="utf-8",
    )

    # ct/utils/config_loading.py (your script imports this; it may or may not use it)
    (ct_root / "utils" / "config_loading.py").write_text(
        "import yaml\n"
        "from pathlib import Path\n"
        "def load_config(path):\n"
        "    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )

    # ct/utils/metadata.py
    (ct_root / "utils" / "metadata.py").write_text(
        "def get_git_commit_hash():\n"
        "    return 'TEST-GIT-HASH'\n",
        encoding="utf-8",
    )

    # Ensure our fake ct/ is imported first
    sys.path.insert(0, str(tmp_path))

    # ---- Config (TRAILING SLASH because script does `dest + filename`) ----
    out_csv = tmp_path / "out.csv"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        (
            f"output:\n"
            f"  dest: {str(tmp_path)}/\n"
            f"  filename: {out_csv.name}\n"
            f"source:\n"
            f"  sql_file_path: {tmp_path / 'q.sql'}\n"
            f"  sql_params: {{}}\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "q.sql").write_text("SELECT 1 as x", encoding="utf-8")

    # Simulate CLI
    monkeypatch.setattr(sys, "argv", ["00_pull_data.py", "--config", str(cfg)])

    # ---- Execute the script file-by-path ----
    spec = importlib.util.spec_from_file_location("pull_data_script", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs top-level code

    # ---- Diagnostics: which steps ran? ----
    connect_called = (tmp_path / "connect_called.txt").exists()
    load_sql_called = (tmp_path / "load_sql_called.txt").exists()
    written_log = tmp_path / "_written_path.txt"

    # If writer never ran, provide precise pointers
    assert connect_called, (
        "db_utils.connect() was never called. "
        "The script likely exited before attempting a DB connection. "
        "Check argument parsing / config loading."
    )
    assert load_sql_called, (
        "db_utils.load_sql() was never called. "
        "The script likely exited after connecting but before querying. "
        "Verify your fake 'sql_file_path' and any control flow between connect->load_sql."
    )
    assert written_log.exists(), (
        "Fake write_sessions_to_csv() was never called. "
        "The script likely exited after load_sql but before writing. "
        "Check the branch that builds output path and writes results."
    )

    # If we did write, verify file and contents
    written_path = Path(written_log.read_text(encoding="utf-8")).resolve()
    assert written_path.exists(), f"Expected CSV at {written_path}"
    df = pd.read_csv(written_path)
    assert list(df.columns) == ["x"]
    assert df["x"].tolist() == [1]

    # Optional: ensure it wrote to the intended location
    assert written_path == out_csv.resolve(), (
        f"Script wrote to {written_path}, but test expected {out_csv}. "
        "If your script intentionally writes elsewhere, update the expectation."
    )