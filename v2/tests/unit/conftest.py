
"""
Global fixtures for the test suite.

These fixtures keep tests small and focused on behavior:
- tmp_csv_* helpers create small CSVs on disk
- capture_calls lets us assert how collaborators were called without a heavy mocking framework
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import types
import pandas as pd
import numpy as np
import pytest

# --- Tiny sample CSV creators ------------------------------------------------

@pytest.fixture
def tmp_csv_preprocessed(tmp_path: Path) -> Path:
    """Create a small CSV with columns expected by read_preprocessed_session_file."""
    cols = ["patient_id"] + \
           [f"domain {i} encoding" for i in range(1, 15)] + \
           [f"domain {i} score" for i in range(1, 15)] + \
           [f"domain {i} target" for i in range(1, 15)] + \
           ["time_stamp"]
    # Build a single row of valid data
    row = [123] + \
          [1 for _ in range(14)] + \
          [0.5 for _ in range(14)] + \
          [1.0 for _ in range(14)] + \
          [1700000000]
    df = pd.DataFrame([row], columns=cols)
    p = tmp_path / "preprocessed.csv"
    df.to_csv(p, index=False)
    return p

@pytest.fixture
def tmp_csv_raw(tmp_path: Path) -> Path:
    """Raw CSV with columns your function expects, including start_time."""
    import pandas as pd
    import numpy as np

    n = 101
    df = pd.DataFrame({
        "id": np.arange(n, dtype=np.int32),
        "patient_id": np.arange(n, dtype=np.int32),
        "task_type_id": np.random.randint(0, 3, size=n).astype(np.int16),
        "task_level": np.random.randint(1, 5, size=n).astype(np.int16),
        "domain_ids": ["1,2"] * n,
        "domain_scores": ["0.2,0.8"] * n,
        "start_time": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
    })
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p

# --- A minimal call recorder -------------------------------------------------

class CallRecorder:
    def __init__(self):
        self.calls = []
    def __call__(self, *a, **k):
        self.calls.append((a, k))
        # Can return a sentinel if needed
        return getattr(self, "return_value", None)

@pytest.fixture
def capture_calls():
    """Return a CallRecorder to assert collaborator usage (used for monkeypatching)."""
    return CallRecorder()
