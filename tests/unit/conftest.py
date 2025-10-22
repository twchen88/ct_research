
"""
Global fixtures for unit tests.
"""
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    return p

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a":[1,2,3], "b":[0.1, 0.2, 0.3]})

@pytest.fixture
def sample_array() -> np.ndarray:
    return np.array([[1.0, 0.0],[np.nan, 2.5]], dtype=float)
