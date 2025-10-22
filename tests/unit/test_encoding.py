
from __future__ import annotations
import importlib, sys, pathlib
import numpy as np
import pandas as pd
import pytest

def _import_encoding():
    try:
        return importlib.import_module("ct.data.encoding")
    except Exception:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        return importlib.import_module("encoding")

enc = _import_encoding()

class Test_filter_nonzero_rows:
    def test_happy_filters_rows(self):
        df = pd.DataFrame({"a":[0,1,0], "b":[0,0,2], "c":[3,0,0]})
        out = enc.filter_nonzero_rows(df, max_zeros=1)
        assert out.shape[0] == 1
        assert out.iloc[0]["a"] == 1

    def test_edge_max_zeros_zero(self):
        df = pd.DataFrame({"a":[0,1], "b":[0,1]})
        out = enc.filter_nonzero_rows(df, max_zeros=0)
        assert out.index.tolist() == [1]

class Test_create_missing_indicator:
    def test_happy_deterministic_with_seed(self, matrix_with_nans):
        encoded1 = enc.create_missing_indicator(matrix_with_nans.copy(), rand_seed=123)
        encoded2 = enc.create_missing_indicator(matrix_with_nans.copy(), rand_seed=123)
        assert np.array_equal(encoded1, encoded2)
        assert encoded1.shape == (2, 4)
        # For a non-NaN value, expect [v, 1-v]
        row0 = encoded1[0]
        assert row0[0] == pytest.approx(0.2)
        assert row0[1] == pytest.approx(0.8)

    def test_edge_all_nan(self):
        mat = np.array([[np.nan, np.nan]], dtype=np.float32)
        out = enc.create_missing_indicator(mat, rand_seed=0)
        assert out.shape == (1, 4)
        # each pair equal either (0,0) or (1,1)
        assert out[0,0] in (0,1) and out[0,1] in (0,1) and out[0,0] == out[0,1]

class Test_encode_target_data:
    def test_happy_multiplies_and_zeros_nans(self, target_and_encoding):
        target, enc_arr = target_and_encoding
        out = enc.encode_target_data(target, enc_arr)
        assert out[0,0] == pytest.approx(0.5)
        assert out[0,1] == 0.0
        assert out[1,0] == 0.0
        assert out[1,1] == pytest.approx(1.0)

    def test_edge_mismatched_shapes(self, target_and_encoding):
        target, enc_arr = target_and_encoding
        with pytest.raises(ValueError):
            _ = enc.encode_target_data(target[:, :1], enc_arr)
