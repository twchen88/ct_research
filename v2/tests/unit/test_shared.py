
# tests/test_shared.py
import numpy as np
import pytest
import torch
import torch.nn as nn

import ct.experiments.shared as shared  # module under test


class TestDeterministicBackend:
    def test_sets_cudnn_deterministic_true(self):
        # flip it first to ensure the function changes it
        torch.backends.cudnn.deterministic = False
        shared.deterministic_backend()
        assert torch.backends.cudnn.deterministic is True


class TestLoadModel:
    def test_load_model_initializes_predictor_and_sets_eval(self, monkeypatch, tmp_path):
        # Dummy Predictor to capture calls/state
        class DummyPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1, bias=False)
                self.eval_called = False
                self.loaded_state = None

            def load_state_dict(self, state_dict, strict=True, assign=False):
                self.loaded_state = state_dict
                # Return a real _IncompatibleKeys object to match nn.Module's return type
                from torch.nn.modules.module import _IncompatibleKeys
                return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

            def eval(self):
                self.eval_called = True
                return super().eval()

        # Patch Predictor reference used inside shared.py
        monkeypatch.setattr(shared, "Predictor", DummyPredictor, raising=True)

        # Patch torch.load to return a fake state dict
        fake_state = {"linear.weight": torch.tensor([[1.0]])}
        monkeypatch.setattr(
            torch, 
            "load", 
            lambda path, map_location=None, **kwargs: fake_state, 
            raising=True
        )

        # Call load_model; any path will do since we patch torch.load
        model = shared.load_model("fake_checkpoint.pt", device="cpu")

        assert isinstance(model, DummyPredictor)
        assert model.eval_called is True
        assert model.loaded_state == fake_state


class TestInference:
    def test_inference_returns_numpy_with_model_forward(self):
        class SumModel(nn.Module):
            def forward(self, x: torch.Tensor):
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                return x.sum(dim=1)

        model = SumModel().eval()
        x = np.array([[1.0, 2.0, 3.0],
                      [0.5, 0.5, 0.5]], dtype=float)

        x_tensor = torch.from_numpy(x).float()
        out = shared.inference(model, x_tensor)
        # Expect row-wise sums as numpy
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert np.allclose(out, np.array([6.0, 1.5]))


class TestAddEncoding:
    def test_concatenates_encoding_columns(self):
        x = np.array([[0.1, 0.9]*14], dtype=float)  # shape (1,28)
        enc = np.zeros((1,14), dtype=int)
        enc[0, 3] = 1  # one-hot at column 3
        out = shared.add_encoding(x, enc)
        assert out.shape == (1, 42)
        # First 28 unchanged
        assert np.array_equal(out[0, 14:], x[0])
        # Last 14 equal to encoding
        assert np.array_equal(out[0, :14], enc[0])


class TestCreateSingleEncoding:
    def test_creates_one_hot_column_per_row(self):
        rows, cols, idx = 5, 7, 3
        mat = shared.create_single_encoding(rows, cols, idx)
        assert mat.shape == (rows, cols)
        # column idx should be all ones, others zeros
        assert np.array_equal(mat[:, idx], np.ones(rows, dtype=int))
        assert np.all(mat[:, :idx] == 0) and np.all(mat[:, idx+1:] == 0)
