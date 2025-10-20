# tests/test_unit.py
import numpy as np
import pytest
import torch
import torch.nn as nn

import src.experiments.aggregate_average as aa
import src.experiments.shared as shared


# --------------------- Shared (unit) ---------------------

@pytest.mark.unit
def test_deterministic_backend_sets_flag():
    torch.backends.cudnn.deterministic = False
    shared.deterministic_backend()
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.unit
def test_load_model_initializes_predictor_and_sets_eval(monkeypatch):
    class DummyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.eval_called = False
            self.loaded_state = None

        def load_state_dict(self, state_dict, strict=True, assign=False):
            self.loaded_state = state_dict
            # Return a torch._IncompatibleKeys object as expected by torch
            from torch.nn.modules.module import _IncompatibleKeys
            return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

        def eval(self):
            self.eval_called = True
            return super().eval()

    fake_state = {"w": torch.tensor([1.0])}
    monkeypatch.setattr(shared, "Predictor", DummyPredictor, raising=True)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None, **kw: fake_state, raising=True)

    model = shared.load_model("fake_checkpoint.pt", device="cpu")
    assert isinstance(model, DummyPredictor)
    assert model.eval_called is True
    assert model.loaded_state == fake_state


@pytest.mark.unit
def test_add_encoding_concatenates():
    scores = np.array([[0.1, 0.9]*14], dtype=float)  # (1,28)
    enc = np.zeros((1, 14), dtype=int)
    enc[0, 3] = 1
    out = shared.add_encoding(scores, enc)
    assert out.shape == (1, 42)
    assert np.array_equal(out[0, :14], enc[0])
    assert np.array_equal(out[0, 14:], scores[0])


@pytest.mark.unit
def test_create_single_encoding_one_hot():
    rows, cols, idx = 3, 7, 4
    mat = shared.create_single_encoding(rows, cols, idx)
    assert mat.shape == (rows, cols)
    assert np.array_equal(mat[:, idx], np.ones(rows, dtype=int))
    assert np.all(mat[:, :idx] == 0) and np.all(mat[:, idx+1:] == 0)


# --------------------- Aggregate Average (unit) ---------------------

@pytest.mark.unit
def test_filter_rows_by_sum_keeps_leq_threshold():
    data = np.array([[1,2,3],[0,0,5],[2,2,2]])
    filt, mask = aa.filter_rows_by_sum(data, slice(0,2), 3)  # sums: 3,0,4
    assert np.array_equal(mask, np.array([True, True, False]))
    assert np.array_equal(filt, data[mask])


@pytest.mark.unit
def test_find_missing_mask_basic():
    x1 = np.array([0., 1., 0.5, 0.0, 1.0])
    x2 = np.array([0., 1., 0.5, 1.0, 0.0])
    mask = aa.find_missing_mask(x1, x2, eps=0.0)
    assert np.array_equal(mask, np.array([True, True, False, False, False]))


class TestAssignRepeat_Unit:
    @staticmethod
    def _row(enc14, pairs14):
        enc = np.asarray(enc14, dtype=float)
        flat = np.asarray([v for xy in pairs14 for v in xy], dtype=float)
        return np.concatenate([enc, flat])

    @pytest.mark.unit
    def test_vacuous_true_all_enc_zero(self):
        enc = [0]*14
        pairs = [(0.0,0.0)]*14
        out = aa.assign_repeat(self._row(enc, pairs)[np.newaxis, :])
        assert bool(out[0])

    @pytest.mark.unit
    def test_required_valid_pass_and_invalid_fails(self):
        enc = [0]*14
        for i in (0,5,9): enc[i] = 1
        pairs = [(0.5,0.5)]*14
        assert bool(aa.assign_repeat(self._row(enc, pairs)[np.newaxis, :])[0])

        enc2 = [0]*14; enc2[3] = 1
        pairs2 = [(0.4,0.6)]*14; pairs2[3] = (0.0,0.0)
        assert not bool(aa.assign_repeat(self._row(enc2, pairs2)[np.newaxis, :])[0])


class TestCreateRandomEncoding_Unit:
    @staticmethod
    def _row_scores(pairs14):
        return np.asarray([v for xy in pairs14 for v in xy], dtype=float)

    @pytest.fixture(autouse=True)
    def _deterministic_rng(self, monkeypatch):
        monkeypatch.setattr(aa.np.random, "default_rng", lambda: np.random.default_rng(12345), raising=False)

    @pytest.mark.unit
    def test_repeat_selects_valid(self):
        pairs = [(0.4,0.6)]*14; pairs[4] = (0.0,0.0)
        out = aa.create_random_encoding(self._row_scores(pairs)[np.newaxis, :], "repeat")
        assert out.shape == (1,14)
        assert out.sum() == 1 and out[0,4] == 0

    @pytest.mark.unit
    def test_nonrepeat_selects_invalid(self):
        pairs = [(0.4,0.6)]*14; pairs[7] = (1.0,1.0)
        out = aa.create_random_encoding(self._row_scores(pairs)[np.newaxis, :], "nonrepeat")
        assert out.sum() == 1 and out[0,7] == 1

    @pytest.mark.unit
    def test_all_valid_or_all_invalid_edge(self):
        all_valid = self._row_scores([(0.5,0.5)]*14)[np.newaxis, :]
        all_invalid = self._row_scores([(0.0,0.0)]*14)[np.newaxis, :]
        assert np.array_equal(aa.create_random_encoding(all_valid, "nonrepeat"), np.zeros((1,14)))
        assert np.array_equal(aa.create_random_encoding(all_invalid, "repeat"), np.zeros((1,14)))


class TestSplitEncodingAndScores_Unit:
    @staticmethod
    def _row(enc, pairs):
        enc = np.asarray(enc, dtype=float)
        flat = np.asarray([v for xy in pairs for v in xy], dtype=float)
        return np.concatenate([enc, flat])

    @pytest.mark.unit
    def test_split_ok(self):
        enc = [1,0,1]+[0]*11
        pairs = [(0.4,0.6),(0.0,0.0),(0.5,0.5)] + [(0.0,0.0)]*11
        data = self._row(enc, pairs)[np.newaxis, :]
        e, s = aa.split_encoding_and_scores(data, dims=14)
        assert e.shape == (1,14) and s.shape == (1,28)
        assert np.array_equal(e[0, :3], np.array([1,0,1]))


class _SumModel(nn.Module):
    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.sum(dim=1)


@pytest.mark.unit
def test_find_random_predictions_unit_flow(monkeypatch):
    model = _SumModel().eval()
    data = np.array([[0.5,0.5]*14, [0.4,0.6]*14], dtype=float)

    enc_fixed = np.zeros((2,14), dtype=int); enc_fixed[0,3] = 1; enc_fixed[1,9] = 1
    monkeypatch.setattr(aa, "create_random_encoding", lambda d, rt: enc_fixed, raising=True)
    monkeypatch.setattr(aa, "add_encoding", lambda sc, en: np.hstack((en, sc)), raising=True)
    monkeypatch.setattr(aa, "inference", lambda m, x: x.sum(axis=1), raising=True)

    rand_enc, preds = aa.find_random_predictions(model, data, run_type="repeat")
    assert np.array_equal(rand_enc, enc_fixed)
    assert np.allclose(preds, np.hstack((enc_fixed, data)).sum(axis=1))


@pytest.mark.unit
def test_predict_all_domains_unit(monkeypatch):
    class DummyModel(nn.Module): pass
    model = DummyModel()

    x = np.array([[0.5,0.5]*14, [0.2,0.8]*14], dtype=float)
    y = np.zeros((2,14), dtype=float)

    def fake_create_single_encoding(rows, cols, domain):
        mat = np.zeros((rows, cols), dtype=int); mat[:, domain] = 1; return mat
    def fake_add_encoding(x_in, enc):
        dom = int(np.argmax(enc[0])); return np.full((x_in.shape[0], 1), float(dom))
    def fake_inference(mdl, x_single):
        return x_single.squeeze(-1)

    monkeypatch.setattr(aa, "create_single_encoding", fake_create_single_encoding, raising=True)
    monkeypatch.setattr(aa, "add_encoding", fake_add_encoding, raising=True)
    monkeypatch.setattr(aa, "inference", fake_inference, raising=True)

    mat = aa.predict_all_domains(model, x, y, loop_range=list(range(14)))
    assert mat.shape == (2,14)
    assert np.array_equal(mat[0], np.arange(14))
    assert np.array_equal(mat[1], np.arange(14))


@pytest.mark.unit
def test_max_prediction_from_difference_pair_with_14_domains():
    # Build inputs with D=14 to match any hard-coded reshape assumptions
    N, D = 2, 14
    diff = np.zeros((N, D)); diff[:, [1, 5, 9]] = [0.5, 0.8, 0.6]
    preds = np.zeros((N, D)); preds[:, [1, 5, 9]] = [0.6, 0.9, 0.7]
    # current matrix pairs: mark 0 and 9 as missing in row0/row1 respectively
    cur = np.zeros((N, D, 2)); 
    cur[0, 0] = [0,0]     # missing
    cur[0, 5] = [0.2,0.8] # valid
    cur[1, 9] = [1,1]     # missing
    cur[1, 5] = [0.3,0.7] # valid

    # repeat: pick among valid -> row0 pick domain 5 (0.9), row1 pick domain 5 (0.7)
    mv_rep, mi_rep = aa.max_prediction_from_difference_pair(diff, preds, cur, run_type="repeat")
    assert np.allclose(mv_rep[[0,1]], np.array([0.9, 0.7]))
    assert mi_rep[0] == 5 and mi_rep[1] == 5

    # nonrepeat: pick among missing -> row0 domain 0 (0.0), row1 domain 9 (0.0)
    mv_non, mi_non = aa.max_prediction_from_difference_pair(diff, preds, cur, run_type="nonrepeat")
    assert mi_non[0] == 0 and mi_non[1] == 9


@pytest.mark.unit
def test_reconstruct_max_matrices():
    max_vals = np.array([0.9, 0.0, 0.7])
    max_idx = np.array([2, np.nan, 0])
    shape = (3, 4)
    values_mat, onehot_mat = aa.reconstruct_max_matrices(max_vals, max_idx, shape)
    assert values_mat.shape == shape and onehot_mat.shape == shape
    assert values_mat[0,2] == 0.9 and onehot_mat[0,2] == 1
    assert np.all(values_mat[1] == 0) and np.all(onehot_mat[1] == 0)
    assert values_mat[2,0] == 0.7 and onehot_mat[2,0] == 1


@pytest.mark.unit
def test_filter_n_missing_exact():
    def row(enc, pairs):
        return np.concatenate([np.asarray(enc,float), np.asarray([v for xy in pairs for v in xy], float)])
    enc = [0]*14
    pairsA = [(0,0),(1,1),(0.5,0.5)] + [(0.4,0.6)]*11  # 2 missing
    pairsB = [(0.4,0.6)]*14                             # 0 missing
    pairsC = [(0,0)]*14                                 # 14 missing
    data = np.vstack([row(enc,pairsA), row(enc,pairsB), row(enc,pairsC)])
    out = aa.filter_n_missing(data, n_missing=2)
    assert out.shape[0] == 1 and np.array_equal(out[0], data[0])


@pytest.mark.unit
def test_extract_score_pairs_shape():
    scores = np.array([[0.5,0.5]*14, [0.2,0.8]*14])
    pairs = aa.extract_score_pairs(scores)
    assert pairs.shape == (2,14,2)
    assert np.allclose(pairs[0,0], np.array([0.5,0.5]))


@pytest.mark.unit
def test_compute_errors_mae_std():
    gt = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 1.5, 3.5])
    mae, gstd = aa.compute_errors(gt, pred)
    assert np.isclose(mae, (0.5+0.5+0.5)/3)
    assert np.isclose(gstd, np.std(gt))


@pytest.mark.unit
def test_filter_with_masks_sequential():
    data = np.arange(10)
    m1 = data % 2 == 0      # length 10
    # Apply m1 first so we can derive m2 on the filtered view
    d1 = data[m1]           # [0,2,4,6,8] length 5
    m2 = d1 < 8             # compatible length 5
    out = aa.filter_with_masks(data, [m1, m2])
    assert np.array_equal(out, np.array([0,2,4,6]))


@pytest.mark.unit
def test_compute_averages_and_stds_with_compatible_masks():
    cur = np.array([0.3, 0.4, 0.5, 0.6])
    fut = np.array([0.4, 0.6, 0.6, 0.9])
    mask1 = np.array([True, False, True, True])
    # build mask2 in the filtered space where mask1 is True
    diff_idx = np.where(mask1)[0]  # [0,2,3]
    m2_small = np.array([True, True, False])  # keep first two of the filtered
    # Expand m2_small back to full-length mask aligned with code expectations
    mask2 = np.zeros_like(mask1, dtype=bool)
    mask2[diff_idx[m2_small]] = True

    avg, std = aa.compute_averages_and_stds(cur, fut, [mask1, mask2])
    diff = (fut - cur)[mask1][m2_small]
    assert np.isclose(avg, np.mean(diff))
    assert np.isclose(std, np.std(diff))
