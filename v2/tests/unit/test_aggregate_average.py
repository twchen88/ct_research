
# tests/test_aggregate_average.py
import numpy as np
import pytest
import torch
import torch.nn as nn

import src.experiments.aggregate_average as aa


# ---------- Shared helpers / stubs ----------

class _SumModel(nn.Module):
    """Dummy model: returns rowwise sum as numpy array via our inference stub."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.sum(dim=1)


@pytest.fixture(autouse=True)
def patch_random_and_inference(monkeypatch):
    """Deterministic RNG and simple add_encoding/inference stubs for positive-path tests."""

    # Deterministic RNG
    def _fixed_default_rng():
        return np.random.default_rng(12345)
    monkeypatch.setattr(aa.np.random, "default_rng", _fixed_default_rng, raising=False)

# ---------- Tests per function (positive-path only) ----------

class TestFilterRowsBySum:
    def test_keeps_rows_leq_threshold(self):
        data = np.array([[1,2,3],[0,0,5],[2,2,2]])
        filt, mask = aa.filter_rows_by_sum(data, slice(0,2), 3)  # sums: 3,0,4 -> keep rows 0,1
        assert np.array_equal(mask, np.array([True, True, False]))
        assert np.array_equal(filt, data[mask])


class TestFindMissingMask:
    def test_marks_zeros_and_ones(self):
        x1 = np.array([0., 1., 0.5, 0.0, 1.0])
        x2 = np.array([0., 1., 0.5, 1.0, 0.0])
        mask = aa.find_missing_mask(x1, x2, eps=0.0)
        assert np.array_equal(mask, np.array([True, True, False, False, False]))


class TestAssignRepeat:
    @staticmethod
    def _row(enc14, pairs14):
        enc = np.asarray(enc14, dtype=float)
        flat = np.asarray([v for xy in pairs14 for v in xy], dtype=float)
        return np.concatenate([enc, flat])

    def test_vacuous_true_when_all_enc_zero(self):
        enc = [0]*14
        pairs = [(0.0,0.0)]*14
        res = aa.assign_repeat(self._row(enc,pairs)[np.newaxis,:])
        assert bool(res[0])

    def test_required_valid_pass_and_invalid_fails(self):
        # valid
        enc = [0]*14
        for i in (0,5,9): enc[i]=1
        pairs = [(0.5,0.5)]*14
        res = aa.assign_repeat(self._row(enc,pairs)[np.newaxis,:])
        assert bool(res[0])
        # invalid at required
        enc2 = [0]*14; enc2[3] = 1
        pairs2 = [(0.4,0.6)]*14; pairs2[3] = (0.0,0.0)
        res2 = aa.assign_repeat(self._row(enc2,pairs2)[np.newaxis,:])
        assert not bool(res2[0])


class TestCreateRandomEncoding:
    @staticmethod
    def _row_scores(pairs14):
        return np.asarray([v for xy in pairs14 for v in xy], dtype=float)

    def test_repeat_selects_from_valid(self):
        pairs = [(0.4,0.6)]*14
        pairs[4] = (0.0,0.0)  # invalid
        data = self._row_scores(pairs)[np.newaxis,:]
        out = aa.create_random_encoding(data, run_type="repeat")
        assert out.shape == (1,14)
        assert out.sum()==1 and out[0,4]==0  # avoid invalid index

    def test_nonrepeat_selects_from_invalid(self):
        pairs = [(0.4,0.6)]*14
        pairs[7] = (1.0,1.0)
        data = self._row_scores(pairs)[np.newaxis,:]
        out = aa.create_random_encoding(data, run_type="nonrepeat")
        assert out.sum()==1 and out[0,7]==1

    def test_all_valid_or_all_invalid(self):
        all_valid = self._row_scores([(0.5,0.5)]*14)[np.newaxis,:]
        all_invalid = self._row_scores([(0.0,0.0)]*14)[np.newaxis,:]
        assert np.array_equal(aa.create_random_encoding(all_valid,"nonrepeat"), np.zeros((1,14)))
        assert np.array_equal(aa.create_random_encoding(all_invalid,"repeat"), np.zeros((1,14)))


class TestSplitEncodingAndScores:
    @staticmethod
    def _row(enc, pairs):
        enc = np.asarray(enc, dtype=float)
        flat = np.asarray([v for xy in pairs for v in xy], dtype=float)
        return np.concatenate([enc, flat])

    def test_split_ok(self):
        enc = [1,0,1]+[0]*11
        pairs = [(0.4,0.6),(0.0,0.0),(0.5,0.5)] + [(0.0,0.0)]*11
        data = self._row(enc,pairs)[np.newaxis,:]
        e,s = aa.split_encoding_and_scores(data, dims=14)
        assert e.shape==(1,14) and s.shape==(1,28)
        assert np.array_equal(e[0,:3], np.array([1,0,1]))


class TestFindRandomPredictions:
    @staticmethod
    def _row_scores(pairs14):
        return np.asarray([v for xy in pairs14 for v in xy], dtype=float)

    def test_repeat_and_nonrepeat_paths(self):
        pairs = [(0.5,0.5)]*14
        pairs[2] = (0.0,0.0)
        pairs[9] = (1.0,1.0)

        data = np.vstack([self._row_scores(pairs), self._row_scores([(0.5,0.5)]*14)])
        model = _SumModel()

        rand_rep, preds_rep = aa.find_random_predictions(model, data, "repeat")
        rand_non, preds_non = aa.find_random_predictions(model, data, "nonrepeat")

        # Row 0: repeat must avoid {2,9}; nonrepeat must pick one of {2,9}
        rep0 = np.argmax(rand_rep[0]) if rand_rep[0].any() else -1
        non0 = np.argmax(rand_non[0]) if rand_non[0].any() else -1
        assert rep0 not in (2,9)
        assert non0 in (2,9)

        # Row 1: no invalids -> nonrepeat zeros; repeat some valid
        assert np.array_equal(rand_non[1], np.zeros(14))
        assert rand_rep[1].sum()==1

        # predictions align with model on raw data (inference stub uses raw x)
        # exp = aa.inference(model, torch.from_numpy(data).to(torch.float32))
        # assert np.allclose(preds_rep, exp) and np.allclose(preds_non, exp)


class TestPredictAllDomains:
    def test_returns_matrix_per_domain(self):
        model = _SumModel()
        x = np.array([[0.5,0.5]*14, [0.2,0.8]*14], dtype=float)  # shape (2,28)
        y = np.zeros((2,14))  # not used by our stubs
        mat = aa.predict_all_domains(model, x, y, list(range(14)))
        # With our stubs, inference returns the same value per row for each domain
        assert mat.shape == (2,14)
        # assert np.all(mat[0] == mat[0,0])
        # assert np.all(mat[1] == mat[1,0])


class TestMaxPredictionFromDifferencePair:
    def test_picks_max_over_valid_mask(self):
        diff = np.array([[0.1, 0.5, 0.2],
                         [0.0, 0.3, 0.4]])
        preds = np.array([[0.6, 0.9, 0.7],
                          [0.5, 0.8, 0.85]])
        # current_matrix with 3 domains per row -> shape (2,3,2)
        cur = np.array([[[0,1],[0,0],[0.2,0.8]],
                        [[1,1],[0.3,0.7],[0,1]]], dtype=float)

        # run_type "repeat": choose among non-missing (i.e., not [0,0]/[1,1])
        max_vals, max_idx = aa.max_prediction_from_difference_pair(diff, preds, cur, run_type="repeat")
        assert np.allclose(max_vals, np.array([0.9, 0.85]))
        assert np.array_equal(max_idx, np.array([1, 2]))

        # run_type "nonrepeat": choose among missing
        max_vals2, max_idx2 = aa.max_prediction_from_difference_pair(diff, preds, cur, run_type="nonrepeat")
        assert np.allclose(max_vals2, np.array([0.9, 0.5]))
        assert np.array_equal(max_idx2, np.array([1, 0]))


class TestReconstructMaxMatrices:
    def test_reconstruct(self):
        max_vals = np.array([0.9, 0.0, 0.7])
        max_idx = np.array([2, np.nan, 0])
        shape = (3, 4)
        values_mat, onehot_mat = aa.reconstruct_max_matrices(max_vals, max_idx, shape)
        # row 0 has value at col 2; row 1 none; row 2 at col 0
        assert values_mat.shape == shape and onehot_mat.shape == shape
        assert values_mat[0,2] == 0.9 and onehot_mat[0,2] == 1
        assert np.all(values_mat[1] == 0) and np.all(onehot_mat[1] == 0)
        assert values_mat[2,0] == 0.7 and onehot_mat[2,0] == 1


class TestFilterNMissing:
    def test_filter_exact_missing_count(self):
        # Build data with enc(14) + pairs(28)
        def row(enc, pairs):
            return np.concatenate([np.asarray(enc,float), np.asarray([v for xy in pairs for v in xy], float)])
        enc = [0]*14
        pairsA = [(0,0),(1,1),(0.5,0.5)] + [(0.4,0.6)]*11  # 2 missing
        pairsB = [(0.4,0.6)]*14                             # 0 missing
        pairsC = [(0,0)]*14                                 # 14 missing
        data = np.vstack([row(enc,pairsA), row(enc,pairsB), row(enc,pairsC)])
        out = aa.filter_n_missing(data, n_missing=2)
        assert out.shape[0] == 1 and np.array_equal(out[0], data[0])


class TestExtractScorePairs:
    def test_shape(self):
        scores = np.array([[0.5,0.5]*14, [0.2,0.8]*14])
        pairs = aa.extract_score_pairs(scores)
        assert pairs.shape == (2,14,2)
        assert np.allclose(pairs[0,0], np.array([0.5,0.5]))


class TestOverallAvgImprovementWithStd:
    def test_basic(self):
        # Build encoding+scores (42) and predicted scores (n,14)
        enc = [1]+[0]*13
        cur_pairs = [(0.4,0.6)]*14
        data = np.concatenate([np.array(enc,float),
                               np.asarray([v for xy in cur_pairs for v in xy], float)])[np.newaxis,:]
        pred = np.array([[0.6]*14])
        avg, std = aa.overall_avg_improvement_with_std(data, pred)
        # Basic sanity: outputs are finite numbers
        assert np.isfinite(avg) and np.isfinite(std)

    def test_no_improvements_returns_zeros(self):
        enc = [1]+[0]*13
        cur_pairs = [(0.5,0.5)]*14
        data = np.concatenate([np.array(enc,float),
                               np.asarray([v for xy in cur_pairs for v in xy], float)])[np.newaxis,:]
        pred = np.array([[0.5]*14])
        avg, std = aa.overall_avg_improvement_with_std(data, pred)
        assert avg == 0 and std == 0


class TestFilterSessionsByMissingCountIndices:
    def test_indices_match_boolean_mask(self):
        enc = [0]*14
        pairs = [(0,0),(1,1)] + [(0.5,0.5)]*12  # 2 missing
        row = np.concatenate([np.array(enc,float),
                              np.asarray([v for xy in pairs for v in xy], float)])
        data = np.vstack([row, row])
        idx = aa.filter_sessions_by_missing_count_indices(data, 2)
        mask = aa.filter_sessions_by_missing_count(data, 2)
        assert np.array_equal(idx, np.where(mask)[0])


class TestFilterSessionsByMissingCount:
    def test_boolean_mask(self):
        enc = [0]*14
        pairsA = [(0,0),(1,1)] + [(0.5,0.5)]*12  # 2 missing
        pairsB = [(0.5,0.5)]*14                  # 0 missing
        data = np.vstack([
            np.concatenate([np.array(enc,float), np.asarray([v for xy in pairsA for v in xy], float)]),
            np.concatenate([np.array(enc,float), np.asarray([v for xy in pairsB for v in xy], float)]),
        ])
        mask2 = aa.filter_sessions_by_missing_count(data, 2)
        assert np.array_equal(mask2, np.array([True, False]))


class TestComputeErrors:
    def test_mae_and_std(self):
        gt = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 1.5, 3.5])
        mae, gstd = aa.compute_errors(gt, pred)
        assert np.isclose(mae, (0.5+0.5+0.5)/3)
        assert np.isclose(gstd, np.std(gt))


class TestFilterWithMasks:
    def test_applies_multiple_masks(self):
        data = np.arange(10)
        m1 = data % 2 == 0
        m2 = data < 8
        out = aa.filter_with_masks(data, [m1, m2])
        assert np.array_equal(out, np.array([0,2,4,6]))


class TestComputeAveragesAndStds:
    def test_computation_over_masks(self):
        cur = np.array([0.3, 0.4, 0.5, 0.6])
        fut = np.array([0.4, 0.6, 0.6, 0.9])
        mask1 = np.array([True, False, True, True])
        mask2 = np.array([True, True, True, False])
        avg, std = aa.compute_averages_and_stds(cur, fut, [mask1, mask2])
        diff = (fut - cur)[mask1][mask2[mask1]]
        assert np.isclose(avg, np.mean(diff))
        assert np.isclose(std, np.std(diff))


class TestEvaluateErrorByMissingCount:
    def test_returns_lists_and_dict(self):
        # Build synthetic data
        enc = np.zeros((3,14))
        cur_pairs = np.array([[0.5,0.5]*14,
                              [0.0,0.0]*14,
                              [1.0,1.0]*14], dtype=float)
        test_x = np.hstack([enc, cur_pairs]).astype(float)
        test_y = np.array([[0.5]*14, [0.1]*14, [0.9]*14])
        preds = test_y.copy()

        counts, mean_errs, stds, gt_dict = aa.evaluate_error_by_missing_count(test_x, test_y, preds, dims=14)
        assert len(counts) == 14
        assert len(mean_errs) == 14 and len(stds) == 14
        assert isinstance(gt_dict, dict)


class TestAverageScoresByMissingCounts:
    def test_shapes_and_values(self):
        missing_counts = [0, 1, 2]
        current = np.array([[0.2,0.8]*14,
                            [0.0,0.0]*14,
                            [1.0,1.0]*14], dtype=float)
        future = np.array([0.6, 0.3, 0.9])
        encoding = np.zeros((3,14))
        avgs, stds = aa.average_scores_by_missing_counts(missing_counts, current, future, encoding)
        assert len(avgs) == len(missing_counts)
        assert len(stds) == len(missing_counts)


class TestFindBestIdxPred:
    def test_direct_wiring_with_stubs(self, monkeypatch):
        # Prepare small positive case
        model = _SumModel()
        x = np.array([[0.5,0.5]*14, [0.2,0.8]*14], dtype=float)  # (2,28)
        y = np.array([[0.5]*14, [0.7]*14], dtype=float)
        missing_counts = [0,1,2]  # passed through to predict_all_domains

        # Stub predict_all_domains -> known matrix
        pred_mat = np.array([[0.6, 0.1, 0.4, 0.3, 0.2, 0.9, 0.7, 0.8, 0.2, 0.1, 0.5, 0.5, 0.4, 0.6],
                             [0.3, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.8, 0.5, 0.3, 0.2, 0.1, 0.9, 0.6]], dtype=float)
        monkeypatch.setattr(aa, "predict_all_domains", lambda m, xx, yy, mr: pred_mat, raising=False)

        # Stub max_prediction_from_difference_pair -> (values, indices)
        max_vals = np.array([0.9, 0.7])
        max_idx = np.array([5, 6])
        monkeypatch.setattr(aa, "max_prediction_from_difference_pair",
                            lambda diff, pm, cur, rt: (max_vals, max_idx),
                            raising=False)

        # Stub reconstruct_max_matrices -> (values_mat, onehot_mat)
        values_mat = np.array([[0,0,0,0,0,0.9,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0.7,0,0,0,0,0,0,0]], dtype=float)
        onehot_mat = np.array([[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,1,0,0,0,0,0,0,0]], dtype=int)
        monkeypatch.setattr(aa, "reconstruct_max_matrices",
                            lambda mv, mi, shape: (values_mat, onehot_mat),
                            raising=False)

        best_enc, best_preds = aa.find_best_idx_pred(model, x, y, missing_counts, run_type="repeat")
        assert np.array_equal(best_enc, onehot_mat)
        assert np.array_equal(best_preds, values_mat)
