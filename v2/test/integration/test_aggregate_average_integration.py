# tests/test_integration.py
import numpy as np
import pytest
import torch
import torch.nn as nn

import aggregate_average as aa
import shared


class SumModel(nn.Module):
    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.sum(dim=1)


# --------------------- Integration: predict_all_domains ---------------------

@pytest.mark.integration
def test_predict_all_domains_e2e():
    model = SumModel().eval()
    x = np.array([[0.5, 0.5]*14, [0.2, 0.8]*14], dtype=float)
    y = np.zeros((2, 14), dtype=float)

    mat = aa.predict_all_domains(model, x, y, loop_range=range(14))
    assert mat.shape == (2, 14)

    for d in (0, 7, 13):
        onehot = shared.create_single_encoding(rows=x.shape[0], cols=14, domain=d)
        x_d = shared.add_encoding(x, onehot)
        expected = shared.inference(model, x_d)
        assert np.allclose(mat[:, d], expected)


# --------------------- Integration: find_random_predictions ---------------------

def _row_from_pairs(pairs14):
    return np.asarray([v for xy in pairs14 for v in xy], dtype=float)


@pytest.mark.integration
def test_find_random_predictions_repeat_real():
    pairsA = [(0.5, 0.5)] * 14
    pairsA[3] = (0.0, 0.0)        # invalid
    pairsB = [(0.4, 0.6)] * 14    # all valid
    data = np.vstack([_row_from_pairs(pairsA),
                      _row_from_pairs(pairsB)])

    model = SumModel().eval()

    rand_enc, preds = aa.find_random_predictions(model, data, run_type="repeat")
    assert rand_enc.shape == (data.shape[0], 14)
    assert preds.shape == (data.shape[0],)
    row_sums = rand_enc.sum(axis=1)
    assert np.all(np.isin(row_sums, [0, 1]))

    x_random = shared.add_encoding(data, rand_enc)
    expected = shared.inference(model, x_random)
    assert np.allclose(preds, expected)


@pytest.mark.integration
def test_find_random_predictions_nonrepeat_real():
    pairsA = [(0.5, 0.5)] * 14
    pairsA[2] = (1.0, 1.0)        # invalid
    pairsA[9] = (0.0, 0.0)        # invalid
    pairsB = [(0.5, 0.5)] * 14    # all valid
    data = np.vstack([_row_from_pairs(pairsA),
                      _row_from_pairs(pairsB)])

    model = SumModel().eval()

    rand_enc, preds = aa.find_random_predictions(model, data, run_type="nonrepeat")
    assert rand_enc.shape == (data.shape[0], 14)
    assert preds.shape == (data.shape[0],)
    sums = rand_enc.sum(axis=1)
    assert sums[0] == 1 and sums[1] == 0

    x_random = shared.add_encoding(data, rand_enc)
    expected = shared.inference(model, x_random)
    assert np.allclose(preds, expected)


# --------------------- Integration: higher-level metrics ---------------------

@pytest.mark.integration
def test_overall_avg_improvement_with_std_shapes_and_values():
    enc = [1] + [0]*13
    cur_pairs = [(0.4, 0.6)]*14
    data = np.concatenate([np.array(enc, float),
                           np.asarray([v for xy in cur_pairs for v in xy], float)])[np.newaxis, :]
    preds = np.array([[0.6]*14], dtype=float)  # simple predicted scores

    avg, std = aa.overall_avg_improvement_with_std(data, preds)
    assert np.isfinite(avg) and np.isfinite(std)


@pytest.mark.integration
def test_filter_sessions_by_missing_count_roundtrip():
    enc = [0]*14
    pairsA = [(0,0),(1,1)] + [(0.5,0.5)]*12  # 2 missing
    pairsB = [(0.5,0.5)]*14                  # 0 missing
    data = np.vstack([
        np.concatenate([np.array(enc,float), np.asarray([v for xy in pairsA for v in xy], float)]),
        np.concatenate([np.array(enc,float), np.asarray([v for xy in pairsB for v in xy], float)]),
    ])
    mask2 = aa.filter_sessions_by_missing_count(data, 2)
    idx = aa.filter_sessions_by_missing_count_indices(data, 2)
    assert np.array_equal(idx, np.where(mask2)[0])


@pytest.mark.integration
def test_evaluate_error_by_missing_count_runs():
    enc = np.zeros((3,14))
    cur_pairs = np.array([[0.5,0.5]*14,
                          [0.0,0.0]*14,
                          [1.0,1.0]*14], dtype=float)
    test_x = np.hstack([enc, cur_pairs]).astype(float)
    test_y = np.array([[0.5]*14, [0.1]*14, [0.9]*14])
    preds = test_y.copy()

    counts, means, stds, gt_dict = aa.evaluate_error_by_missing_count(test_x, test_y, preds, dims=14)
    assert len(counts) == 14 and len(means) == 14 and len(stds) == 14
    assert isinstance(gt_dict, dict)


@pytest.mark.integration
def test_average_scores_by_missing_counts_executes():
    missing_counts = [0, 1, 2]
    current = np.array([[0.2,0.8]*14,
                        [0.0,0.0]*14,
                        [1.0,1.0]*14], dtype=float)
    future = np.array([0.6, 0.3, 0.9])
    encoding = np.zeros((3,14))
    avgs, stds = aa.average_scores_by_missing_counts(missing_counts, current, future, encoding)
    assert len(avgs) == len(missing_counts)
    assert len(stds) == len(missing_counts)
