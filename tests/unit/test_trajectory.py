"""Unit tests for module: trajectory (/mnt/data/unzipped_modules/experiments/experiments/trajectory.py)\nAuto-generated scaffolding: class-per-function with happy/edge cases.\nFill in details per docstrings and usage.\n"""
from __future__ import annotations
import pytest
import numpy as np
import torch
from ct.experiments import trajectory as mod
from ct.utils.reproducibility import set_global_seed

class Test_max_prediction_from_difference:
    """
    Doc: Find the maximum prediction improvement for each row based on the difference
    between current and predicted scores. Parameters: difference_matrix (np.ndarray):
    A 2D array where each element represents the difference between predicted and current scores.
    """

    def test_happy_basic(self):
        # current has NaNs indicating unpracticed domains
        current = np.array([
            [np.nan, 0.5,   0.2],
            [0.1,    np.nan, np.nan],
        ], dtype=float)

        # predicted scores
        pred = np.array([
            [0.8, 0.6, 0.4],
            [0.2, 0.9, 0.7],
        ], dtype=float)

        # difference between predicted and current (can be NaN where current is NaN)
        diff = pred - current

        max_vals, max_idxs = mod.max_prediction_from_difference(diff, pred, current)

        # Row 0: only column 0 is NaN in current -> choose pred[0,0] = 0.8
        # Row 1: columns 1 and 2 are NaN in current -> choose max(pred[1,1], pred[1,2]) = 0.9 at idx 1
        assert np.allclose(max_vals, np.array([0.8, 0.9]))
        assert np.array_equal(max_idxs, np.array([0, 1]))

    def test_edge_minimal_or_empty(self):
        # Minimal 1x1 case where the single entry is unpracticed (NaN)
        current = np.array([[np.nan]])
        pred = np.array([[0.42]])
        diff = pred - current  # will be NaN

        max_vals, max_idxs = mod.max_prediction_from_difference(diff, pred, current)

        assert max_vals.shape == (1,)
        assert max_idxs.shape == (1,)
        assert np.allclose(max_vals, np.array([0.42]))
        assert np.array_equal(max_idxs, np.array([0]))

    def test_error_invalid_input(self):
        # Mismatched shapes should raise an error
        current = np.array([[np.nan, 0.1]])
        pred = np.array([[0.2, 0.3]])
        diff_bad_shape = np.array([[0.1, 0.2],
                                   [0.3, 0.4]])  # 2x2, not 1x2

        with pytest.raises(Exception):
            mod.max_prediction_from_difference(diff_bad_shape, pred, current)

class Test_min_prediction_from_difference:
    """
    Doc: Find the minimum prediction improvement for each row based on the difference
    between current and predicted scores. Parameters: difference_matrix (np.ndarray):
    A 2D array where each element represents the difference between predicted and current scores.
    """

    def test_happy_basic(self):
        # current has NaNs indicating unpracticed domains
        current = np.array([
            [np.nan, 0.4, 0.6],
            [0.2, np.nan, np.nan],
        ], dtype=float)

        # predicted scores
        pred = np.array([
            [0.5, 0.8, 0.7],
            [0.9, 0.3, 0.5],
        ], dtype=float)

        # difference between predicted and current (dummy, shape consistency)
        diff = pred - current

        min_vals, min_idxs = mod.min_prediction_from_difference(diff, pred, current)

        # Row 0: unpracticed only at col 0 → pick 0.5, idx 0
        # Row 1: unpracticed at cols 1 & 2 → pick min(0.3, 0.5) = 0.3, idx 1
        assert np.allclose(min_vals, np.array([0.5, 0.3]))
        assert np.array_equal(min_idxs, np.array([0, 1]))

    def test_edge_minimal_or_empty(self):
        # Minimal 1×1 where entry is NaN → take that prediction
        current = np.array([[np.nan]])
        pred = np.array([[0.12]])
        diff = pred - current

        min_vals, min_idxs = mod.min_prediction_from_difference(diff, pred, current)

        assert min_vals.shape == (1,)
        assert min_idxs.shape == (1,)
        assert np.allclose(min_vals, np.array([0.12]))
        assert np.array_equal(min_idxs, np.array([0]))

    def test_error_invalid_input(self):
        # Mismatched shapes should raise
        current = np.array([[np.nan, 0.1]])
        pred = np.array([[0.2, 0.3]])
        diff_bad = np.array([[0.1], [0.2]])  # shape mismatch

        with pytest.raises(Exception):
            mod.min_prediction_from_difference(diff_bad, pred, current)



class Test_random_prediction_from_difference:
    """Auto-generated tests for `random_prediction_from_difference`.
    Doc: Find a random prediction improvement for each row based on the difference
    between current and predicted scores. Parameters: difference_matrix (np.ndarray):
    A 2D array where each element represents the difference between predicted and current scores.
    """

    def test_happy_basic(self):
        # Ensure reproducibility
        set_global_seed(123)

        # current has NaNs indicating unpracticed domains
        current = np.array([
            [np.nan, 0.4, 0.6],
            [0.2, np.nan, np.nan],
        ], dtype=float)

        # predicted scores
        pred = np.array([
            [0.5, 0.8, 0.7],
            [0.9, 0.3, 0.5],
        ], dtype=float)

        # dummy difference matrix
        diff = pred - current

        random_vals, random_idxs = mod.random_prediction_from_difference(diff, pred, current)

        # Both outputs should have one value per row
        assert random_vals.shape == (2,)
        assert random_idxs.shape == (2,)

        # Results should be reproducible with the same seed
        set_global_seed(123)
        r2_vals, r2_idxs = mod.random_prediction_from_difference(diff, pred, current)
        assert np.allclose(random_vals, r2_vals)
        assert np.array_equal(random_idxs, r2_idxs)

        # For each row, the selected value should correspond to a NaN position in current
        for row, (idx, val) in enumerate(zip(random_idxs, random_vals)):
            print("row, idx, val:", row, idx, val)
            assert np.isnan(current[row, idx])
            assert np.isclose(val, pred[row, idx])

    def test_edge_minimal_or_empty(self):
        set_global_seed(0)
        # Minimal case: single NaN entry
        current = np.array([[np.nan]])
        pred = np.array([[0.42]])
        diff = pred - current

        vals, idxs = mod.random_prediction_from_difference(diff, pred, current)

        assert vals.shape == (1,)
        assert idxs.shape == (1,)
        assert np.allclose(vals, np.array([0.42]))
        assert np.array_equal(idxs, np.array([0]))

    def test_error_invalid_input(self):
        # Mismatched shapes should raise
        current = np.array([[np.nan, 0.1]])
        pred = np.array([[0.2, 0.3]])
        diff_bad = np.array([[0.1], [0.2]])  # invalid shape

        with pytest.raises(Exception):
            mod.random_prediction_from_difference(diff_bad, pred, current)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Not used directly; we pass a custom inference_fn in tests
        return x

def _dummy_create_single_encoding(rows, cols, i):
    # simple one-hot encoding appended later; shape match not critical for test
    return np.zeros((rows, cols), dtype=float)

def _dummy_create_missing_indicator(initial_scores):
    # replace NaNs with 0 for simplicity
    return np.nan_to_num(initial_scores, nan=0.0).astype(float)

def _dummy_add_encoding(x, enc):
    # no-op: return x unchanged for testing
    return x

def _dummy_inference_factory(pred_matrix):
    """
    Returns an inference_fn that ignores inputs and returns pred_matrix as a torch tensor
    (broadcasted across batch rows).
    """
    def _inference_fn(model, x_tensor):
        rows = x_tensor.shape[0]
        # pred_matrix is shape (num_domains,) or (rows, num_domains). Make it (rows, num_domains).
        pm = pred_matrix
        if pm.ndim == 1:
            pm = np.tile(pm, (rows, 1))
        return torch.tensor(pm, dtype=torch.float32)
    return _inference_fn

class TestFind_next_domain:
    """
    Doc: Take in model and mode ("best", "random", or "worst"), return index and value of next domain to practice.
    Parameters:
        initial_scores (np.ndarray): Current scores with NaNs for unpracticed domains.
        model (torch.nn.Module): The predictive model.
        mode (str): Selection mode - "best", "random", or "worst".
    """


    def test_best_mode_calls_max_and_returns_selection(self):
        initial = np.array([[np.nan, 0.2, 0.3]], dtype=float)

        # Craft a prediction matrix (for all domains) via dummy inference
        # Say there are 3 domains; pick max at index 2
        pred_vec = np.array([0.4, 0.5, 0.9], dtype=float)

        # Stub selectors to check correct one is called and return a known answer
        def fake_max_fn(diff, pred, cur):
            return np.array([0.9]), np.array([2])

        # Run with injectables
        idx, val = mod.find_next_domain(
            initial_scores=initial,
            model=DummyModel(),
            mode="best",
            create_single_encoding_fn=_dummy_create_single_encoding,
            create_missing_indicator_fn=_dummy_create_missing_indicator,
            add_encoding_fn=_dummy_add_encoding,
            inference_fn=_dummy_inference_factory(pred_vec),
            max_fn=fake_max_fn,
            # Supply dummies for completeness (won't be used)
            min_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("min_fn should not be called")),
            rand_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("rand_fn should not be called")),
            num_domains=3,
        )

        assert idx == 2
        assert np.isclose(val, 0.9)

    def test_worst_mode_calls_min_and_returns_selection(self):
        initial = np.array([[0.1, np.nan, np.nan]], dtype=float)
        pred_vec = np.array([0.6, 0.2, 0.4], dtype=float)

        def fake_min_fn(diff, pred, cur):
            return np.array([0.2]), np.array([1])

        idx, val = mod.find_next_domain(
            initial_scores=initial,
            model=DummyModel(),
            mode="worst",
            create_single_encoding_fn=_dummy_create_single_encoding,
            create_missing_indicator_fn=_dummy_create_missing_indicator,
            add_encoding_fn=_dummy_add_encoding,
            inference_fn=_dummy_inference_factory(pred_vec),
            min_fn=fake_min_fn,
            max_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("max_fn should not be called")),
            rand_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("rand_fn should not be called")),
            num_domains=3,
        )

        assert idx == 1
        assert np.isclose(val, 0.2)

    def test_middle_alias_random_calls_rand(self):
        initial = np.array([[np.nan, np.nan, 0.0]], dtype=float)
        pred_vec = np.array([0.3, 0.8, 0.1], dtype=float)

        def fake_rand_fn(diff, pred, cur):
            # Pretend RNG picked column 0
            return np.array([0.3]), np.array([0])

        for mode in ("middle", "random"):
            idx, val = mod.find_next_domain(
                initial_scores=initial,
                model=DummyModel(),
                mode=mode,
                create_single_encoding_fn=_dummy_create_single_encoding,
                create_missing_indicator_fn=_dummy_create_missing_indicator,
                add_encoding_fn=_dummy_add_encoding,
                inference_fn=_dummy_inference_factory(pred_vec),
                rand_fn=fake_rand_fn,
                max_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("max_fn should not be called")),
                min_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("min_fn should not be called")),
                num_domains=3,
            )
            assert idx == 0
            assert np.isclose(val, 0.3)

    def test_raises_on_unknown_mode(self):
        initial = np.array([[np.nan, 0.2, 0.3]], dtype=float)

        with pytest.raises(ValueError):
            mod.find_next_domain(
                initial_scores=initial,
                model=DummyModel(),
                mode="unknown",
                create_single_encoding_fn=_dummy_create_single_encoding,
                create_missing_indicator_fn=_dummy_create_missing_indicator,
                add_encoding_fn=_dummy_add_encoding,
                inference_fn=_dummy_inference_factory(np.array([0.1, 0.2, 0.3])),
                num_domains=3,
            )

    def test_builds_prediction_matrix_for_multiple_rows(self):
        # Ensure it works with multiple rows; selector gets called with proper shapes.
        initial = np.array([
            [np.nan, 0.1, 0.2],
            [0.3,   np.nan, 0.4],
        ], dtype=float)

        pred_vec = np.array([0.5, 0.6, 0.7], dtype=float)

        # Selector: return a deterministic pick to verify output passthrough
        def fake_max_fn(diff, pred, cur):
            # pick column 1 with value 0.6
            return np.array([0.6]), np.array([1])

        idx, val = mod.find_next_domain(
            initial_scores=initial,
            model=DummyModel(),
            mode="best",
            create_single_encoding_fn=_dummy_create_single_encoding,
            create_missing_indicator_fn=_dummy_create_missing_indicator,
            add_encoding_fn=_dummy_add_encoding,
            inference_fn=_dummy_inference_factory(pred_vec),
            max_fn=fake_max_fn,
            num_domains=3,
        )

        assert idx == 1
        assert np.isclose(val, 0.6)


class Test_trajectory:
    def test_deterministic_unique_domains_and_performance(self):
        """
        Use 3 domains and 3 steps; select each domain exactly once with known scores.
        Then we can compute performance precisely: mean of known scores at each step.
        """
        num_domains = 3
        num_steps = 3

        # Predefine selection sequence: domains [2, 0, 1] with scores [0.9, 0.3, 0.6]
        plan = [(2, 0.9), (0, 0.3), (1, 0.6)]
        calls = {"i": 0}

        def fake_find_next_domain(current_scores, model, mode):
            i = calls["i"]
            d, s = plan[i]
            calls["i"] += 1
            return d, s

        perf, cur, order = mod.trajectory(
            DummyModel(),
            mode="best",  # mode is irrelevant here; we control the selection
            find_next_domain_fn=fake_find_next_domain,
            num_domains=num_domains,
            num_steps=num_steps,
        )

        # Order should be 1-based indices from our plan: [3, 1, 2]
        assert order == [3, 1, 2]

        # Current scores should have those three scores in their chosen slots
        expected = np.array([[0.3, 0.6, 0.9]])  # after all steps (final state)
        assert np.allclose(cur, expected, equal_nan=False)

        # Performance is mean over known scores at each step:
        # step 1: mean([0.9])           = 0.9
        # step 2: mean([0.9, 0.3])      = 0.6
        # step 3: mean([0.9, 0.3, 0.6]) = 0.6
        assert np.allclose(perf, [0.9, 0.6, 0.6])

    def test_handles_repeated_domain_overwrite(self):
        """
        If the helper returns the same domain more than once, trajectory overwrites that cell.
        Performance should reflect the updated mean.
        """
        num_domains = 3
        num_steps = 3
        # Always pick domain 1 (0-based) with increasing scores
        plan = [(1, 0.2), (1, 0.5), (1, 0.8)]
        calls = {"i": 0}
        def fake_find_next_domain(current_scores, model, mode):
            d, s = plan[calls["i"]]
            calls["i"] += 1
            return d, s

        perf, cur, order = mod.trajectory(
            DummyModel(),
            mode="worst",
            find_next_domain_fn=fake_find_next_domain,
            num_domains=num_domains,
            num_steps=num_steps,
        )

        # Order is 1-based domain indices (always 2 here)
        assert order == [2, 2, 2]

        # Only the picked domain is filled at the end
        expected = np.array([[np.nan, 0.8, np.nan]])
        assert np.allclose(cur, expected, equal_nan=True)

        # Performance after each step:
        # step1: mean([0.2]) = 0.2
        # step2: mean([0.5]) = 0.5 (overwrite)
        # step3: mean([0.8]) = 0.8 (overwrite)
        assert np.allclose(perf, [0.2, 0.5, 0.8])

    def test_raises_on_bad_mode_even_with_patch(self):
        with pytest.raises(ValueError):
            mod.trajectory(
                DummyModel(),
                mode="nope",
                find_next_domain_fn=lambda *a, **k: (0, 0.0),
                num_domains=3,
                num_steps=1,
            )