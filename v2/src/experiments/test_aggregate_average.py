import pytest
import numpy as np
import src.experiments.aggregate_average as aa


class TestFilterRowsBySum:
    def test_basic_case(self):
        data = np.array([
            [1, 2, 3],   # sum(0:2)=3 ≤ 5 → keep
            [5, 5, 5],   # sum(0:2)=10 > 5 → drop
            [0, 0, 0],   # sum(0:2)=0 ≤ 5 → keep
        ])
        filtered, mask = aa.filter_rows_by_sum(data, slice(0, 2), 5)

        expected_mask = np.array([True, False, True])
        assert np.array_equal(mask, expected_mask)
        assert np.array_equal(filtered, data[expected_mask])

    def test_inclusive_boundary(self):
        data = np.array([
            [2, 3],   # sum=5 → keep
            [3, 3],   # sum=6 → drop
        ])
        filtered, mask = aa.filter_rows_by_sum(data, slice(0, 2), 5)
        assert np.array_equal(mask, np.array([True, False]))
        assert np.array_equal(filtered, data[[0]])

    def test_negative_values(self):
        data = np.array([
            [10, -20],  # sum=-10 ≤ -5 → keep
            [-5, -6],   # sum=-11 ≤ -5 → keep
            [0, 0],     # sum=0 > -5 → drop
        ])
        filtered, mask = aa.filter_rows_by_sum(data, slice(0, 2), -5)
        expected_mask = np.array([True, True, False])
        assert np.array_equal(mask, expected_mask)
        assert np.array_equal(filtered, data[expected_mask])

    def test_empty_slice_means_zero_sum(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        filtered, mask = aa.filter_rows_by_sum(data, slice(1, 1), 0)
        assert np.all(mask)
        assert np.array_equal(filtered, data)

    def test_mask_matches_filtered(self):
        data = np.arange(12).reshape(4, 3)
        filtered, mask = aa.filter_rows_by_sum(data, slice(0, 3), 6)
        assert np.array_equal(filtered, data[mask])

    def test_output_shapes(self):
        data = np.arange(12).reshape(3, 4)
        filtered, mask = aa.filter_rows_by_sum(data, slice(1, 3), 5)
        assert mask.shape == (3,)
        assert filtered.shape[1] == 4

    def test_input_not_modified(self):
        data = np.array([[1, 2], [3, 4]])
        original = data.copy()
        _ = aa.filter_rows_by_sum(data, slice(0, 2), 5)
        assert np.array_equal(data, original)

    @pytest.mark.parametrize(
        "col_range, threshold, expected_mask",
        [
            (slice(0, 2), 2, np.array([True, False, True])),  # sums=[3,6,0]
            (slice(1, 3), 2, np.array([True, False, True])),   # sums=[2,3,0]
            (slice(2, 3), 1, np.array([True, False, True])),   # vals=[3,2,0]
        ],
    )
    def test_parametrized(self, col_range, threshold, expected_mask):
        data = np.array([
            [1, 1, 1],
            [2, 1, 2],
            [0, 0, 0],
        ])
        filtered, mask = aa.filter_rows_by_sum(data, col_range, threshold)
        assert np.array_equal(mask, expected_mask)
        assert np.array_equal(filtered, data[expected_mask])


class TestFindMissingMask:
    def test_basic_exact_zeros_and_ones(self):
        x1 = np.array([0.0, 0.0, 1.0, 1.0, 0.3])
        x2 = np.array([0.0, 1.0, 1.0, 0.0, 0.3])
        # Missing only at [0,0] and [1,1]
        mask = aa.find_missing_mask(x1, x2)
        expected = np.array([True, False, True, False, False], dtype=bool)

        assert mask.dtype == bool
        assert mask.shape == x1.shape
        assert np.array_equal(mask, expected)

    def test_equal_but_not_zero_or_one_is_false(self):
        # Equal values like [0.5, 0.5] should NOT be marked missing
        x1 = np.array([0.5, 0.2, 0.8])
        x2 = np.array([0.5, 0.2, 0.8])
        mask = aa.find_missing_mask(x1, x2)
        assert np.array_equal(mask, np.array([False, False, False]))

    def test_tolerance_within_eps_counts_as_zero_or_one(self):
        eps = 1e-8
        # Values within eps of 0 or 1 should be treated as 0 or 1
        x1 = np.array([0.0 + 1e-10, 1.0 - 1e-10, 0.0, 1.0])
        x2 = np.array([0.0 + 1e-10, 1.0 - 1e-10, 0.0, 1.0])
        mask = aa.find_missing_mask(x1, x2, eps=eps)
        assert np.array_equal(mask, np.array([True, True, True, True]))

    def test_tolerance_outside_eps_is_not_zero_or_one(self):
        eps = 1e-8
        # Values outside eps from 0 or 1 should not be considered 0 or 1
        x1 = np.array([0.0 + 1e-7, 1.0 - 1e-7, 0.0, 1.0])
        x2 = np.array([0.0 + 1e-7, 1.0 - 1e-7, 0.0, 1.0])
        mask = aa.find_missing_mask(x1, x2, eps=eps)
        expected = np.array([False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_mixed_pairs_vectorized(self):
        # 2D arrays work elementwise
        x1 = np.array([
            [0.0, 1.0, 0.5],
            [0.0, 0.3, 1.0],
        ])
        x2 = np.array([
            [0.0, 1.0, 0.5],
            [1.0, 0.3, 1.0],
        ])
        mask = aa.find_missing_mask(x1, x2)
        expected = np.array([
            [True, True, False],   # [0,0], [1,1], [0.5,0.5]
            [False, False, True],  # [0,1] -> not missing; [0.3,0.3] equal but not 0/1; [1,1]
        ], dtype=bool)

        assert mask.shape == x1.shape
        assert np.array_equal(mask, expected)

    def test_nan_pairs_are_not_missing(self):
        # NaNs should not be treated as equal (np.isclose is False for NaNs)
        x1 = np.array([np.nan, 0.0, 1.0])
        x2 = np.array([np.nan, 0.0, 1.0])
        mask = aa.find_missing_mask(x1, x2)
        expected = np.array([False, True, True])
        assert np.array_equal(mask, expected)

    def test_dtype_and_shape(self):
        rng = np.random.default_rng(0)
        x1 = rng.random((4, 5))
        x2 = x1.copy()  # equal everywhere, but not 0/1 → all False
        mask = aa.find_missing_mask(x1, x2)
        assert mask.dtype == bool
        assert mask.shape == x1.shape
        assert not mask.any()