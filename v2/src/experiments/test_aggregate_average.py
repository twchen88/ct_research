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

def _make_row(encoding14, pairs14):
    """
    Build a single 42-length row: 14 encodings + 14 (x,y) pairs.
    """
    enc = np.asarray(encoding14, dtype=float)
    flat_pairs = np.asarray([v for xy in pairs14 for v in xy], dtype=float)
    assert enc.shape == (14,)
    assert flat_pairs.shape == (28,)
    return np.concatenate([enc, flat_pairs])


class TestAssignRepeat:
    def test_vacuous_true_when_all_encodings_zero(self):
        enc = [0] * 14
        pairs = [(0.0, 0.0)] * 14  # invalid pairs but irrelevant since all enc=0
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert res.shape == (1,)
        assert bool(res[0])

    def test_single_domain_valid_pair_true(self):
        enc = [0] * 14
        enc[3] = 1
        pairs = [(0.0, 0.0)] * 14
        pairs[3] = (0.25, 0.75)  # valid: x + y == 1 and not [0,0]/[1,1]
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert bool(res[0])

    @pytest.mark.parametrize("bad_pair", [(0.0, 0.0), (1.0, 1.0)])
    def test_single_domain_invalid_pairs_false(self, bad_pair):
        enc = [0] * 14
        enc[5] = 1
        pairs = [(0.25, 0.75)] * 14
        pairs[5] = bad_pair  # explicitly invalid
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert not bool(res[0])

    def test_multiple_domains_all_valid_true(self):
        enc = [0] * 14
        for i in (0, 7, 13):
            enc[i] = 1
        pairs = [(0.4, 0.6)] * 14  # valid everywhere
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert bool(res[0])

    def test_unencoded_domains_can_be_invalid_and_still_true(self):
        enc = [0] * 14
        enc[8] = 1
        pairs = [(0.0, 0.0)] * 14          # many invalid pairs
        pairs[8] = (0.6, 0.4)              # required domain is valid
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert bool(res[0])

    def test_batch_multiple_rows(self):
        # Row A: True
        encA = [0] * 14; encA[0] = 1
        pairsA = [(0.3, 0.7)] * 14

        # Row B: False (required domain 5 invalid)
        encB = [0] * 14; encB[1] = 1; encB[5] = 1
        pairsB = [(0.25, 0.75)] * 14
        pairsB[5] = (1.0, 1.0)

        # Row C: True (all required and valid)
        encC = [1] * 14
        pairsC = [(0.5, 0.5)] * 14

        rows = np.vstack([
            _make_row(encA, pairsA),
            _make_row(encB, pairsB),
            _make_row(encC, pairsC),
        ])

        res = aa.assign_repeat(rows)
        expected = np.array([True, False, True])
        assert res.shape == (3,)
        assert np.array_equal(res, expected)

    def test_output_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        encs = rng.integers(0, 2, size=(5, 14))
        rows = np.vstack([_make_row(enc, [(0.4, 0.6)] * 14) for enc in encs])

        res = aa.assign_repeat(rows)
        assert res.dtype == bool
        assert res.shape == (5,)

    def test_tolerance_within_eps_counts_as_valid(self):
        # Assumes implementation uses isclose(..., atol=1e-8, rtol=0.0)
        eps_edge = 1e-10
        enc = [0] * 14
        enc[3] = 1
        # Pair sums to 1 within tiny eps; not exactly [0,0]/[1,1]
        pairs = [(0.0, 0.0)] * 14
        pairs[3] = (0.4 + eps_edge, 0.6 - eps_edge)
        row = _make_row(enc, pairs)

        res = aa.assign_repeat(row[np.newaxis, :])
        assert bool(res[0])

class TestCreateRandomEncoding:
    # --- Class-local helpers/fixtures ----------------------------------------

    @staticmethod
    def _make_row(pairs14):
        """
        Build one row of shape (28,) from 14 (x,y) pairs.
        This helper is local to this class to avoid cross-class name collisions.
        """
        flat = np.asarray([v for xy in pairs14 for v in xy], dtype=float)
        assert flat.shape == (28,), "pairs14 must expand to exactly 28 values"
        return flat

    @pytest.fixture(autouse=True)
    def _deterministic_rng(self, monkeypatch):
        """
        Make randomness deterministic for this class only.
        """
        def _fixed_default_rng():
            return np.random.default_rng(12345)
        monkeypatch.setattr(aa.np.random, "default_rng", _fixed_default_rng)

    # --- Positive-path tests --------------------------------------------------

    def test_repeat_mode_selects_from_valid_pairs(self):
        pairs = [(0.4, 0.6)] * 14
        pairs[5] = (0.0, 0.0)  # invalid/missing
        row = self._make_row(pairs)[np.newaxis, :]

        res = aa.create_random_encoding(row, run_type="repeat")

        assert res.shape == (1, 14)
        assert res.sum() == 1 # there is always one chosen
        # Must avoid the invalid index in repeat mode
        assert res[0, 5] == 0
        chosen = np.argmax(res[0])
        assert chosen != 5

    def test_nonrepeat_mode_selects_from_invalid_pairs(self):
        pairs = [(0.4, 0.6)] * 14
        pairs[7] = (1.0, 1.0)  # invalid/missing
        row = self._make_row(pairs)[np.newaxis, :]

        res = aa.create_random_encoding(row, run_type="nonrepeat")

        assert res.shape == (1, 14)
        assert res.sum() == 1
        # Must pick the invalid index in nonrepeat mode
        assert res[0, 7] == 1

    def test_repeat_mode_all_invalid_means_all_zeros(self):
        pairs = [(0.0, 0.0)] * 14  # all missing
        row = self._make_row(pairs)[np.newaxis, :]

        res = aa.create_random_encoding(row, run_type="repeat")

        assert np.array_equal(res, np.zeros((1, 14), dtype=res.dtype))

    def test_nonrepeat_mode_all_valid_means_all_zeros(self):
        pairs = [(0.5, 0.5)] * 14  # all valid/complementary
        row = self._make_row(pairs)[np.newaxis, :]

        res = aa.create_random_encoding(row, run_type="nonrepeat")

        assert np.array_equal(res, np.zeros((1, 14), dtype=res.dtype))

    def test_multiple_rows_mixed(self):
        # Row A: invalid at 2
        pairsA = [(0.4, 0.6)] * 14
        pairsA[2] = (0.0, 0.0)

        # Row B: all valid
        pairsB = [(0.5, 0.5)] * 14

        # Row C: invalid at 1 and 9
        pairsC = [(0.4, 0.6)] * 14
        pairsC[1] = (1.0, 1.0)
        pairsC[9] = (0.0, 0.0)

        data = np.vstack([
            self._make_row(pairsA),
            self._make_row(pairsB),
            self._make_row(pairsC),
        ])

        res_repeat = aa.create_random_encoding(data, run_type="repeat")
        res_nonrepeat = aa.create_random_encoding(data, run_type="nonrepeat")

        # Row A: repeat must avoid 2, nonrepeat must pick 2
        assert res_repeat[0, 2] == 0
        assert res_nonrepeat[0, 2] == 1

        # Row B: repeat chooses some valid, nonrepeat all zeros
        assert res_nonrepeat[1].sum() == 0
        assert res_repeat[1].sum() == 1

        # Row C: repeat must avoid {1,9}; nonrepeat must choose one of {1,9}
        chosen_repeat = np.argmax(res_repeat[2]) if res_repeat[2].any() else -1
        chosen_nonrepeat = np.argmax(res_nonrepeat[2]) if res_nonrepeat[2].any() else -1
        assert chosen_repeat not in (1, 9)
        assert chosen_nonrepeat in (1, 9)

    def test_dtype_and_shape(self):
        pairs = [(0.0, 0.0)] * 14
        row = self._make_row(pairs)
        data = np.vstack([row, row])

        res = aa.create_random_encoding(data, run_type="nonrepeat")

        assert res.shape == (2, 14)
        assert res.dtype in (np.int_, np.int64, np.bool_)
        assert np.all(res.sum(axis=1) == 1)

    # --- Negative-path / validation tests ------------------------------------

    def test_raises_on_wrong_num_columns(self):
        # Too few columns
        bad = np.zeros((3, 27))
        with pytest.raises((ValueError, AssertionError)):
            aa.create_random_encoding(bad, run_type="repeat")

        # Too many columns
        bad = np.zeros((3, 29))
        with pytest.raises((ValueError, AssertionError)):
            aa.create_random_encoding(bad, run_type="nonrepeat")

    def test_invalid_run_type_gracefully_handled(self):
        # Depending on your choice: raise or treat as no-op.
        pairs = [(0.4, 0.6)] * 14
        row = self._make_row(pairs)[np.newaxis, :]

        # If your implementation RAISES on bad run_type:
        with pytest.raises((ValueError, AssertionError)):
            aa.create_random_encoding(row, run_type="unknown")


