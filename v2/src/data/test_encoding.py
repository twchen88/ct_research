import numpy as np
from src.data.encoding import create_missing_indicator


def test_create_missing_indicator():
    # Test with a simple case
    data = np.array([[0.3, 0.5, np.nan], [np.nan, 0.1, 0.2]])
    encoded = create_missing_indicator(data, rand_seed=42)
    
    expected_shape = (2, 6)  # Each value becomes two columns
    assert encoded.shape == expected_shape
    
    # Check NaN handling, there should be no NaN values in the encoded output
    assert np.isnan(encoded).any() == False

    # Check non-NaN values are encoded correctly
    assert encoded[0, 0] == 0.3 and encoded[0, 1] == 0.7
    assert encoded[0, 2] == 0.5 and encoded[0, 3] == 0.5
    assert encoded[1, 2] == 0.1 and encoded[1, 3] == 0.9
    assert encoded[1, 4] == 0.2 and encoded[1, 5] == 0.8

    # Check NaN values are encoded as [0, 0] or [1, 1]
    assert encoded[0, 4] == 0 or encoded[0, 4] == 1
    assert encoded[0, 5] == 0 or encoded[0, 5] == 1
    assert encoded[0, 4] == encoded[0, 5]

    assert encoded[1, 0] == 0 or encoded[1, 0] == 1
    assert encoded[1, 1] == 0 or encoded[1, 1] == 1
    assert encoded[1, 0] == encoded[1, 1]