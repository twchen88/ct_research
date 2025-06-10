import numpy as np
import src.data.encoding as encoding


def test_create_missing_indicator():
    # Test with a simple case
    data = np.array([[0.3, 0.5, np.nan], [np.nan, 0.1, 0.2]])
    encoded = encoding.create_missing_indicator(data, rand_seed=42)
    
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


def test_encode_target_data():
    # Test with a simple case
    target = np.array([[0.5, 0.2, 0.8], [0.1, 0.4, 0.6]])
    encoded_domains = np.array([[1, 0, 0,], [0, 1, 0]])
    
    encoded_target = encoding.encode_target_data(target, encoded_domains)
    
    expected_target = np.array([[0.5, 0, 0], [0., 0.4, 0.]])
    
    assert np.array_equal(encoded_target, expected_target)