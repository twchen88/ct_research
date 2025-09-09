import numpy as np
import torch
import random

from src.experiments.shared import *

"""
src/experiments/aggregate_average.py
--------------------------------

"""

def filter_rows_by_sum(data: np.ndarray, col_range: slice, sum_threshold: int) -> tuple:
    """
    Filters out rows where the sum of specified range of columns falls below a given threshold.

    Returns:
    - filtered_data (np.ndarray): Rows where column sum <= threshold
    - sum_mask (np.ndarray): Boolean mask of kept rows
    """
    sum_mask = data[:, col_range].sum(axis=1) <= sum_threshold
    filtered_data = data[sum_mask]
    return filtered_data, sum_mask

def find_missing_mask(x1, x2):
    eq = x1 == x2
    is_0_or_1 = (x1 == 0) | (x1 == 1)
    return eq & is_0_or_1


def assign_repeat(data: np.ndarray) -> np.ndarray:
    """
    Determine if each row/session is a repeat based on domain encoding and score pair values.

    A session is considered a repeat if all domains with encoding == 1 have valid scores,
    where a valid score is defined as a pair of [x, 1 - x]. Invalid pairs are [0, 0] or [1, 1].

    Parameters:
        data (np.ndarray): 2D array with shape (n_samples, 42). First 14 columns are encodings.
                           Columns 14-41 are 14 pairs of scores.

    Returns:
        np.ndarray: Boolean array of shape (n_samples,) where True means session is a repeat.
    """
    encodings = data[:, :14]  # (N, 14)
    score_pairs = data[:, 14:42].reshape(-1, 14, 2)  # (N, 14, 2)

    # Determine if a score pair is invalid: [0, 0] or [1, 1]
    invalid_mask = find_missing_mask(score_pairs[:, :, 0], score_pairs[:, :, 1])  # (N, 14)

    # A session is invalid (i.e., not repeat) if any domain with encoding==1 has an invalid score pair
    violating_domains = (encodings == 1) & invalid_mask  # (N, 14)

    is_repeat = ~np.any(violating_domains, axis=1)

    return is_repeat



def create_random_encoding(data: np.ndarray, run_type: str) -> np.ndarray:
    """
    Creates a random domain encoding for each row by selecting one domain
    at random where the score pair is invalid ([0,0] or [1,1]).
    
    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=42). Assumes columns 14–41 are 14 score pairs.

    Returns:
        np.ndarray: Binary array of shape (n_rows, 14) with a single 1 randomly placed per row
                    where an invalid score pair exists. Rows with no invalids are all zeros.
    """
    n_rows = data.shape[0]
    output = np.zeros((n_rows, 14), dtype=int)

    score_pairs = data.reshape(-1, 14, 2)  # Shape: (n_rows, 14, 2)

    # A score pair is invalid if both values are equal (i.e., [0,0] or [1,1])
    missing_mask = find_missing_mask(score_pairs[:, :, 0], score_pairs[:, :, 1])  # Shape: (n_rows, 14)

    if run_type == "repeat":
        valid_mask = ~missing_mask # we want to select from non missing pairs
    else:
        valid_mask = missing_mask # we want to select from missing pairs

    for i in range(n_rows):
        valid_indices = np.where(valid_mask[i])[0]  # Domains with invalid score pairs
        if valid_indices.size > 0:
            chosen = np.random.choice(valid_indices)
            output[i, chosen] = 1

    return output


def split_encoding_and_scores(data: np.ndarray, dims=14):
    encoding = data[:, :dims]
    scores = data[:, dims:]
    return encoding, scores


# take in a dataframe, find predictions and return the predictions
def find_random_predictions(model, data, run_type):
    random_encoding = create_random_encoding(data, run_type=run_type)
    x_random = add_encoding(data, random_encoding)
    predictions = inference(model, x_random)
    return random_encoding, predictions



# given scores with missing indicators (x) and target (y), return a list of predictions according to which index is 1 in encoding
def predict_all_domains(model, x, y, loop_range):
    prediction_list = []
    rows, cols = y.shape
    # loop through fourteen domains, get the predictions and store the predictions for that domain only in a list
    for domain in loop_range:
        single_encoding = create_single_encoding(rows, cols, domain)
        x_single = add_encoding(x, single_encoding)
        single_prediction = inference(model, x_single)
        prediction_list.append(single_prediction[:, domain])
    
    matrix = np.column_stack(prediction_list)
    return matrix

def max_prediction_from_difference_pair(difference_matrix, prediction_matrix, current_matrix, run_type):
    """
    For each row, find the index of the largest improvement among the domains
    where current_matrix is 'missing' (i.e., [0, 0] or [1, 1]).

    Parameters:
        difference_matrix (np.ndarray): of shape (N, D)
        prediction_matrix (np.ndarray): of shape (N, D)
        current_matrix (np.ndarray): of shape (N, D, 2), score pairs

    Returns:
        Tuple[np.ndarray, np.ndarray]: (max_values, max_indices)
    """
    # Step 1: Create a boolean mask of missing values: [0,0] or [1,1]
    current_matrix_pairs = current_matrix.reshape(-1, 14, 2)

    eq_mask = current_matrix_pairs[:, :, 0] == current_matrix_pairs[:, :, 1]
    val_mask = (current_matrix_pairs[:, :, 0] == 0) | (current_matrix_pairs[:, :, 0] == 1)
    missing_mask = eq_mask & val_mask  # Shape: (N, D)

    # Step 2: Allocate outputs
    max_indices = np.full(difference_matrix.shape[0], np.nan)
    max_values = np.full(difference_matrix.shape[0], np.nan)

    if run_type == "repeat":
        valid_mask = ~missing_mask
    else:
        valid_mask = missing_mask

    # Step 3: Iterate through each row
    for i in range(difference_matrix.shape[0]):
        valid_indices = np.where(valid_mask[i])[0]
        if valid_indices.size > 0:
            valid_differences = difference_matrix[i, valid_indices]
            max_idx = np.argmax(valid_differences)
            max_indices[i] = valid_indices[max_idx]
            max_values[i] = prediction_matrix[i, valid_indices[max_idx]]

    return max_values, max_indices


# given x and y and original dataframe, find the encoding that results in the best scores, return encoding and predictions
def find_best_idx_pred(model, x, y, missing_counts, run_type):
    prediction_matrix = predict_all_domains(model, x, y, missing_counts)
    difference = prediction_matrix - x[:, ::2]
    # Find the index of the max difference for each row
    max_values, max_indices = max_prediction_from_difference_pair(difference, prediction_matrix, x, run_type)
    # Create a zero matrix of shape (100000, 14)
    future_scores_best, best_encoding = reconstruct_max_matrices(max_values, max_indices, prediction_matrix.shape)
    return best_encoding, future_scores_best


def reconstruct_max_matrices(max_values, max_indices, shape):
    max_values_matrix = np.zeros(shape)  # Matrix for max values
    max_indices_matrix = np.zeros(shape, dtype=int)  # Matrix for 1s at max indices

    # Iterate through rows
    for i in range(shape[0]):
        if not np.isnan(max_indices[i]):  # Ensure there's a valid index
            col_idx = int(max_indices[i])
            max_values_matrix[i, col_idx] = max_values[i]
            max_indices_matrix[i, col_idx] = 1  # Mark the index with 1

    return max_values_matrix, max_indices_matrix


def filter_n_missing(data: np.ndarray, n_missing: int) -> np.ndarray:
    """
    Filter rows where the number of missing domains (invalid score pairs) equals `n_missing`.
    Missing is defined as a score pair of [0, 0] or [1, 1].

    Returns:
        np.ndarray: Filtered array with only rows that have exactly `n_missing` invalid scores.
    """
    score_pairs = data[:, 14:42].reshape(-1, 14, 2)  # Shape: (n_rows, 14, 2)
    invalid_mask = score_pairs[:, :, 0] == score_pairs[:, :, 1]  # True if [0,0] or [1,1]
    invalid_count = np.sum(invalid_mask, axis=1)

    return data[invalid_count == n_missing]


def extract_score_pairs(data: np.ndarray) -> np.ndarray:
    """
    Extracts the score pairs from the input data.

    Parameters:
        data (np.ndarray): Input array with scores only

    Returns:
        np.ndarray: Array of shape (n_rows, 14, 2) containing the score pairs.
    """
    return data.reshape(-1, 14, 2)  # Reshape to (n_rows, 14, 2)


def overall_avg_improvement_with_std(data, pred_score):
    encoding, cur_score = split_encoding_and_scores(data)
    cur_score = cur_score[::2]

    pred_score = pred_score * encoding
    cur_score = cur_score * encoding

    # Compute improvement
    improvement = encoding * (pred_score - cur_score)

    # Extract nonzero values
    nonzero_improvement = improvement[improvement != 0]

    if len(nonzero_improvement) == 0:
        avg_improvement = 0
        std_dev = 0
        print("No sessions with nonzero improvement")
    else:
        avg_improvement = np.mean(nonzero_improvement)
        std_dev = np.std(nonzero_improvement, ddof=1)  # Using sample standard deviation (ddof=1)

        print("Number of predicted domains:", len(nonzero_improvement))
        print("Average improvement:", avg_improvement)
        print("Standard deviation:", std_dev)

    return avg_improvement, std_dev


def filter_sessions_by_missing_count_indices(data: np.ndarray, n_missing: int) -> np.ndarray:
    """
    Returns indices of rows where the number of missing domains equals `n_missing`.
    """
    score_pairs = extract_score_pairs(data)  # Shape: (n_rows, 14, 2)
    left, right = score_pairs[:, :, 0], score_pairs[:, :, 1]
    missing_mask = left == right
    missing_counts = np.sum(missing_mask, axis=1)
    return np.where(missing_counts == n_missing)[0]

def filter_sessions_by_missing_count(data: np.ndarray, n_missing: int) -> np.ndarray:
    """
    Filters rows from `data` where the number of missing domain scores equals `n_missing`.

    A domain is considered missing if its score pair is either [0, 0] or [1, 1].

    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=42), assuming 14 score pairs start at col 14.
        n_missing (int): Target number of missing score pairs.

    Returns:
        np.ndarray: Boolean mask array where True indicates rows with exactly `n_missing` missing domains.
    """
    score_pairs = extract_score_pairs(data)  # Shape: (n_rows, 14, 2)
    left, right = score_pairs[:, :, 0], score_pairs[:, :, 1]

    missing_mask = find_missing_mask(left, right)  # Shape: (n_rows, 14)
    missing_counts = np.sum(missing_mask, axis=1)  # Shape: (n_rows,)

    return missing_counts == n_missing


def compute_errors(gt_score, prediction_score):
    """
    Takes in score and ground truth, and computes the mean absolute error
    and the standard deviation of the ground truth scores.
    """
    # calculate mean absolute error
    mean_error = np.mean(np.abs(prediction_score - gt_score))
    # calculate ground truth std
    ground_truth_std = np.std(gt_score)
    
    return mean_error, ground_truth_std


def filter_with_masks(data, masks):
    """
    Filters the data based on the provided masks.
    
    Parameters:
    - data (np.ndarray): The data to be filtered.
    - masks (list of np.ndarray): List of boolean masks to filter the data.
    
    Returns:
    - np.ndarray: Filtered data.
    """
    for mask in masks:
        data = data[mask]
    return data


def compute_averages_and_stds(cur_scores, future_scores, masks):
    """
    masks: first mask for missing count, second mask for location of target value (encoding == 1)
    """
    difference = future_scores - cur_scores
    difference_filtered = filter_with_masks(difference, masks)

    average = np.mean(difference_filtered)
    std_dev = np.std(difference_filtered)

    return average, std_dev