import numpy as np
import torch
import random

from typing import Tuple, List, Any, Literal

from src.experiments.shared import *

"""
src/experiments/aggregate_average.py
--------------------------------
This module contains functions to process experimental data used by 04_aggregate_average.py. Any helper function that is specific to this experiment is defined here.
"""

def filter_rows_by_sum(data: np.ndarray, col_range: slice, sum_threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters for rows where the sum of specified range of columns falls below a given threshold.

    Returns:
    - filtered_data (np.ndarray): Rows where column sum <= threshold
    - sum_mask (np.ndarray): Boolean mask of kept rows
    """
    sum_mask = data[:, col_range].sum(axis=1) <= sum_threshold
    filtered_data = data[sum_mask]
    return filtered_data, sum_mask

def find_missing_mask(x1, x2, eps=1e-8):
    """
    Given two arrays x1 and x2, return a boolean mask where the pairs (same index) are missing
    - i.e., both values are equal and either 0 or 1 (i.e., [0,0] or [1,1]).
    """
    eq = np.isclose(x1, x2, atol=eps, rtol=0)  # they are (almost) equal
    is_0_or_1 = np.isclose(x1, 0.0, atol=eps, rtol=0) | np.isclose(x1, 1.0, atol=eps, rtol=0)  # they are (almost) 0 or 1
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
        data (np.ndarray): Array with shape (n_rows, 28). Contains current scores and complements only.
        run_type (str): Either "repeat" or "nonrepeat". If "repeat", select from non-missing pairs; if "nonrepeat", select from missing pairs.

    Returns:
        np.ndarray: Binary array of shape (n_rows, 14) with a single 1 randomly placed per row
                    where an invalid score pair exists.
    """
    n_rows = data.shape[0]
    output = np.zeros((n_rows, 14), dtype=int)

    score_pairs = data.reshape(-1, 14, 2)  # Shape: (n_rows, 14, 2)

    # A score pair is invalid if both values are equal (i.e., [0,0] or [1,1])
    missing_mask = find_missing_mask(score_pairs[:, :, 0], score_pairs[:, :, 1])  # Shape: (n_rows, 14)

    if run_type == "repeat":
        valid_mask = ~missing_mask # we want to select from non missing pairs
    elif run_type == "nonrepeat":
        valid_mask = missing_mask # we want to select from missing pairs
    else:
        raise ValueError("run_type must be either 'repeat' or 'nonrepeat'")

    for i in range(n_rows):
        valid_indices = np.where(valid_mask[i])[0]  # Domains with invalid score pairs
        if valid_indices.size > 0:
            chosen = np.random.choice(valid_indices)
            output[i, chosen] = 1

    return output


def split_encoding_and_scores(data: np.ndarray, dims=14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array that combines encodings and scores, split them into two arrays and return encoding and scores separately.

    Parameters:
        data (np.ndarray): Input array with shape (n_rows, >=28).
        dims (int): Number of encoding columns. Default is 14.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (encoding, scores)
    """
    if data.shape[1] < dims:
        raise ValueError(f"Data must have at least {dims} columns for encoding.")
    encoding = data[:, :dims]
    scores = data[:, dims:]
    return encoding, scores


def find_random_predictions(model: torch.nn.Module, data: np.ndarray, run_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take in a dataframe, find random predictions for the given data using the specified run type.
    If run_type is "repeat", select from non-missing pairs; if "nonrepeat", select from missing pairs.
    Returns the random encoding and the corresponding predictions.

    Parameters:
        model: Trained model for inference.
        data (np.ndarray): Input data array with shape (n_rows, ==28, scores only).
        run_type (str): Either "repeat" or "nonrepeat".

    Returns:
        Tuple[np.ndarray, np.ndarray]: (random_encoding, predictions)

    """
    random_encoding = create_random_encoding(data, run_type=run_type)
    x_random = add_encoding(data, random_encoding)
    assert x_random.shape[1] == 42, "Input to model must have shape (N, 42)"
    predictions = inference(model, torch.from_numpy(x_random).float())
    return random_encoding, predictions



def predict_all_domains(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, loop_range: List[int]) -> np.ndarray:
    """
    Given scores with missing indicators (x) and target (y), return a list of predictions according to which index is 1
    in encoding.

    Parameters:
        model (torch.nn.Module): The trained model for inference.
        x (np.ndarray): Input data array with shape (n_rows, 28).
        y (np.ndarray): Target data array with shape (n_rows, 14).
        loop_range (range): Range of domain indices to loop through (e.g., range(14)).

    Returns:
        np.ndarray: Prediction matrix of shape (n_rows, 14) with predictions for each domain.
    """
    prediction_list = []
    rows, cols = y.shape
    # loop through fourteen domains, get the predictions and store the predictions for that domain only in a list
    for domain in loop_range:
        single_encoding = create_single_encoding(rows, cols, domain)
        print("single encoding: ", single_encoding.shape)
        x_single = add_encoding(x, single_encoding)
        print("x: ", x.shape)
        print("x single: ", x_single.shape)
        single_prediction = inference(model, torch.from_numpy(x_single).float())
        print("single prediction: ", single_prediction.shape)
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


def find_best_idx_pred(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, missing_counts: List[int], run_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given x and y and original dataframe, find the encoding that results in the best scores, return encoding and predictions

    Parameters:
        model (torch.nn.Module): The trained model for inference.
        x (np.ndarray): Current Score data array with shape (n_rows, 28).
        y (np.ndarray): Target data array with shape (n_rows, 14).
        missing_counts (List[int]): List of missing counts to consider.
        run_type (str): Either "repeat" or "nonrepeat".

    Returns:
        Tuple[np.ndarray, np.ndarray]: (best_encoding, best_predictions)
    """
    prediction_matrix = predict_all_domains(model, x, y, missing_counts)
    difference = prediction_matrix - x[:, ::2]
    # Find the index of the max difference for each row
    max_values, max_indices = max_prediction_from_difference_pair(difference, prediction_matrix, x, run_type)
    # Create a zero matrix of shape (100000, 14)
    future_scores_best, best_encoding = reconstruct_max_matrices(max_values, max_indices, prediction_matrix.shape)
    return best_encoding, future_scores_best


def reconstruct_max_matrices(max_values: np.ndarray, max_indices: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the max values and their corresponding indices, reconstruct two matrices:
    1. A matrix with the max values placed at their respective indices, and 0s elsewhere.
    2. A binary matrix with 1s at the positions of the max indices, and 0s elsewhere.

    Parameters:
        max_values (np.ndarray): Array of max values for each row.
        max_indices (np.ndarray): Array of indices corresponding to the max values for each row.
        shape (tuple): Desired shape of the output matrices (n_rows, n_cols).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (max_values_matrix, max_indices_matrix)
    """
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


def decode_missing_indicator(encoded: np.ndarray) -> np.ndarray:
    """
    Vectorized reverse of create_missing_indicator().
    Input:  (n, 28) where each pair [a,b] is either [x, 1-x] (observed) or [0,0]/[1,1] (missing).
    Output: (n, 14) original scores, with np.nan for missing.
    """
    if encoded.ndim != 2:
        raise ValueError("encoded must be a 2D array")
    n, m = encoded.shape
    if m % 2 != 0:
        raise ValueError(f"Expected even number of columns, got {m}")
    if m != 28:
        # Not strictly required, but helps catch shape drift
        raise ValueError(f"Expected 28 columns (14 pairs), got {m}")

    pairs = encoded.reshape(n, m // 2, 2)           # (n, 14, 2)
    a = pairs[..., 0].astype(float)                  # first of each pair (the original value if observed)
    b = pairs[..., 1]

    # Missing if pair is [0,0] or [1,1]
    eps = 1e-12
    missing = (np.abs(a - b) < eps) & ((np.abs(a) < eps) | (np.abs(a - 1) < eps))

    decoded = a.copy()
    decoded[missing] = np.nan
    assert decoded.shape == (n, 14), f"Expected shape (n,14), got {decoded.shape}"
    return decoded

def overall_avg_improvement_with_std(data: np.ndarray, pred_score: np.ndarray) -> Tuple[np.floating[Any] | Literal[0], np.floating[Any] | Literal[0]]:
    """
    Given the an array of encoding and scores combined as well as an array of the predicted scores, 
    Find the average improvement and standard deviation of the improvement for the nonzero improvements.

    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=28), assuming 14 score pairs start at col 14.
        pred_score (np.ndarray): Array of predicted scores with shape (n_rows, 14).

    Returns:
        Tuple[np.floating[Any], np.floating[Any]]: (average improvement, standard deviation)
    """
    encoding, cur_score = split_encoding_and_scores(data)
    cur_score = decode_missing_indicator(cur_score)

    # Compute improvement
    improvement = (pred_score - cur_score)[encoding == 1]

    if len(improvement) == 0:
        avg_improvement = 0
        std_dev = 0
        print("No sessions with nonzero improvement")
    else:
        avg_improvement = np.mean(improvement)
        std_dev = np.std(improvement, ddof=1)  # Using sample standard deviation (ddof=1)

        print("Number of predicted domains:", len(improvement))
        print("Average improvement:", avg_improvement)
        print("Standard deviation:", std_dev)

    return avg_improvement, std_dev


def filter_sessions_by_missing_count_indices(data: np.ndarray, n_missing: int) -> np.ndarray:
    """
    Returns indices of rows where the number of missing domains equals `n_missing`.

    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=42), assuming 14 score pairs start at col 14.
        n_missing (int): Target number of missing score pairs

    Returns:
        np.ndarray: Array of indices where the number of missing score pairs equals `n_missing`.
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


def compute_errors(gt_score: np.ndarray, prediction_score: np.ndarray) -> Tuple[np.floating[Any] | Literal[0], np.floating[Any] | Literal[0]]:
    """
    Takes in score and ground truth, and computes the mean absolute error
    and the standard deviation of the ground truth scores.

    Parameters:
        gt_score (np.ndarray): Ground truth scores.
        prediction_score (np.ndarray): Predicted scores.
    
    Returns:
        Tuple[float, float]: (mean absolute error, standard deviation of ground truth)
    """
    # calculate mean absolute error
    mean_error = np.mean(np.abs(prediction_score - gt_score))
    # calculate ground truth std
    ground_truth_std = np.std(gt_score)
    
    return mean_error, ground_truth_std


def filter_with_masks(data: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """
    Filters the data based on the provided masks, masks can be in a list.
    
    Parameters:
        data (np.ndarray): The data to be filtered.
        masks (list of np.ndarray): List of boolean masks to filter the data.

    Returns:
        np.ndarray: Filtered data.
    """
    for mask in masks:
        data = data[mask]
    return data


def safe_mean_std(x: np.ndarray) -> Tuple[np.floating[Any] | Literal[0], np.floating[Any] | Literal[0]]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.float64(np.nan), np.float64(np.nan)
    if x.size == 1:
        return np.float64(x[0]), np.float64(np.nan)
    return np.float64(np.mean(x)), np.float64(np.std(x, ddof=1))


def compute_averages_and_stds(cur_scores: np.ndarray, future_scores: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.floating[Any] | Literal[0], np.floating[Any] | Literal[0]]:
    """
    Computes the average and standard deviation of the improvements between current and future scores,
    applying the provided masks to filter the data.

    List of masks - first mask for missing count, second mask for location of target value (encoding == 1)

    Parameters:
        cur_scores (np.ndarray): Current scores.
        future_scores (np.ndarray): Future scores.
        masks (list of np.ndarray): List of boolean masks to filter the data.
    
    Returns:
        Tuple[float, float]: (average improvement, standard deviation of improvement)
    """
    difference = np.where(np.isnan(cur_scores), future_scores, future_scores - cur_scores)
    difference_filtered = filter_with_masks(difference, masks)

    if np.any(difference_filtered < 0):
        print("Warning: difference_filtered contains negative values.")

    average, std_dev = safe_mean_std(difference_filtered)

    return average, std_dev


def evaluate_error_by_missing_count(test_x, test_y, test_predictions, dims=14):
    encoding, cur_score = split_encoding_and_scores(test_x, dims=dims)
    future_score_gt = test_y

    mean_errors_list = []
    ground_truth_std_list = []
    masks = []

    ground_truth_dict = {}
    missing_counts = list(range(0, dims))

    for n in missing_counts:
        filter_mask = filter_sessions_by_missing_count(cur_score, n)
        filtered_encoding = encoding[filter_mask]

        masks = [filter_mask, (filtered_encoding == 1)]

        filtered_gt = filter_with_masks(future_score_gt, masks)
        filtered_pred = filter_with_masks(test_predictions, masks)

        ground_truth_dict[str(n)] = filtered_gt

        if filtered_gt.size == 0:
            mean_errors_list.append(np.nan)
            ground_truth_std_list.append(np.nan)
            continue

        mean_error, std_dev = compute_errors(filtered_gt, filtered_pred)
        mean_errors_list.append(mean_error)
        ground_truth_std_list.append(std_dev)

    return missing_counts, mean_errors_list, ground_truth_std_list, ground_truth_dict


def compute_avg_std_selected(
    cur_scores: np.ndarray,        # shape (N, 14)
    future_scores: np.ndarray,     # shape (N, 14)
    row_mask: np.ndarray,          # shape (N,) -> from filter_sessions_by_missing_count
    encoding_mask: np.ndarray,     # shape (N, 14) -> True where chosen domain(s), i.e., encoding==1
    nonrepeat_baseline_zero: bool = True
) -> Tuple[np.floating[Any] | Literal[0], np.floating[Any] | Literal[0]]:
    """
    1) Filter rows (sessions) by row_mask.
    2) Compute improvements element-wise:
         - if nonrepeat_baseline_zero: improvement = future (baseline 0 when current is NaN)
         - else: improvement = future - current, but if current is NaN treat as 0 => improvement = future
    3) Select ONLY elements indicated by encoding_mask in the filtered rows.
    4) Return mean/std over those selected elements.
    """
    # --- shape checks
    cur_scores = np.asarray(cur_scores, dtype=float)
    future_scores = np.asarray(future_scores, dtype=float)
    row_mask = np.asarray(row_mask, dtype=bool)
    encoding_mask = np.asarray(encoding_mask, dtype=bool)

    N, C1 = cur_scores.shape
    N2, C2 = future_scores.shape
    if cur_scores.ndim != 2 or future_scores.ndim != 2 or C1 != C2:
        raise ValueError(f"cur_scores {cur_scores.shape} and future_scores {future_scores.shape} must be 2D with same second dim")
    if row_mask.shape != (N,):
        raise ValueError(f"row_mask must be shape {(N,)}, got {row_mask.shape}")
    if encoding_mask.shape != (N, C1):
        raise ValueError(f"encoding_mask must be shape {(N, C1)}, got {encoding_mask.shape}")

    # --- filter rows
    cur_f = cur_scores[row_mask]        # (K, 14)
    fut_f = future_scores[row_mask]     # (K, 14)
    enc_f = encoding_mask[row_mask]     # (K, 14)


    violations = (enc_f == 1) & ~np.isnan(cur_f)

    if np.any(violations):
        rows, cols = np.where(violations)
        print(
            f"Warning: Found {len(rows)} violations where encoding==1 but current score is not NaN."
        )
        print(f"Example indices (row, col): {list(zip(rows[:10], cols[:10]))}")
    # --- compute improvement
    # If nonrepeat: baseline is 0 for missing (or for all; both yield fut_f when cur is NaN)
    if nonrepeat_baseline_zero:
        improvement = np.where(np.isnan(cur_f), fut_f, fut_f - cur_f)
    else:
        improvement = np.where(np.isnan(cur_f), fut_f, fut_f - cur_f)

    # --- pick only encoded elements
    selected_vals = improvement[enc_f]  # 1D: all True positions flattened

    # Optional: sanity checks per row (one-hot expectation)
    # rows_with_any = enc_f.any(axis=1).sum()
    # rows_with_none = (~enc_f.any(axis=1)).sum()
    # rows_with_multi = (enc_f.sum(axis=1) > 1).sum()

    # --- warn on unexpected negatives (beyond tiny FP noise)
    if np.any(selected_vals < -1e-12):
        print("Warning: selected improvements contain negative values.")

    return safe_mean_std(selected_vals)


def average_scores_by_missing_counts(missing_counts, current_scores, future_scores, encoding):
    avg_lst = []
    std_lst = []
    
    for n in missing_counts:
        print(f"Computing averages for missing count: {n}")

        row_mask = filter_sessions_by_missing_count(current_scores, n)
        print(f"Number of sessions with {n} missing domains: {np.sum(row_mask)}")

        encoding_mask = (encoding == 1)  # shape (N, 14)

        decoded_current_scores = decode_missing_indicator(current_scores)  # shape (N, 14)
        avg, std = compute_avg_std_selected(
            cur_scores=decoded_current_scores,      # (N,14)
            future_scores=future_scores,    # (N,14)
            row_mask=row_mask,                      # (N,)
            encoding_mask=encoding_mask,          # (N,14)
            nonrepeat_baseline_zero=True            # set True for nonrepeat runs
        )

        avg_lst.append(avg)
        std_lst.append(std)

    return avg_lst, std_lst