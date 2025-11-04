import numpy as np
import torch

from typing import Tuple, List, Any, Literal

from ct.experiments.shared import *

"""
src/experiments/aggregate_average.py
--------------------------------
This module contains functions to process experimental data used by 04_aggregate_average.py. Any helper function that is specific to this experiment is defined here.
"""

def filter_rows_by_sum(data: np.ndarray, col_range: slice, sum_threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters for rows where the sum of specified range of columns falls below a given threshold.

    Parameters:
        data (np.ndarray): The input data array.
        col_range (slice): The range of columns to consider for summation.
        sum_threshold (int): The threshold value for filtering.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (filtered_data, sum_mask)
    """
    sum_mask = data[:, col_range].sum(axis=1) <= sum_threshold
    filtered_data = data[sum_mask]
    return filtered_data, sum_mask

def find_missing_mask(x1: np.ndarray, x2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Given two arrays x1 and x2, return a boolean mask where the pairs (same index) are missing
    - i.e., both values are equal and either 0 or 1 (i.e., [0,0] or [1,1]).
    Parameters:
        x1 (np.ndarray): First array.
        x2 (np.ndarray): Second array.
        eps (float): Tolerance for floating point comparison. Default to 1e-8.
    
    Returns:
        np.ndarray: Boolean mask array where True indicates missing pairs.
    
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
    n_domains = 14
    encodings, score = split_encoding_and_scores(data, dims=n_domains)
    score_pairs = score.reshape(-1, n_domains, 2)

    # Determine if a score pair is invalid: [0, 0] or [1, 1]
    invalid_mask = find_missing_mask(score_pairs[:, :, 0], score_pairs[:, :, 1])  # (N, 14)

    # A session is invalid (i.e., not repeat) if any domain with encoding==1 has an invalid score pair
    violating_domains = (encodings == 1) & invalid_mask  # (N, 14)

    is_repeat = ~np.any(violating_domains, axis=1)

    return is_repeat


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

def predict_all_domains(model: torch.nn.Module, x: np.ndarray, loop_range: List[int]) -> np.ndarray:
    """
    Given scores with missing indicators (x) and target (y), return a list of predictions according to which index is 1
    in encoding.

    Parameters:
        model (torch.nn.Module): The trained model for inference.
        x (np.ndarray): Input data array with shape (n_rows, 28), i.e. current scores.
        loop_range (range): Range of domain indices to loop through (e.g., range(14)).

    Returns:
        np.ndarray: Prediction matrix of shape (n_rows, 14) with predictions for each domain.
    """
    prediction_list = []
    rows, cols = x.shape[0], x.shape[1]//2
    # loop through fourteen domains, get the predictions and store the predictions for that domain only in a list
    for domain in loop_range:
        single_encoding = create_single_encoding(rows, cols, domain)
        x_single = add_encoding(x, single_encoding)
        single_prediction = inference(model, torch.from_numpy(x_single).float())
        prediction_list.append(single_prediction[:, domain])
    
    matrix = np.column_stack(prediction_list)
    return matrix


def mask_by_missing_count(scores: np.ndarray, missing_count: int) -> np.ndarray:
    """
    Create a boolean mask for rows where the number of missing values matches the specified count.

    Parameters:
        scores (np.ndarray): 2D array of scores with shape (n_rows, n_domains).
        missing_count (int): Target number of missing values.

    Returns:
        np.ndarray: Boolean mask array where True indicates rows with exactly `missing_count` missing values
    """
    return np.sum(np.isnan(scores), axis=1) == missing_count

def average_scores_by_missing_counts(missing_counts: List[int], cur_scores: np.ndarray, future_scores: np.ndarray, encoding: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Compute average and standard deviation of score differences for each missing count.

    Parameters:
        missing_counts (List[int]): List of missing counts to evaluate.
        cur_scores (np.ndarray): Current scores array with shape (n_rows, n_domains).
        future_scores (np.ndarray): Future scores array with shape (n_rows, n_domains).
        encoding (np.ndarray): Encoding array with shape (n_rows, n_domains).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: (avg_lst, std_lst)
            - avg_lst (List[np.ndarray]): List of average differences for each missing count.
            - std_lst (List[np.ndarray]): List of standard deviations for each missing count.
    """
    avg_lst = []
    std_lst = []

    for mc in missing_counts:
        mc_mask = mask_by_missing_count(cur_scores, mc)

        # filter by missing count
        filtered_cur = cur_scores[mc_mask]
        filtered_fut = future_scores[mc_mask]
        filtered_enc = encoding[mc_mask]

        # only consider domains where encoding is 1
        enc_mask = (filtered_enc == 1)
        # if no rows with mc missing, append NaNs
        if np.sum(mc_mask) == 0 or np.sum(enc_mask) == 0:
            avg_lst.append(np.nan)
            std_lst.append(np.nan)
            continue

        # compute difference between future and current, treating NaN as 0 baseline
        difference = np.where(np.isnan(filtered_cur), filtered_fut, filtered_fut - filtered_cur)
        selected = difference[enc_mask]

        # compute average and std
        avg = np.mean(selected)
        std = np.std(selected, ddof=1)  # sample std

        # append to lists
        avg_lst.append(avg)
        std_lst.append(std)

    return avg_lst, std_lst

def find_valid_domains(scores: np.ndarray, run_type: str) -> np.ndarray:
    """
    Given scores and run_type, return a boolean mask indicating valid domains for choosing as a candidate for encoding. Used for simulating different experimental domain selecting strategies.
    If repeat: only domains with observed scores (not missing) are valid.
    If nonrepeat: only domains with missing scores are valid.

    Parameters:
        scores (np.ndarray): 2D array of scores with shape (n_rows, n_domains).
        run_type (str): Type of run, either "repeat" or "nonrepeat".
    
    Returns:
        np.ndarray: Boolean mask array where True indicates valid domains.
    """
    is_missing = np.isnan(scores)
    if run_type == "repeat":
        valid_mask = ~is_missing
    elif run_type == "nonrepeat":
        valid_mask = is_missing
    else:
        raise ValueError(f"Invalid run_type: {run_type}. Must be 'repeat' or 'nonrepeat'.")
    return valid_mask

def create_best(cur_score: np.ndarray, pred_score: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the best encoding and predicted scores based on the current scores and predictions.

    Parameters:
        cur_score (np.ndarray): Current scores array with shape (n_rows, n_domains * 2).
        pred_score (np.ndarray): Predicted scores array with shape (n_rows, n_domains).
        valid_mask (np.ndarray): Boolean mask indicating valid domains, shape (n_rows, n_domains).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (best_enc, best_pred_scores)
            - best_enc (np.ndarray): Best encoding array with shape (n_rows, n_domains).
            - best_pred_scores (np.ndarray): Best predicted scores array with shape (n_rows, n_domains).
    """
    rows, cols = valid_mask.shape
    best_enc = np.zeros((rows, cols), dtype=int)
    best_pred_scores = np.full((rows, cols), np.nan)

    # mask invalid domains with -inf so it won't be chosen
    masked_pred_score = np.where(valid_mask, pred_score, -np.inf)
    # decode cur_score to original scores and impute missing with NaN
    cur_score_decoded = decode_missing_indicator(cur_score)

    # compute difference between predicted and current, treating NaN as 0 baseline
    # if current is NaN, should be nonrepeat, so diff = pred - 0 = pred
    diff = np.where(np.isnan(cur_score_decoded),
                        masked_pred_score,
                        masked_pred_score - cur_score_decoded)
    
    # choose best domain to encode
    for i in range(rows):
        if valid_mask[i].any():
            chosen = np.nanargmax(diff[i])
            best_enc[i, chosen] = 1
            best_pred_scores[i, chosen] = masked_pred_score[i, chosen]

    return best_enc, best_pred_scores

def create_random(model: torch.nn.Module, cur_score: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random encoding and predicted scores based on the current scores and predictions.

    Parameters:
        model (torch.nn.Module): The trained model for inference.
        cur_score (np.ndarray): Current scores array with shape (n_rows, n_domains * 2).
        valid_mask (np.ndarray): Boolean mask indicating valid domains, shape (n_rows, n_domains).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (rand_enc, rand_pred_scores)
            - rand_enc (np.ndarray): Random encoding array with shape (n_rows, n_domains).
            - rand_pred_scores (np.ndarray): Random predicted scores array with shape (n_rows, n_domains).
    """
    rows, cols = valid_mask.shape
    rand_enc = np.zeros((rows, cols), dtype=int)
    rand_pred_scores = np.full((rows, cols), np.nan)
    # randomly choose domains to encode
    for i in range(rows):
        valid_indices = np.where(valid_mask[i])[0]
        if valid_indices.size > 0:
            chosen = np.random.choice(valid_indices)
            rand_enc[i, chosen] = 1

    rand_pred_scores = inference(model, torch.from_numpy(add_encoding(cur_score, rand_enc)).float())

    return rand_enc, rand_pred_scores


def choose_best_and_random(model: torch.nn.Module, cur_score: np.ndarray, missing_counts: List[int], valid_mask: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Choose the best and random encodings and predicted scores.

    Parameters:
        model (torch.nn.Module): The trained model for inference.
        cur_score (np.ndarray): Current scores array with shape (n_rows, n_domains * 2).
        missing_counts (List[int]): List of missing counts to evaluate.
        valid_mask (np.ndarray): Boolean mask indicating valid domains, shape (n_rows, n_domains).

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            - ((best_enc, best_pred_scores), (rand_enc, rand_pred_scores))
    """
    predictions_all_domains = predict_all_domains(model, cur_score, loop_range=missing_counts)
    best_enc, best_pred_scores = create_best(cur_score, predictions_all_domains, valid_mask)
    random_enc, random_pred_scores = create_random(model, cur_score, valid_mask)
    return (best_enc, best_pred_scores), (random_enc, random_pred_scores)

def filter_n_missing(data: np.ndarray, n_missing: int) -> np.ndarray:
    """
    Filter rows where the number of missing domains (invalid score pairs) equals `n_missing`.
    Missing is defined as a score pair of [0, 0] or [1, 1].

    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=42), assuming 14 score pairs start at col 14.
        n_missing (int): Target number of missing score pairs.

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
    else:
        avg_improvement = np.mean(improvement)
        std_dev = np.std(improvement, ddof=1)  # Using sample standard deviation (ddof=1)

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

    A domain is considered missing if its score is np.nan

    Parameters:
        data (np.ndarray): Array with shape (n_rows, >=42), assuming 14 score pairs start at col 14.
        n_missing (int): Target number of missing score pairs.

    Returns:
        np.ndarray: Boolean mask array where True indicates rows with exactly `n_missing` missing domains.
    """
    return np.isnan(data).sum(axis=1) == n_missing


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

def evaluate_error_by_missing_count(test_x: np.ndarray, test_y: np.ndarray, test_predictions: np.ndarray, dims: int = 14) -> Tuple[List[int], List[float], List[float], dict]:
    """
    Given test inputs and predictions, evaluate mean errors by missing count.

    Parameters:
        test_x (np.ndarray): Test input data with shape (n_rows, n_domains * 3; encoding + encoded current scores).
        test_y (np.ndarray): Ground truth scores with shape (n_rows, n_domains).
        test_predictions (np.ndarray): Predicted scores with shape (n_rows, n_domains).
        dims (int): Number of domains. Default is 14 (n_domains).
    
    Returns:
        Tuple[List[int], List[float], List[float], dict]: (missing_counts, mean_errors_list, ground_truth_std_list, ground_truth_dict)
            - missing_counts (List[int]): List of missing counts evaluated.
            - mean_errors_list (List[float]): List of mean absolute errors for each missing count
            - ground_truth_std_list (List[float]): List of standard deviations of ground truth for each missing count.
            - ground_truth_dict (dict): Dictionary mapping missing count to ground truth scores.
    """
    encoding, cur_score = split_encoding_and_scores(test_x, dims=dims)
    future_score_gt = test_y

    mean_errors_list = []
    ground_truth_std_list = []
    masks = []

    ground_truth_dict = {}
    missing_counts = list(range(0, dims))

    for n in missing_counts:
        filter_mask = filter_sessions_by_missing_count(decode_missing_indicator(cur_score), n)
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