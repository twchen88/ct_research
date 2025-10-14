import numpy as np
import torch
import random

from typing import Tuple, List, Any, Literal

from src.experiments.shared import *

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
    return np.sum(np.isnan(scores), axis=1) == missing_count

def average_scores_by_missing_counts(missing_counts: List[int], cur_scores: np.ndarray, future_scores: np.ndarray, encoding: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    avg_lst = []
    std_lst = []

    for mc in missing_counts:
        mc_mask = mask_by_missing_count(cur_scores, mc)

        # filter by missing count
        filtered_cur = cur_scores[mc_mask]
        filtered_fut = future_scores[mc_mask]
        filtered_enc = encoding[mc_mask]

        # only consider domains where encoding is 1
        enc_mask = filtered_enc.astype(bool)
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
    is_missing = np.isnan(scores)
    if run_type == "repeat":
        valid_mask = ~is_missing
    elif run_type == "nonrepeat":
        valid_mask = is_missing
    else:
        raise ValueError(f"Invalid run_type: {run_type}. Must be 'repeat' or 'nonrepeat'.")
    return valid_mask

def create_best(cur_score, pred_score, valid_mask):
    rows, cols = cur_score.shape
    best_enc = np.zeros((rows, cols), dtype=int)
    best_pred_scores = np.full((rows, cols), np.nan)

    # mask invalid domains with -inf so it won't be chosen
    masked_pred_score = np.where(valid_mask, pred_score, -np.inf)
    # choose best domain to encode
    for i in range(rows):
        if valid_mask[i].any():
            chosen = np.nanargmax(masked_pred_score[i])
            best_enc[i, chosen] = 1
            best_pred_scores[i, chosen] = masked_pred_score[i, chosen]

    return best_enc, best_pred_scores

def create_random(model, cur_score, valid_mask):
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

def choose_random(model, cur_score, run_type):
    valid_mask = find_valid_domains(decode_missing_indicator(cur_score), run_type=run_type)
    rand_enc, rand_pred_scores = create_random(model, cur_score, valid_mask)
    return rand_enc, rand_pred_scores

def choose_best(model, cur_score, missing_counts, run_type):
    predictions_all_domains = predict_all_domains(model, cur_score, loop_range=missing_counts)
    valid_mask = find_valid_domains(cur_score, run_type=run_type)
    best_enc, best_pred_scores = create_best(decode_missing_indicator(cur_score), predictions_all_domains, valid_mask)
    return best_enc, best_pred_scores