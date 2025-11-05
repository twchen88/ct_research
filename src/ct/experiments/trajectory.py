import torch
import numpy as np

from typing import Tuple, List, Callable, Optional, Union

from ct.experiments.shared import create_single_encoding, inference, add_encoding
from ct.data.encoding import create_missing_indicator

"""
src/experiments/trajectory.py
--------------------------------
This module contains helper functions to determine the next domain to practice based on different strategies (best, random, worst).
"""

def max_prediction_from_difference(difference_matrix: np.ndarray, prediction_matrix: np.ndarray, current_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the maximum prediction improvement for each row based on the difference between current and predicted scores.

    Parameters:
        difference_matrix (np.ndarray): A 2D array where each element represents the difference between predicted and current scores.
        prediction_matrix (np.ndarray): A 2D array of predicted scores.
        current_matrix (np.ndarray): A 2D array of current scores, with NaNs indicating unpracticed domains.

    Returns:
        max_values (np.ndarray): An array containing the maximum predicted scores for each row where current scores are NaN.
        max_indices (np.ndarray): An array containing the column indices of the maximum predicted scores.
    """
    nan_mask = np.isnan(current_matrix)  # Boolean mask where True indicates NaN

    # Initialize arrays to store results
    max_indices = np.full(difference_matrix.shape[0], 9999, dtype=np.int64)  # Store max indices
    max_values = np.full(difference_matrix.shape[0], np.nan)  # Store corresponding prediction values

    # Iterate through each row
    for i in range(difference_matrix.shape[0]):
        valid_indices = np.where(nan_mask[i])[0]  # Get column indices where current_matrix has NaN
        if valid_indices.size > 0:
            valid_differences = difference_matrix[i, valid_indices]  # Select values where NaN exists in current_matrix
            max_idx = np.argmax(valid_differences)  # Find index of max value (relative to valid_indices)
            max_indices[i] = valid_indices[max_idx]  # Store original column index
            max_values[i] = prediction_matrix[i, valid_indices[max_idx]]  # Get corresponding prediction value

    return max_values, max_indices

def min_prediction_from_difference(difference_matrix: np.ndarray, prediction_matrix: np.ndarray, current_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the minimum prediction improvement for each row based on the difference between current and predicted scores.

    Parameters:
        difference_matrix (np.ndarray): A 2D array where each element represents the difference between predicted and current scores.
        prediction_matrix (np.ndarray): A 2D array of predicted scores.
        current_matrix (np.ndarray): A 2D array of current scores, with NaNs indicating unpracticed domains.

    Returns:
        min_values (np.ndarray): An array containing the minimum predicted scores for each row where current scores are NaN.
        min_indices (np.ndarray): An array containing the column indices of the minimum predicted scores.
    """
    nan_mask = np.isnan(current_matrix)  # Boolean mask where True indicates NaN

    # Initialize arrays to store results
    min_indices = np.full(difference_matrix.shape[0], 9999, dtype=np.int64)  # Store min indices
    min_values = np.full(difference_matrix.shape[0], np.nan)  # Store corresponding prediction values

    # Iterate through each row
    for i in range(difference_matrix.shape[0]):
        valid_indices = np.where(nan_mask[i])[0]  # Get column indices where current_matrix has NaN
        if valid_indices.size > 0:
            valid_differences = difference_matrix[i, valid_indices]  # Select values where NaN exists in current_matrix
            min_idx = np.argmin(valid_differences)  # Find index of max value (relative to valid_indices)
            min_indices[i] = valid_indices[min_idx]  # Store original column index
            min_values[i] = prediction_matrix[i, valid_indices[min_idx]]  # Get corresponding prediction value

    return min_values, min_indices


def random_prediction_from_difference(difference_matrix: np.ndarray, prediction_matrix: np.ndarray, current_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a random prediction improvement for each row based on the difference between current and predicted scores.

    Parameters:
        difference_matrix (np.ndarray): A 2D array where each element represents the difference between predicted and current scores.
        prediction_matrix (np.ndarray): A 2D array of predicted scores.
        current_matrix (np.ndarray): A 2D array of current scores, with NaNs indicating unpracticed domains.

    Returns:
        random_values (np.ndarray): An array containing the random predicted scores for each row where current scores are NaN.
        random_indices (np.ndarray): An array containing the column indices of the random predicted scores.
    """
    nan_mask = np.isnan(current_matrix)  # Boolean mask where True indicates NaN

    # Initialize arrays to store results
    random_indices = np.full(difference_matrix.shape[0], 9999, dtype=np.int64)  # Store random indices
    random_values = np.full(difference_matrix.shape[0], np.nan)  # Store corresponding prediction values

    # Iterate through each row
    for i in range(difference_matrix.shape[0]):
        valid_indices = np.where(nan_mask[i])[0]  # Get column indices where current_matrix has NaN
        if valid_indices.size > 0:
            chosen_idx = np.random.choice(valid_indices)  # Choose one at random
            random_indices[i] = chosen_idx
            random_values[i] = prediction_matrix[i, chosen_idx]  # Get corresponding prediction value

    return random_values, random_indices

def find_next_domain(
    initial_scores: np.ndarray,
    model: torch.nn.Module,
    mode: str,
    *,
    # injectables for easy testing
    create_single_encoding_fn: Optional[Callable[[int, int, int], np.ndarray]] = None,
    create_missing_indicator_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    add_encoding_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    inference_fn: Optional[Callable[[torch.nn.Module, torch.Tensor], Union[torch.Tensor, np.ndarray]]] = None,
    max_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    min_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    rand_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    num_domains: int = 14,
) -> Tuple[int, float]:
    """
    Take in initial data, mode, and model, return the next domain and predicted score according to mode

    Parameters:
        initial_scores (np.ndarray): A 2D array of initial scores with NaNs for unpracticed domains.
        model (torch.nn.Module): A trained model used for making predictions.
        mode (str): The strategy to use for selecting the next domain. Options are "best, random, worst".

    Returns:
        next_domain (int): The index of the next domain to practice (0-13 corresponds to domain 1-14).
        next_score (float): The predicted score for the selected domain.
    """

    # Bind defaults lazily (so tests can inject dummies)
    if create_single_encoding_fn is None:
        create_single_encoding_fn = create_single_encoding
    if create_missing_indicator_fn is None:
        create_missing_indicator_fn = create_missing_indicator
    if add_encoding_fn is None:
        add_encoding_fn = add_encoding
    if inference_fn is None:
        inference_fn = inference
    if max_fn is None:
        max_fn = max_prediction_from_difference
    if min_fn is None:
        min_fn = min_prediction_from_difference
    if rand_fn is None:
        rand_fn = random_prediction_from_difference

    # Build prediction matrix
    prediction_list = []
    rows, cols = initial_scores.shape
    for i in range(num_domains):
        encoding = create_single_encoding_fn(rows, cols, i)
        x = create_missing_indicator_fn(initial_scores)
        x = add_encoding_fn(x, encoding)
        single_prediction = inference_fn(model, torch.from_numpy(x).float())
        # Expect shape (rows, num_domains) from inference; pick column i
        # Allow inference_fn to return either a Tensor or a NumPy array and normalize to Tensor.
        if isinstance(single_prediction, np.ndarray):
            single_prediction = torch.from_numpy(single_prediction)
        single_prediction = single_prediction.detach().cpu().numpy()
        prediction_list.append(single_prediction[:, i])
    prediction_matrix = np.column_stack(prediction_list).astype(float)

    # Differences vs current (treat NaNs in current as zeros when computing diff)
    difference = prediction_matrix - np.nan_to_num(initial_scores)

    # Choose based on mode
    if mode == "best":
        value, index = max_fn(difference, prediction_matrix, initial_scores)
    elif mode in ("middle", "random"):
        value, index = rand_fn(difference, prediction_matrix, initial_scores)
    elif mode == "worst":
        value, index = min_fn(difference, prediction_matrix, initial_scores)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of 'best', 'random' (alias 'middle'), 'worst'.")

    return int(index[0]), float(value[0])


def trajectory(
    model: torch.nn.Module,
    mode: str,
    *,
    # Injectables for unit tests
    find_next_domain_fn: Optional[Callable[[np.ndarray, torch.nn.Module, str], Tuple[int, float]]] = None,
    num_domains: int = 14,
    num_steps: int = 14,
) -> Tuple[List[float], np.ndarray, List[int]]:
    """
    Take in model and mode ("best", "random", or "worst"), return a list of known domain averages across time, 
    a matrix of individual domain scores across time, and order of domains practiced.

    Parameters:
        model (torch.nn.Module): A trained model used for making predictions.
        mode (str): The strategy to use for selecting the next domain. Options are "best, random, worst".
    
    Returns:
        performance (List[float]): A list of known domain averages at each time step.
        current_scores (np.ndarray): A 2D array of individual domain scores at all time steps.
        order (List[int]): A list of the order in which domains were practiced (1-14).
    """
    if mode not in {"best", "random", "worst", "middle"}:
        raise ValueError("Unknown mode. Expected one of: 'best', 'random', 'worst' (alias: 'middle').")

    if find_next_domain_fn is None:
        # Fall back to the local implementation if import is unavailable (keeps tests and single-file usage working)
        find_next_domain_fn = find_next_domain

    performance: List[float] = []  # known domain averages over time
    order: List[int] = []          # 1-based domain indices
    current_scores = np.empty((1, num_domains), dtype=float)
    current_scores[:] = np.nan

    # run steps of predictions
    for _ in range(num_steps):
        next_domain, next_score = find_next_domain_fn(current_scores, model, mode)
        current_scores[0, next_domain] = float(next_score)

        cur_mean = float(np.nanmean(current_scores))
        performance.append(cur_mean)
        order.append(next_domain + 1)

    return performance, current_scores, order