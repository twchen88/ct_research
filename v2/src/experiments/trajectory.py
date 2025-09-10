import torch
import numpy as np

from typing import Tuple, List

from src.experiments.shared import create_single_encoding, inference, add_encoding
from src.data.encoding import create_missing_indicator

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
    max_indices = np.full(difference_matrix.shape[0], np.nan)  # Store max indices
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
    min_indices = np.full(difference_matrix.shape[0], np.nan)  # Store min indices
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
    random_indices = np.full(difference_matrix.shape[0], np.nan)  # Store random indices
    random_values = np.full(difference_matrix.shape[0], np.nan)  # Store corresponding prediction values

    # Iterate through each row
    for i in range(difference_matrix.shape[0]):
        valid_indices = np.where(nan_mask[i])[0]  # Get column indices where current_matrix has NaN
        if valid_indices.size > 0:
            chosen_idx = np.random.choice(valid_indices)  # Choose one at random
            random_indices[i] = chosen_idx
            random_values[i] = prediction_matrix[i, chosen_idx]  # Get corresponding prediction value

    return random_values, random_indices


def find_next_domain(initial_scores: np.ndarray, model: torch.nn.Module, mode: str) -> Tuple[int, float]:
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
    prediction_list = [] # initialize where to store the prediction scores
    for i in range(14):
        rows, cols = initial_scores.shape
        # create encoding
        encoding = create_single_encoding(rows, cols, i)
        # create data to put into model
        x = create_missing_indicator(initial_scores)
        # add encoding to the data
        x = add_encoding(x, encoding)
        # predict the score if its domain i and append to prediction list
        single_prediction = inference(model, x)
        prediction_list.append(single_prediction[:, i])
    prediction_matrix = np.column_stack(prediction_list)
    difference = prediction_matrix - np.nan_to_num(initial_scores)

    index = np.arange(14).reshape(1, -1)  # create index for columns
    value = np.empty((1, 14))
    value[:] = np.nan  # initialize value with nans

    # choose based on mode
    if mode == "best":
        # find max indices
        value, index = max_prediction_from_difference(difference, prediction_matrix, initial_scores)
    elif mode == "middle":
        value, index = random_prediction_from_difference(difference, prediction_matrix, initial_scores)
    elif mode == "worst":
        value, index = min_prediction_from_difference(difference, prediction_matrix, initial_scores)
    return int(index[0]), value[0]

def trajectory(model: torch.nn.Module, mode: str) -> Tuple[List[float], np.ndarray, List[int]]:
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
    performance = [] # known domain average, s_i, at time step i
    order = [] # order of domains practiced
    current_scores = np.empty((1,14))
    current_scores[:] = np.nan # current scores, filled by nans first
    # go through 14 steps of predictions
    for i in range(14):
        # find predicted domain and score in this step according to mode
        next_domain, next_score = find_next_domain(current_scores, model, mode)
        # update current scores
        current_scores[0, next_domain] = next_score
        # calculate known domain average
        cur_mean = np.nanmean(current_scores)
        # append to score list and order list
        performance.append(cur_mean)
        order.append(next_domain + 1)
    return performance, current_scores, order