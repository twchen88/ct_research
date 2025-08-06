import torch
import numpy as np
from src.training.model_torch import Predictor

def deterministic_backend():
    """
    Set the PyTorch backend to deterministic mode to ensure reproducibility.
    """
    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a PyTorch model from a saved state_dict.

    Parameters:
        model_path (str): Path to the .pt or .pth file containing the state_dict.
        device (str): The device to map the model to ("cpu" or "cuda").
        model_class (type): The model class to instantiate.
        *model_args, **model_kwargs: Arguments to initialize the model.

    Returns:
        torch.nn.Module: The loaded model ready for inference.
    """
    # Step 1: Instantiate the model, assume Predictor is the model class for this version
    model = Predictor()
    
    # Step 2: Load the state_dict
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(state_dict)
    
    # Step 3: Set to eval mode
    model.to(device)
    model.eval()
    
    return model

def inference(model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
    """
    Perform inference using the provided model and data.
    
    :param model: The PyTorch model to use for inference.
    :param data: Input data as a PyTorch tensor.
    :return: Model predictions as a PyTorch tensor.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(data)
    return predictions.numpy()

# add encoding to scores and return a tensor that can be put directly into the model
def add_encoding(scores : np.ndarray, encoding : np.ndarray):
    return torch.from_numpy(np.hstack((encoding, scores))).float()


# given the number of rows and cols and index, return a numpy array of according size with all 0s except for specified column
def create_single_encoding(rows, cols, column_index):
    if column_index < 0 or column_index >= cols:
        raise ValueError("Column index is out of bounds.")

    # Create a zero matrix
    matrix = np.zeros((rows, cols), dtype=int)

    # Set all values in the specified column to 1
    matrix[:, column_index] = 1

    return matrix

def max_prediction_from_difference_pair(difference_matrix, prediction_matrix, current_matrix, run_type):
    """
    For each row, find the index of the largest improvement among the domains
    where current_matrix is 'missing' (i.e., [0, 0] or [1, 1]).

    Parameters:
        difference_matrix: np.ndarray of shape (N, D)
        prediction_matrix: np.ndarray of shape (N, D)
        current_matrix: np.ndarray of shape (N, D, 2), score pairs

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

def max_prediction_from_difference(difference_matrix, prediction_matrix, current_matrix):
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