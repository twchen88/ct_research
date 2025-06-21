import torch
import numpy as np

def deterministic_backend():
    """
    Set the PyTorch backend to deterministic mode to ensure reproducibility.
    """
    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a PyTorch model from the specified path.
    
    :param model_path: Path to the saved model file.
    :return: Loaded PyTorch model.
    """
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()  # Set the model to evaluation mode
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
    return predictions

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
