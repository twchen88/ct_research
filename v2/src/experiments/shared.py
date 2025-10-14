import torch
import numpy as np
from src.training.model_torch import Predictor

"""
src/experiments/shared.py
-------------------------
Shared utilities for experiments, including model loading, inference, and data processing.
"""

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

def inference(model: torch.nn.Module, data: torch.Tensor) -> np.ndarray:
    """
    Perform inference using the provided model and data.

    Parameters:
        model (torch.nn.Module): The PyTorch model to use for inference.
        data (torch.Tensor): Input data as a PyTorch tensor.
    Returns:
        torch.Tensor: Model predictions as a PyTorch tensor.
    """
    print("Data shape:", data.shape)
    assert data.shape[1] == 42, "Input data must have shape (N, 42). Got shape {}".format(data.shape)
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(data)
    return predictions.numpy()


def add_encoding(scores : np.ndarray, encoding : np.ndarray) -> np.ndarray:
    """
    Add encoding in front of scores and return a tensor that can be put directly into the model

    Parameters:
        scores (np.ndarray): of shape (N, 28)
        encoding (np.ndarray): of shape (N, 14)
    
    Returns:
        np.ndarray: of shape (N, 42)
    """
    out = np.hstack((encoding, scores))
    return out


def create_single_encoding(rows, cols, column_index):
    """
    Given the number of rows and cols and index, return a numpy array of according size with 
    all 0s except for specified column

    Parameters:
        rows (int): number of rows
        cols (int): number of columns
        column_index (int): index of the column to set to 1
    """
    if column_index < 0 or column_index >= cols:
        raise ValueError("Column index is out of bounds.")

    # Create a zero matrix
    matrix = np.zeros((rows, cols), dtype=int)

    # Set all values in the specified column to 1
    matrix[:, column_index] = 1

    return matrix