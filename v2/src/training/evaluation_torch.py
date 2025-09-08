import torch

from src.training.model_torch import Predictor
from torch.utils.data import DataLoader
from src.training.training_torch import MAE

"""
src/training/evaluation_torch.py
This module provides functions for evaluating a PyTorch predictor model.
* predict: Generates predictions using the model and a data loader.
* evaluate_error: Computes the mean absolute error between predictions and targets.
"""

def predict(model: Predictor, data_loader: DataLoader) -> torch.Tensor:
    """
    Generate predictions using the provided model and data loader.
    
    Parameters:
        model: The trained PyTorch model.
        data_loader: DataLoader providing the input data, usually a test dataset loader.

    Returns:
        torch.Tensor: The model predictions.
    """
    model.eval()

    outputs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            output = model(inputs)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)  # Concatenate all outputs
    return outputs


def evaluate_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Custom evaluation function that computes the mean absolute error between predictions and targets.
    This function uses the MAE function defined in training_torch.py.
    
    Parameters:
        predictions (torch.Tensor): The model predictions.
        targets (torch.Tensor): The ground truth values.
    
    Returns:
        torch.Tensor: The mean absolute error.
    """
    return MAE(predictions, targets)