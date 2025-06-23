import torch

from src.training.training_torch import MAE, MSE


def evaluate_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Custom evaluation function that computes the mean absolute error between predictions and targets.
    
    Parameters:
        predictions (torch.Tensor): The model predictions.
        targets (torch.Tensor): The ground truth values.
    
    Returns:
        torch.Tensor: The mean absolute error.
    """
    return MAE(predictions, targets)