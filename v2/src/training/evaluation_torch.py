import torch

from src.training.training_torch import MAE, MSE

def predict(model, data_loader):
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
    
    Parameters:
        predictions (torch.Tensor): The model predictions.
        targets (torch.Tensor): The ground truth values.
    
    Returns:
        torch.Tensor: The mean absolute error.
    """
    return MAE(predictions, targets)