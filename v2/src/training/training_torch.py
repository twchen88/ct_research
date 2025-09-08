import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple

from src.training.model_torch import Predictor

"""
src/training/training_torch.py
------------------------
This module provides functions for training a PyTorch model, including data preparation from model-ready format data, training and validation loops,
and custom dataset class for handling input and target data.
"""


class custom_dataset(Dataset):
    """
    Custom dataset class for handling input and target data in PyTorch.
    This class inherits from torch.utils.data.Dataset and implements the necessary methods to work with DataLoader.
    """
    def __init__(self, data: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.target[index, :]

def get_dataloader(dataset: Dataset, batch_size: int, suffle: bool) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Parameters:
        dataset (Dataset): The dataset to create a DataLoader for.
        batch_size (int): The number of samples per batch.
        suffle (bool): Whether to shuffle the data at every epoch.
    
    Returns:
        DataLoader: A DataLoader instance for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=suffle, num_workers=0, pin_memory=True)

def split_train_test(data: np.ndarray, ratio: float = 0.25, n_samples=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.
    
    Parameters:
        data (np.ndarray): The input data to be split.
        ratio (float): The proportion of the dataset to include in the test split.
        n_samples (int, optional): If specified, limit the number of samples in both train and test sets.
    
    Returns:
        tuple: A tuple containing the training and testing datasets as numpy arrays.
    """
    train_data, test_data = train_test_split(data, test_size=ratio)

    if n_samples is not None: # if n_samples is specified, limit the number of samples
        train_data = train_data[:n_samples].copy()
        test_data = test_data[:n_samples].copy()
    
    return train_data, test_data


def split_input_target(data: np.ndarray, dims: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the full data into input and target data.

    Parameters:
        data (np.ndarray): The full dataset containing both input and target data.
        dims (int): The number of dimensions for the input data. The target data will be the remaining dimensions.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors: input data and target data.
    The input data will have the first `dims` columns, and the target data will have the remaining columns.
    """
    input_data = data[:, :dims]
    target_data = data[:, dims:]
    return torch.from_numpy(input_data), torch.from_numpy(target_data)


def evaluate_loss(model, data_loader, loss_function):
    """
    Given a model and a DataLoader, evaluate the loss on the dataset.

    Parameters:
        model (Predictor, as defined in model_torch.py): The model to evaluate.
        data_loader (DataLoader): The DataLoader containing the dataset.
        loss_function (Callable): The loss function to use for evaluation.
    
    Returns:
        float: The average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # scale by batch size
    return total_loss / len(data_loader.dataset)


def train_one_epoch(model: Predictor, data_loader: DataLoader, loss_function: Callable, optimizer: torch.optim.Optimizer) -> float:
    """
    Train the model for one epoch using the provided DataLoader, loss function, and optimizer.
    Returns the average loss for the epoch over all samples in the DataLoader.

    Parameters:
        model (Predictor): The model to train.
        data_loader (DataLoader): The DataLoader containing the training dataset.
        loss_function (Callable): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters.

    Returns:
        float: The average loss for the epoch.
    """
    
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        output = model(batch_x)

        # Reshape only if needed and safe
        if batch_y.shape != output.shape:
            batch_y = batch_y.view_as(output)

        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return running_loss / total_samples


def train_model(model: Predictor, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int, optimizer: torch.optim.Optimizer,
    loss_function: Callable, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the model for a specific number of epochs, calling the train_one_epoch function for each epoch.
    Keeps track of training and validation loss history and returns them.

    Parameters:
        model (Predictor): The model to train.
        train_data_loader (DataLoader): The DataLoader for the training dataset.
        val_data_loader (DataLoader): The DataLoader for the validation dataset.
        epochs (int): The number of epochs to train the model.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_function (Callable): The loss function to use for training and validation.
        device (str): The device to run the training on ('cpu' or 'cuda'), currently not used.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the training and validation loss history.
    """

    train_loss_history = np.zeros(epochs + 1)
    val_loss_history = np.zeros(epochs + 1)

    # Initial loss before training
    train_loss_history[0] = (evaluate_loss(model, train_data_loader, loss_function))
    val_loss_history[0] = (evaluate_loss(model, val_data_loader, loss_function))

    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs", unit="epoch"):
        train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer)
        val_loss = evaluate_loss(model, val_data_loader, loss_function)

        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss

    return train_loss_history, val_loss_history


def MSE(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MSE loss function that computes the mean squared error between predictions and targets.
    """
    return torch.mean((predictions - targets) ** 2)


def MAE(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MAE loss function that computes the mean squared error between predictions and targets.
    """
    return torch.mean(torch.abs((predictions - targets)))


def get_optimizer(optimizer_name: str, learning_rate: float, model: Predictor) -> torch.optim.Optimizer:
    """
    Get the optimizer based on the specified name and learning rate.
    
    Parameters:
        optimizer_name (str): The name of the optimizer to use (e.g., 'adam', 'sgd').
        learning_rate (float): The learning rate for the optimizer.
        model (Predictor): The model for which the optimizer is being created.
    
    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

def get_loss_function(loss_function_name: str) -> Callable:
    """
    Get the loss function based on the specified name.
    
    Parameters:
        loss_function_name (str): The name of the loss function to use (e.g., 'mse', 'mae').
    
    Returns:
        Callable: A callable loss function.
    """
    if loss_function_name.lower() == 'mse':
        return MSE
    elif loss_function_name.lower() == 'mae':
        return MAE
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")