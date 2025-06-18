import torch

"""
src/training/model_torch.py
----------------------------
This module defines a simple neural network model using PyTorch.
It is designed to take input of current scores and target domain encoding, and output predictions for the target domains.
This is the same model used in v1.
"""


class Predictor(torch.nn.Module):
    """
    A simple neural network model for predicting target domains based on current scores and target domain encoding.
    The model consists of two linear layers with a sigmoid activation function in between.
    """
    def __init__(self):
        super().__init__()
        n_domains = 14
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_domains * 3, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, n_domains)
        )

    def forward(self, x):
        return self.model(x)