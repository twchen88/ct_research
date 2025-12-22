import torch.nn as nn

"""
Defines PyTorch layers
"""

TORCH_LAYER_MAP = {
    "Linear": nn.Linear,
    "Dropout": nn.Dropout,
    "BatchNorm1d": nn.BatchNorm1d
}

TORCH_ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh
}