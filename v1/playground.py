import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

layer = nn.Linear(2, 2)
print("weights of linear layer: ", layer.weight)
print("bias of linear layer: ", layer.bias)

sample = torch.tensor([1, 2], dtype=torch.float32)
print("original sample: ", sample)

calculated_output = torch.matmul(layer.weight, sample) + layer.bias
print("calcuated output: ", calculated_output)

output = layer(sample)
print("output after linear layer: ", output)