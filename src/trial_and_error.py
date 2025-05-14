# -*- coding: utf-8 -*-
# src/trial_and error.py

'''
Created Fri 09 May 2025
Test things here
'''

import numpy as np
import torch
import os

from repeaters import RepeaterNetwork
from gnn_env import Environment
from models import GNN
from rl_loops import QTrainer

net = RepeaterNetwork()
model = GNN()
experiment = Environment(model, n=5)
new_model = GNN(output_dim=10)
state = experiment.network.tensorState()
output = experiment.model(state)


def binary_tensor_to_int(onehot):
    """
    Converts a binary tensor of shape (4, N) to a unique integer.
    The left-most element is treated as the most significant bit.
    
    Args:
        tensor: torch.Tensor of shape (1, N) containing 0s and 1s
        
    Returns:
        int: Unique integer representation
    """
    onehot = onehot.view(-1)
    N = onehot.shape[0]
    # Create powers of 2 in descending order (MSB to LSB)
    powers = 2 ** torch.arange(N-1, -1, -1)
    return int((onehot * powers).sum().item())

# Example usage
onehot = experiment.out_to_onehot(output)  
unique_int = binary_tensor_to_int(onehot)
print(unique_int)  # Output: 10 (which is 1*8 + 0*4 + 1*2 + 0*1)





