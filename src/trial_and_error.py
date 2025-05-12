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

model = GNN()
experiment = Environment(model, n=5)
new_model = GNN(output_dim=10)
state = RepeaterNetwork().tensorState()


#load the saved model for evaluation
os.makedirs('logs/model_checkpoints/', exist_ok=True)
model.load_state_dict(torch.load('./logs/model_checkpoints/GNN_model.pth')) # Shape: [1, 4]
model.eval()
with torch.no_grad():
    output = model(state)
output = output.squeeze(0)
print("Model output:", output)
