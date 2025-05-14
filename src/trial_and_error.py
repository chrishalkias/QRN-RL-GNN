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

from tqdm import tqdm
epochs=50
num_steps=1000
batch_size = 100
count = 0
for epoch in tqdm(range(epochs)):
    for start in range(0, num_steps, batch_size):
        count +=1
print(count,  epochs * num_steps / batch_size)




