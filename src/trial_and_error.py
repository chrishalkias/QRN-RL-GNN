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

# net = RepeaterNetwork()
# model = GNN()
# experiment = Environment(model, n=5)
# new_model = GNN(output_dim=10)
# state = experiment.network.tensorState()
# output = experiment.model(state)

def yielder():
    for x in range(5):
        yield x
a = yielder()
print(a)
print(type(a))