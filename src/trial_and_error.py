# -*- coding: utf-8 -*-
# src/trial_and error.py

'''
Created Fri 09 May 2025
Test things here
'''

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd

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

with open('logs/test_metrics.json', "r") as f:
    data = f.read()
print(data)



