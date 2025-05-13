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
state = net.tensorState()

net.entangle(edge=(0,1))
print(net.getLink(edge=(0,1), linkType=1))
net.resetState()
print(net.getLink(edge=(0,1), linkType=1))




