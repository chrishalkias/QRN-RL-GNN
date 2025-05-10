# -*- coding: utf-8 -*-
# src/repeaters.py

'''
Created Fri 09 May 2025
Test things here
'''
from repeaters import RepeaterNetwork
from models import CNN, GNN
from gnn_env import Environment as EnvG
from cnn_env import Environment as EnvC

net = RepeaterNetwork()
model = GNN()
env = EnvG(model)

out = model.forward(net.tensorState())
print(sum(out[1]))
import datetime