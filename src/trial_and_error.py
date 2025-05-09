# -*- coding: utf-8 -*-
# src/repeaters.py

'''
Created Fri 09 May 2025
Test things here
'''
from repeaters import RepeaterNetwork
from models import CNN, GNN
from gnn_env import Environment as gnnenv
from cnn_env import Environment as cnnenv

model = GNN()
env = gnnenv(GNN())
env.preview()