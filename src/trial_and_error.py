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

with open("logs/test_metrics.json", "r") as f:
    data = json.load(f,)

df = pd.DataFrame(data=data)
df = df.rename(index={0 : 'avg_reward', 1: 'std_reward', 2: 'avg_fidelity', 3: 'std_fidelity'})
print(df[0])
df.pivot()
# x=['avg_reward', 'std_reward', 'avg_fidelity', 'std_fidelity']
# # plt.plot(df['trained'],'tab:green', ls=':', label='trained')
# # plt.plot(df['random'],'tab:green', ls=':', label='trained')
# plt.bar(x=x, height=df['trained'], width=.95,label='trained')
# plt.bar(x=x, height=df['alternating'], width=.75, label='alternating')
# plt.bar(x=x, height=df['swapASAP'], width=.55,label='swap-asap')
# plt.bar(x=x, height=df['random'], width=.35,label='random')
# plt.title('Testing metrics')
# plt.legend()
# plt.show()

x=['trained', 'random', 'swapASAP', 'alternating']
plt.bar(x=x, height=df['avg_reward'], width=.5)
plt.title('Testing metrics')
plt.legend()
plt.show()



