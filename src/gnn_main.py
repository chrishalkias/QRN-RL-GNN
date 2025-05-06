# -*- coding: utf-8 -*-
# src/main.py
'''
Created on Thu 6 Mar 2025
The main simulation file for GNN."
'''

import numpy as np
from gnn_environment import Environment
from models import GNN
np.set_printoptions(legacy='1.25')


sys_config = {
    'n_train'        : 4,
    'n_test'         : 6,
    'tau'            : 10_000,
    'p_entangle'     : .85,
    'p_swap'         : .85,
    'kappa'          : 1, # Global depolarizer, legacy code
    } 

agent_config = {
    'train_agent'    : True,
    'train_steps'    : 10_000,
    'learning_rate'  : 0.01,
    'gamma'          : 0.9,
    'epsilon'        : 0.1,
    'plot_metrics'   : True,
    'plot_loss'      : True,
    'print_model'    : True,
    'evaluate_agent' : True,
    'test_steps'     : 10_000,
    'render_eval'    : True,   
    }         

model_config = {}

model = GNN(
            node_dim        = 1, # always
            embedding_dim   = 16,
            num_layers      = 3,
            num_heads       = 4,
            hidden_dim      = 64, 
            unembedding_dim = 16, 
            output_dim      = 4, # always
            ) 

exp = Environment(
            model      = model,
            n          = sys_config['n_train'],
            kappa      = sys_config['kappa'],
            tau        = sys_config['tau'],
            p_entangle = sys_config['p_entangle'], 
            p_swap     = sys_config['p_swap'],
            lr         = agent_config['learning_rate'], 
            gamma      = agent_config['gamma'], 
            epsilon    = agent_config['epsilon']
            )

if __name__ == "__main__":

    if agent_config['train_agent']:
        exp.trainQ(episodes=agent_config['train_steps'], plot=True)

    if agent_config['evaluate_agent']:
        for kind in ['trained', 'random', 'alternating']:
            exp.test(n_test=sys_config['n_test'], 
                        max_steps=agent_config['test_steps'], 
                        kind=kind, 
                        plot=agent_config['render_eval'])

    print(":-)")