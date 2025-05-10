# -*- coding: utf-8 -*-
# src/gnn_main.py

'''
Created Fri 09 May 2025
Test things here
'''

import os
import numpy as np
from gnn_env import Environment
from models import GNN
from rl_loops import Trainer
import plot_config


sys_config = {
    'n_train'        : 4,
    'n_test'         : 4,
    'tau'            : 50_000,
    'p_entangle'     : .85,
    'p_swap'         : .85,
    'kappa'          : 1, # Global depolarizer, legacy code
    } 

# RUN SOME HPO ON THIS!
agent_config = {
    'train_agent'    : True,
    'train_steps'    : 20_000,
    'learning_rate'  : 3e-4,
    'weight_decay'   : 1e-5,
    'temperature'    : 1,
    'gamma'          : 0.9, #Q
    'epsilon'        : 0.1, #Q
    'plot_metrics'   : True,
    'plot_loss'      : True,
    'print_model'    : True,
    'evaluate_agent' : True,
    'test_steps'     : 1_000,
    'render_eval'    : True,   
    }   
  

model_config = {
    'input_features' : 1, # always
    'embedding_dim'  : 4,
    'num_layers'     : 1,
    'num_heads'      : 2,
    'hidden_dim'     : 32, 
    'unembedding_dim': 64, 
    'output_dim'     : 4, # always

}

if __name__ == "__main__":

    np.set_printoptions(legacy='1.25')
    plot_config.set()

    model = GNN(
            node_dim          = model_config['input_features'], 
            embedding_dim     = model_config['embedding_dim'],
            num_layers        = model_config['num_layers'],
            num_heads         = model_config['num_heads'],
            hidden_dim        = model_config['hidden_dim'], 
            unembedding_dim   = model_config['unembedding_dim'], 
            output_dim        = model_config['output_dim'], 
            ) 

    exp = Environment(
                model        = model,
                n            = sys_config['n_train'],
                kappa        = sys_config['kappa'],
                tau          = sys_config['tau'],
                p_entangle   = sys_config['p_entangle'], 
                p_swap       = sys_config['p_swap'],
                lr           = agent_config['learning_rate'], 
                weight_decay = agent_config['weight_decay'],
                gamma        = agent_config['gamma'], 
                epsilon      = agent_config['epsilon'],
                temperature  = agent_config['temperature']
            )
    
    exp.preview()

    if agent_config['train_agent']:
        
        trainer = Trainer(experiment=exp)
        trainer.trainQ(episodes=agent_config['train_steps'], plot=True)

    if agent_config['evaluate_agent']:
        for kind in ['trained', 'random', 'alternating']:
            exp.test(n_test=sys_config['n_test'], 
                        max_steps=agent_config['test_steps'], 
                        kind=kind, 
                        plot=agent_config['render_eval'])

    print(":-)")