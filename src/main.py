# -*- coding: utf-8 -*-
# src/main.py
'''
Created on Thu 6 Mar 2025
The main simulation file.
Part of: MSc Thesis titled "Quantum Network Reinforcement Learning"
'''

import numpy as np
np.set_printoptions(legacy='1.25')

from environment import Environment
from models import CNN


config = {
    'n_train'        : 4,
    'n_test'         : 4,
    'tau'            : 100_000,
    'p_entangle'     : .85,
    'p_swap'         : .85,
    'model'          : 'cnn', # Options: "mlp", "cnn", "gnn"
    'algorithm'      : 'Q-learning', #Options: 'Q-learning', 'Reinforce'
    'train_agent'    : True,
    'train_steps'    : 300_000,
    'plot_metrics'   : True,
    'plot_loss'      : True,
    'print_model'    : True,
    'evaluate_agent' : True,
    'render_eval'    : True,}

model = CNN(convolutions = 2,
            pooling_dim = 16,
            embeding_dim = 32,
            hidden_dim = 128,
            unembeding_dim = 8,)

if __name__ == "__main__":

    if config['model'] == "cnn": # run DQN with CNN network and torch
        exp = Environment(
            model=model,
            n=config['n_train'],
            kappa=1,
            tau=config['tau'],
            p_entangle=config['p_entangle'], 
            p_swap=config['p_swap'],
            lr=0.001, 
            gamma=0.9, 
            epsilon=0.1)
        
        exp.preview() if config['print_model'] else None
        
        if config['train_agent']:

            if config['algorithm'] == 'Q-learning':
                exp.trainQ(episodes=config['train_steps'], plot=True)

            elif config['algorithm'] == 'REINFORCE':   
                pass

        if config['evaluate_agent']:
            for kind in ['trained', 'random', 'alternating']:
                exp.test(n_test=config['n_test'], max_steps=100, kind=kind, plot=config['render_eval'])
    else:
        raise NameError('Model type not supported')
    
    print("Program exited with exit code 0")