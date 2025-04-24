# -*- coding: utf-8 -*-
# src/main.py
'''
Created on Thu 6 Mar 2025
The main simulation file.
Part of: MSc Thesis titled "Quantum Network Reinforcement Learning"
'''

import numpy as np
np.set_printoptions(legacy='1.25')

from gym_env import Experiment as exp
from agent import AgentDQN as agent



config = {
    'n_train'        : 7,
    'n_test'         : 7,
    'tau'            : 10_000,
    'p_entangle'     : 1,
    'p_swap'         : 1,
    'model'          : 'CNN', # Options: "DQN", "CNN"
    'train_agent'    : True,
    'train_steps'    : 10_000,
    'plot_metrics'   : True,
    'plot_loss'      : True,
    'train_logdir'   : './logs',
    'print_model'    : True,
    'evaluate_agent' : False,
    'render_eval'    : True,
}


if __name__ == "__main__":

    if config['model'] == "DQN": # Run DQN with gym

        experiment = exp(n=config['n_train'],
                         tau=config['tau'],
                         p_entangle=config['p_entangle'],
                         p_swap=config['p_swap'],
                         log_dir = config['train_logdir'])
        
        experiment.display_info()
        
        if config['train_agent']:
            experiment.train_agent(total_timesteps=config['train_steps'], plot=config['plot_metrics'], callback=False)

        if config['evaluate_agent']:
            experiment.test_agent(max_steps=10, render=config['render_eval'])
        
        experiment.env.close()
        print("Program exited with exit code 0")


            
    elif config['model'] == "CNN": # run DQN with CNN network and torch

        exp = agent(n = config['n_train'],
                    kappa = 1,
                    tau = config['tau'],
                    p_entangle = config['p_entangle'], 
                    p_swap = config['p_swap'],
                    lr = 0.001, 
                    gamma = 0.9, 
                    epsilon = 0.1
                    )
        
        if config['print_model']:
            exp.preview()
        
        if config['train_agent']:
            exp.train(episodes=config['train_steps'], plot=True)

        if config['evaluate_agent']:
            for kind in ['trained', 'random', 'swap_asap']:
                agent.test(config['n_test'], max_steps=100, kind=kind, plot=config['render_eval'])
    else:
        raise NameError('Model type not supported')