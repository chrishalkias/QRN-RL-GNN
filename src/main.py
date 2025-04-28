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
from models import CNN


config = {
    'n_train'        : 7,
    'n_test'         : 5,
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

model = CNN(convolutions = 7,
            pooling_dim = 16,
            embeding_dim = 16,
            hidden_dim = 16,
            unembeding_dim = 8,)

if __name__ == "__main__":

    if config['model'] == "mlp": # Run DQN with gymnasium and PPO from SB3

        # # rerdundand, somehow breaks total_timesteps being an integer (???)
        # ppo_config = {
        #     'learning_rate' : 1e-4,   # Adjust learning rate
        #     'n_steps'       : 2048,   # Increase number of steps per update
        #     'batch_size'    : 64,     # Adjust batch size
        #     'gamma'         : 0.99,   # Discount factor
        #     'gae_lambda'    : 0.95,   # GAE parameter
        #     'clip_range'    : 0.2,    # Clip range for policy updates
        #     'ent_coef'      : 0.1,    # Entropy coefficient (encourage exploration)
        #     'verbose'       : 1,
        # }

        experiment = exp(
            n=config['n_train'],
            tau=config['tau'],
            p_entangle=config['p_entangle'],
            p_swap=config['p_swap'],
            # ppo_config = ppo_config,
            )
        
        experiment.display_info()
        
        if config['train_agent']:
            experiment.train_agent(total_timesteps=config['train_steps'], plot=config['plot_metrics'], callback=True)

        if config['evaluate_agent']:
            experiment.test_agent(max_steps=10, render=config['render_eval'])
        
        experiment.env.close()
        print("Program exited with exit code 0")


            
    elif config['model'] == "cnn": # run DQN with CNN network and torch

        exp = agent(model=model,
                    n=config['n_train'],
                    kappa=1,
                    tau=config['tau'],
                    p_entangle=config['p_entangle'], 
                    p_swap=config['p_swap'],
                    lr=0.001, 
                    gamma=0.9, 
                    epsilon=0.1
                    )
        
        if config['print_model']:
            exp.preview()
        
        if config['train_agent'] and (config['algorithm'] == 'Q-learning'):
            exp.trainQ(episodes=config['train_steps'], plot=True)
        elif config['train_agent'] and (config['algorithm'] == 'REINFORCE'):
            #exp.trainREINFORCE(episodes=config['train_steps'], plot=True)
            pass

        if config['evaluate_agent']:
            for kind in ['trained', 'random', 'alternating']:
                exp.test(n_test=config['n_test'], max_steps=10_000, kind=kind, plot=config['render_eval'])
    else:
        raise NameError('Model type not supported')