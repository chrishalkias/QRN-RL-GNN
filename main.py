# -*- coding: utf-8 -*-

"""
Created on Thu 6 Mar 2025
The main simulation file. 
Dependencies: experiment.py, quantum_network_env.py, graph_models.py, repeater_network.py
Author: Chris Chalkias
Affiliation: Leiden University, Netherlands
Part of: MSc Thesis titled "Quantum Network Reinforcement Learning"
"""

import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt

from experiment import Experiment as exp
from custom_agent import AgentCNN as agent

N = 7
TAU = 10_000
P_ENTANGLE = 1
P_SWAP = 1

MODEL = "DQN"  # Options: "DQN", "CNN"
TRAIN_AGENT = True
TRAIN_STEPS = 10_000
PLOT_TRAINING_METRICS, PLOT_LOSS = True, True
TRAIN_LOG_DIR = "./logs/"

EVALUATE_AGENT = False
RENDER_EVALUATION = True
FILE_NAME = None
TRAINING_PLOTS = None
TRAINNING_PARAMETERS = None
MODEL_FILES = None

if __name__ == "__main__":
    if MODEL == "DQN":
        experiment = exp(n=N, tau=TAU,
                    p_entangle=P_ENTANGLE,
                    p_swap=P_SWAP,
                    log_dir = TRAIN_LOG_DIR,
                            )
        experiment.display_info()
        if TRAIN_AGENT:
            experiment.train_agent(total_timesteps=TRAIN_STEPS, plot=PLOT_TRAINING_METRICS, callback=False)
        if EVALUATE_AGENT:
            experiment.test_agent(max_steps =10, render=RENDER_EVALUATION)
        if True:
            experiment.env.close()
            print("Program exited with exit code 0")
            
    elif MODEL == "CNN":
        exp = agent(n=4, kappa=1, tau=1_000_000,p_entangle=.85, p_swap=.85, lr=0.001, gamma=0.9, epsilon=0.1)
        if TRAIN_AGENT:
            agent.train(episodes=1_000, plot=True)
        if EVALUATE_AGENT:
            agent.test_agent(max_steps=100)