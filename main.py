# -*- coding: utf-8 -*-

"""
Created on Thu 6 Mar 2025
The main simulation file. 
Dependencies: experiment.py, quantum_network_env.py, graph_models.py, repeater_network.py
Author: Chris Chalkias
"""

import experiment as exp
import numpy as np
import matplotlib.pyplot as plt

N = 7
TAU = 10000
P_ENTANGLE = 1
P_SWAP = 1
MODEL = "DQN"
FILE_NAME = None
TRAINING_PLOTS = None
TRAINNING_PARAMETERS = None
MODEL_FILES = None

if __name__ == "__main__":
    experiment = exp(model = "DQN",
                    n=N, tau=TAU,
                    p_entangle=P_ENTANGLE,
                    p_swap=P_SWAP,
                    #file_name=FILE_NAME,
                    #training_plots=TRAINING_PLOTS,
                    #training_parameters=TRAINNING_PARAMETERS,
                    #model_files=MODEL_FILES,
                            )
    experiment.display_info()