# -*- coding: utf-8 -*-
# src/gnn_main.py

'''
         ██████  ███    ██ ███    ██     ███    ███  █████  ██ ███    ██ 
        ██       ████   ██ ████   ██     ████  ████ ██   ██ ██ ████   ██ 
        ██   ███ ██ ██  ██ ██ ██  ██     ██ ████ ██ ███████ ██ ██ ██  ██ 
        ██    ██ ██  ██ ██ ██  ██ ██     ██  ██  ██ ██   ██ ██ ██  ██ ██ 
         ██████  ██   ████ ██   ████     ██      ██ ██   ██ ██ ██   ████    

                            Created Fri 09 May 2025
                            The main simulation file
'''

import os
import numpy as np
from gnn_env import Environment

from models import GNN
from rl_loops import QTrainer
import plot_config
np.set_printoptions(legacy='1.25')
#Sorry PEP8 but equal signs are aligned

# SYSTEM CONFIGURATION
N_TRAIN         = 4
N_TEST          = 4
TAU             = 1_000
P_ENTANGLE      = .15
P_SWAP          = .15
KAPPA           = 1 # legacy code

# TRAINING CONFIGURATION
TRAIN_AGENT     = True
TRAIN_STEPS     = 10_000
LEARNING_RATE   = 3e-4
GAMMA           = 0.95
EPSILON         = 0.2

# EXPERIMENT CONFIGURATION
WEIGHT_DECAY    = 1e-5
TEMPERATURE     = 1
PLOT_METRICS    = True

# TESTING CONFIGURATION
EVALUATE_AGENT  = True
TEST_STEPS      = 1_000
RENDER_EVAL     = True

#MODEL CONFIGURATION
EMBEDDING_DIM   = 4
NUM_LAYERS      = 1
NUM_HEADS       = 2
HIDDEN_DIM      = 32
UNEMBEDDING_DIM = 64


model = GNN(
        node_dim          = 1, 
        embedding_dim     = EMBEDDING_DIM,
        num_layers        = NUM_LAYERS,
        num_heads         = NUM_HEADS,
        hidden_dim        = HIDDEN_DIM, 
        unembedding_dim   = UNEMBEDDING_DIM, 
        output_dim        = 4, 
        ) 

exp = Environment(
            model        = model,
            n            = N_TRAIN,
            kappa        = KAPPA,
            tau          = TAU,
            p_entangle   = P_ENTANGLE, 
            p_swap       = P_SWAP,
            lr           = LEARNING_RATE, 
            weight_decay = WEIGHT_DECAY,
            gamma        = GAMMA, 
            epsilon      = EPSILON,
            temperature  = TEMPERATURE
        )

if __name__ == "__main__":

    plot_config.set()
    exp.preview()

    if TRAIN_AGENT:
        trainer = QTrainer(experiment=exp)
        trainer.trainQ_tensor(episodes=TRAIN_STEPS, plot=PLOT_METRICS,)

    if EVALUATE_AGENT:
        exp.test(n_test=N_TEST, max_steps=TEST_STEPS,plot=RENDER_EVAL,)
    print(' :) ')
