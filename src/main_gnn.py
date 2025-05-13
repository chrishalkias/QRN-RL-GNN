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

#SYSTEM CONFIGURATION
N_TRAIN         = 4
N_TEST          = 6
TAU             = 1_000
P_ENTANGLE      = 0.85
P_SWAP          = 0.85
KAPPA           = 1

#EXPERIMENT CONFIGURATION
TRAIN_AGENT     = True
TRAIN_STEPS     = 10_000
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-5
TEMPERATURE     = 1
GAMMA           = 0.9
EPSILON         = 0.1
PLOT_METRICS    = True
PLOT_LOSS       = True
PRINT_MODEL     = True
EVALUATE_AGENT  = True
TEST_STEPS      = 4_000
RENDER_EVAL     = True

#MODEL CONFIGURATION
INPUT_FEATURES  = 1
EMBEDDING_DIM   = 4
NUM_LAYERS      = 1
NUM_HEADS       = 2
HIDDEN_DIM      = 32
UNEMBEDDING_DIM = 64
OUTPUT_DIM      = 4


model = GNN(
        node_dim          = INPUT_FEATURES, 
        embedding_dim     = EMBEDDING_DIM,
        num_layers        = NUM_LAYERS,
        num_heads         = NUM_HEADS,
        hidden_dim        = HIDDEN_DIM, 
        unembedding_dim   = UNEMBEDDING_DIM, 
        output_dim        = OUTPUT_DIM, 
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
        trainer.trainQ_tensor(episodes=TRAIN_STEPS, plot=True)

    if EVALUATE_AGENT:

        for kind in ['trained', 'random', 'alternating']:

            exp.test(n_test    = N_TEST, 
                     max_steps = TEST_STEPS, 
                     kind      = kind, 
                     plot      = RENDER_EVAL)

    print(" ------------- Simulation end :-) ------------- ")