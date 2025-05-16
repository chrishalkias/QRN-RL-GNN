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

# SYSTEM CONFIGURATION
N_TRAIN         = 4
N_TEST          = 6
TAU             = 1_000
P_ENTANGLE      = .99
P_SWAP          = .99
KAPPA           = 1 # legacy code

# TRAINING CONFIGURATION
TRAIN_AGENT     = True
ALGORITHM       = 'QL'      # Options: 'QL', 'REINFORCE', 'PPO'
GAMMA           = 0.95
EPSILON         = 0.2

# DQN COnfig
TRAIN_STEPS     = 1_000
LEARNING_RATE   = 3e-4

# PPO CONFIG
NUM_STEPS       = 10_000 
EPOCHS          = 5
BATCH_SIZE      = 10


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
        
        assert ALGORITHM in ['QL', 'REINFORCE', 'PPO'], "Algorithm not in list"
        trainer = QTrainer(experiment=exp)

        if ALGORITHM == 'QL':
            trainer.trainQ_tensor(episodes=TRAIN_STEPS, plot=PLOT_METRICS)
        elif ALGORITHM == 'REINFORCE':
            raise Exception("REINFORCE not yet implemented")
        elif ALGORITHM == 'PPO':
            trainer.train_PPO(num_steps    = NUM_STEPS, 
                              epochs       = EPOCHS, 
                              batch_size   = BATCH_SIZE, 
                              gamma        = GAMMA, 
                              clip_epsilon = EPSILON,
                              plot         = PLOT_METRICS)

    if EVALUATE_AGENT:

        for kind in ['trained', 'random', 'alternating', 'swapASAP']:

            exp.test(n_test    = N_TEST, 
                     max_steps = TEST_STEPS, 
                     kind      = kind, 
                     plot      = RENDER_EVAL,
                     algorithm = ALGORITHM)

    print(f" {'-'*15} Simulation end :-) {'-'*15} ")