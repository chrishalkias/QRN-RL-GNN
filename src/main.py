from agent import AgentGNN
import numpy as np
import matplotlib.pyplot as plt
from stats import train_stats, test_stats, n_scaling_test
plt.style.use('dark_background')
np.set_printoptions(legacy='1.25')

# This file consists of 2 separete simulations
# 2) Training and validation of the agent
# 3) Scaling statistics of the relative performance of the agent

if __name__ == '__main__':

    #Toggle which of the 3 parts you want
    (GET_TEST_STATS, GET_SCALING_STATS) = (True, True)

    #-----------------------PART I (ACTUAL TRAINING AND TESTING OF THE AGENT)-----------------------
    if GET_TEST_STATS:

        TRAIN_STEPS = 30_000

        N_TRAIN = 4

        P_ENTANGLE_TRAIN = 0.6
        P_SWAP_TRAIN = 0.8
        CUTOFF = 50 # after cutoff/tau time the link is discarded
        TAU = 100

        # Re-initialize the agent, now to be used as a zero-shot learner
        agent = AgentGNN(n=N_TRAIN, 
                        cutoff = CUTOFF,
                        p_entangle=P_ENTANGLE_TRAIN, 
                        p_swap=P_SWAP_TRAIN, 
                        tau=TAU)

        #Train ONCE to use for testing (also save the model)
        agent.train(episodes=TRAIN_STEPS, plot=True, save_model=True)

        # Initialize validation variables
        TEST_TRIALS = 5
        ROUNDS = 1_000

        N_TEST = 6
        P_ENTANGLE_TEST = 0.3
        P_SWAP_TEST = 0.8
        TAU_TEST = 100

        # Now gather validation statistics
        test_stats(agent=agent, 
                N=TEST_TRIALS, 
                n_test = N_TEST, 
                p_entangle=P_ENTANGLE_TEST, 
                p_swap=P_SWAP_TEST, 
                tau=TAU_TEST, 
                rounds=ROUNDS) 

    #-----------------------PART II (CHECK AGENT RELATIVE SCALING COMPARED TO SWAP-ASAP)-----------------------
    if GET_SCALING_STATS:
        SCALING_TRIALS = 10
        N_RANGE = range(5, 10)
        TRAIN_STEPS_SCALING = 50_000

        P_ENTANGLE_SCALING = 0.3
        P_SWAP_SCALING = 0.8

        #Train again ONCE again to use for testing (also save the model)
        agent.train(episodes=TRAIN_STEPS_SCALING, plot=True, save_model=True)

        # Observe the relative performance of tha agent as N varies
        n_scaling_test(N=SCALING_TRIALS, 
                    agent=agent, 
                    N_range=N_RANGE, 
                    p_e=P_ENTANGLE_SCALING, 
                    p_s=P_SWAP_SCALING, 
                    tau=TAU)