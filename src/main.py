from agent import AgentGNN
import numpy as np
import matplotlib.pyplot as plt
from stats import train_stats, test_stats, n_scaling_test
plt.style.use('dark_background')
np.set_printoptions(legacy='1.25')

# CAUTION!! THE STATISTICS ACQUIRED FROM THIS FILE CANNOT MAKE USE
# OF THE TRAINED AGENT. SOMWEHERE BETWEEN THE LINES 44-78 SOMETHING
# GOES WRONG!! FOR AN OVERVIEW OF THE PROJECTS FUNCTIONALITY PLEASE
# REFER TO THE NOTEBOOK!


# This file consists of 3 separete simulations
# 1) Training statistics of the Agent
# 2) Testing statistics of the Agent
# 3) Scaling statistics of the relative performance of the agent

if __name__ == '__main__':

    #Toggle which of the 3 parts you want
    (GET_TRAIN_STATS, GET_TEST_STATS, GET_SCALING_STATS) = (False, True, False)

    #-----------------------PART 1 (GATHER GENERAL TRAINING STATISTICS)-----------------------
    if GET_TRAIN_STATS:

        TRIALS = 5
        STEPS = 10_000

        N_STATS = 4
        P_E_STATS = 0.8
        P_S_STATS = 0.8
        TAU_STATS = 700

        # Initialize the agent
        dummy_agent = AgentGNN(n=N_STATS, 
                            p_entangle=P_E_STATS, 
                            p_swap=P_S_STATS, 
                            tau=TAU_STATS)

        # Gather training statistics
        train_stats(dummy_agent, N=TRIALS, steps=STEPS)


    #-----------------------PART 2 (ACTUAL TRAINING AND TESTING OF THE AGENT)-----------------------
    if GET_TEST_STATS:
        TRAIN_STEPS = 1_000

        N_TRAIN = 4
        P_ENTANGLE_TRAIN = 0.6
        P_SWAP_TRAIN = 0.8
        TAU_TRAIN = 700

        # Re-initialize the agent, now to be used as a zero-shot learner
        agent = AgentGNN(n=N_TRAIN, 
                        p_entangle=P_ENTANGLE_TRAIN, 
                        p_swap=P_SWAP_TRAIN, 
                        tau=TAU_TRAIN)

        #Train ONCE to use for testing (also save the model)
        agent.train(episodes=TRAIN_STEPS, plot=True, save_model=True)


        # Initialize validation variables
        TEST_TRIALS = 10
        ROUNDS = 1_000

        N_TEST = 8
        P_ENTANGLE_TEST = 0.3
        P_SWAP_TEST = 0.8
        TAU_TEST = 100

        # Now gather validation statistics
        test_stats(agent, 
                N=TEST_TRIALS, 
                n_test = N_TEST, 
                p_entangle=P_ENTANGLE_TEST, 
                p_swap=P_SWAP_TEST, 
                tau=TAU_TEST, 
                rounds=ROUNDS) 

    #-----------------------PART 3 (CHECK AGENT RELATIVE SCALING COMPARED TO SWAP-ASAP)-----------------------
    if GET_SCALING_STATS:
        SCALING_TRIALS = 5
        N_RANGE = range(5, 10)
        TRAIN_STEPS_SCALING = 50_000

        P_ENTANGLE_SCALING = 0.3
        P_SWAP_SCALING = 0.8
        TAU_SCALING = 100

        #Train again ONCE to use for testing (also save the model)
        agent.train(episodes=TRAIN_STEPS_SCALING, plot=True, save_model=True)

        # Observe the relative performance of tha agent as N varies
        n_scaling_test(N=SCALING_TRIALS, 
                    agent=agent, 
                    N_range=N_RANGE, 
                    p_e=P_ENTANGLE_SCALING, 
                    p_s=P_SWAP_SCALING, 
                    tau=TAU_SCALING)