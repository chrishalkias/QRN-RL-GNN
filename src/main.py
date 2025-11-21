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

    GET_SCALING_STATS = True

    # Initialization
    agent = AgentGNN(n=4, 
                    cutoff = 50,
                    p_entangle=0.6, 
                    p_swap=0.8, 
                    tau=100)

    # Training
    agent.train(episodes=30_000, 
                plot=True, 
                save_model=True)

    # Validation
    test_stats(agent=agent, # Now gather validation statistics
            N=5, 
            n_test = 6, 
            p_entangle=0.3, 
            p_swap=0.8, 
            tau=100, 
            rounds=1_000) 

    # Observe the relative performance of tha agent as N varies
    n_scaling_test(N=10, 
                agent=agent, 
                N_range=range(5,10), 
                p_e=0.3, 
                p_s=0.8, 
                tau=100) if GET_SCALING_STATS else None