from agent import AgentGNN
import numpy as np
import matplotlib.pyplot as plt
from stats import test_stats, n_scaling_test
plt.style.use('dark_background')
np.set_printoptions(legacy='1.25')

# This file consists of 2 separete simulations
# 
# 1) Training and validation
# 2) Scaling statistics of the relative performance 

if __name__ == '__main__':

<<<<<<< HEAD
	#Toggle if scaling performance is evaluated
	GET_SCALING_STATS = False


	# Re-initialize the agent, now to be used as a zero-shot learner
	agent = AgentGNN(n=4, 
					cutoff = 100,
					tau=100,
					p_entangle=.1, 
					p_swap=.85)

	#Train ONCE to use for testing (also save the model)
	agent.train(episodes=40_000, plot=True, save_model=True)

	# Now gather validation statistics
	test_stats(agent=agent, 
			experiments=5, 
			n_test = 6, 
			p_entangle=.05, 
			p_swap=.95, 
			cutoff = 1000,
			tau=1000, 
			rounds=10_000) 

	# Observe the relative performance of tha agent as N varies
	n_scaling_test(experiments=10, 
				agent=agent, 
				N_range=range(5,10), 
				p_e=0.3, 
				p_s=0.8, 
				tau=100) if GET_SCALING_STATS else None
	
=======
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
>>>>>>> 727d82ce2499ad8f2cc793b76aebdd9697a3cd3c
