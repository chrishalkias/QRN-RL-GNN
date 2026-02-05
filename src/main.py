from agent import AgentGNN
import numpy as np
import matplotlib.pyplot as plt
from time import time
from stats import test_stats, n_scaling_test
plt.style.use('dark_background')
np.set_printoptions(legacy='1.25')

'''

	This file consists of 2 separete simulations
 
		1) Training and validation
		2) Scaling statistics of the relative performance 

'''
if __name__ == '__main__':

	t_0 = time()
	GET_SCALING_STATS = False
	GET_TESTING_RES = True


	# Initialize the agent
	agent = AgentGNN(n=4, 
					cutoff = 1000,
					tau=1000,
					p_entangle=0.3, 
					p_swap=0.8)

	# Run training loop
	agent.train(episodes=50_000)

	# Gather validation statistics
	test_stats(agent=agent, 
			experiments=10, 
			n_test = 6, 
			p_entangle=0.2, 
			p_swap=0.85, 
			cutoff = 1000,
			tau=1000, 
			rounds=1000) if GET_TESTING_RES else None

	# Observe the relative performance of tha agent as N varies
	n_scaling_test(experiments=10, 
				agent=agent, 
				N_range=range(5,10), 
				p_e=0.3, 
				p_s=0.8, 
				tau=100) if GET_SCALING_STATS else None
	
	print(f'Exited in {((time()-t_0)/60):.3f} min :)')
	