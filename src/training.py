from base.agent import AgentGNN

if __name__ == '__main__':

	#Initialize Agent
	agent = AgentGNN(n=4, 
					cutoff = 1000,
					tau=1000,
					p_entangle=0.2, 
					p_swap=0.88)

	# Run training loop
	agent.train(episodes=80_000,
				jitter = 1000,
				n_range=[4,6],
				save_model=True)
	