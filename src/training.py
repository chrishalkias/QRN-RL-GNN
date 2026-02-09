from base.agent import AgentGNN

if __name__ == '__main__':

	#Initialize Agent
	agent = AgentGNN(n=3, 
					cutoff = 1000,
					tau=1000,
					p_entangle=1, 
					p_swap=1)

	# Run training loop
	agent.train(episodes=5_000,
				save_model=True,
				jitter=None,
				n_range=[3,3],
				savefig=False)
	