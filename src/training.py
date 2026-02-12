from base.agent import QRNAgent
import numpy
import torch
numpy.set_printoptions(legacy='1.25')


if __name__ == '__main__':

	agent = QRNAgent(buffer_size=10_000)
	dict_dir = 'assets/trained_models/Rd(12-2)l4u4e2450m80p50a99t50c15/Rd(12-2)l4u4e2450m80p50a99t50c15.pth'
	trained_dict = torch.load(dict_dir)
	agent.policy_net.load_state_dict(trained_dict)
	agent.target_net.load_state_dict(trained_dict)

	agent.train(episodes=1500, 
			 	max_steps=80, 
			 	savemodel=True,
			 	plot=True, 
			 	savefig=True,
			 	jitter=None,
				fine_tune=True, 
			 	n_range=[4, 4], 
			 	p_e=0.50, 
			 	p_s=0.99,
			 	tau=50,
			 	cutoff=15)
	