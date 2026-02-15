from base.agent import QRNAgent
import numpy
import torch
numpy.set_printoptions(legacy='1.25')


if __name__ == '__main__':

	agent = QRNAgent(buffer_size=10_000)

	# # # Uncomment for fine-tuning
	# dict_dir = 'assets/trained_models/d(13-2)l4u4e25000m30p70a99t15c8/d(13-2)l4u4e25000m30p70a99t15c8.pth'
	# trained_dict = torch.load(dict_dir)
	# agent.policy_net.load_state_dict(trained_dict)
	# agent.target_net.load_state_dict(trained_dict)

	agent.train(episodes=7_000, 
			 	max_steps=60, 
			 	savemodel=True,
			 	plot=True, 
			 	savefig=True,
			 	jitter=200,
				fine_tune=False, 
			 	n_range=[4, 6], 
			 	p_e=0.70, 
			 	p_s=0.98,
			 	tau=150,
			 	cutoff=30, 
				use_wandb=True)
	