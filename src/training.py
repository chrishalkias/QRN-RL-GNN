from base.agent import QRNAgent
import numpy
import torch
numpy.set_printoptions(legacy='1.25')

args = {
	'episodes': 1_000,
	'max_steps': 50,
	'savemodel': True,
	'plot': True,
	'jitter': 0,
	'fine_tune': False,
	'n_range': [4,6],
	'p_e': 0.85,
	'p_s': 0.95,
	'tau': 200,
	'cutoff': 100,
	'use_wandb': False, 
	}

if __name__ == '__main__':

	agent = QRNAgent(buffer_size=10_000)

	# # # Uncomment for fine-tuning
	# dict_dir = 'assets/trained_models/d(13-2)l4u4e25000m30p70a99t15c8/d(13-2)l4u4e25000m30p70a99t15c8.pth'
	# trained_dict = torch.load(dict_dir)
	# agent.policy_net.load_state_dict(trained_dict)
	# agent.target_net.load_state_dict(trained_dict)

	agent.train(**args)
	