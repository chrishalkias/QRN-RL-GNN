from stats import test_stats, n_scaling_test
from base.agent import AgentGNN
import torch

if __name__ == '__main__':
    TESTING = True
    SCALING = False
	
    #Initialize agent
    agent = AgentGNN()

    #Load state dict from trained model
    dict_dir = "/Users/chrischalkias/GitHub/QRN-RL-GNN/assets/gnn_model.pth"
    model_state_dict = torch.load(dict_dir)
    model = agent.target_model
    model.load_state_dict(model_state_dict)

	# Gather validation statistics
    test_stats(agent=agent, 
            experiments=1, 
            n_test = 6, 
            p_entangle=0.1, 
            p_swap=0.8, 
            cutoff = 1000,
            tau=1000, 
            max_steps=2_000,
            savefig=False) if TESTING else None

    # Observe the relative performance of tha agent as N varies
    n_scaling_test(experiments=10, 
                agent=agent, 
                N_range=range(5,10), 
                p_e=0.1, 
                p_s=0.8, 
                tau=100) if SCALING else None
	