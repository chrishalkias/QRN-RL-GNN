import torch
from base.agent import QRNAgent


if __name__ == '__main__':

    dict_dir = "./assets/gnn_model.pth"
    trained_dict = torch.load(dict_dir)
    agent_val = QRNAgent()
    agent_val.policy_net.load_state_dict(trained_dict)

    agent_val.validate(n_episodes=20, 
                       max_steps=200, 
                       n_nodes=5, 
                       p_e=0.1, 
                       p_s=0.9,
                       tau = 1000,
                       cutoff=100, 
                       logging=True)
    
    # early termination seems to
    # improve the swapASAP varians
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	