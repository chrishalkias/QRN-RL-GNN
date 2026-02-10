import torch
from base.agent import QRNAgent


if __name__ == '__main__':

    dict_dir = "./assets/gnn_model.pth"
    trained_dict = torch.load(dict_dir)
    agent_val = QRNAgent()
    agent_val.policy_net.load_state_dict(trained_dict)

    agent_val.validate(n_episodes=40, 
                       max_steps=2000, 
                       n_nodes=6, 
                       p_e=0.05, 
                       p_s=0.9,
                       tau = 1000,
                       cutoff=500)
	