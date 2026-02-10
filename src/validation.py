import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN.pth"

    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=100, 
                       max_steps=600, 
                       n_nodes=6,
                       p_e=0.05, 
                       p_s=0.95,
                       tau = 1000,
                       cutoff=50, 
                       logging=True)
    
    # early termination seems to
    # improve the swapASAP varians
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	