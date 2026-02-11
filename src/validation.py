import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    # dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN.pth"       # high P no cutoff model
     # dict_dir = "./assets/trained_models/d(11-2)l4u6e1000m200p35a95t100c40.pth   # mid P cutoff model not well trained
    dict_dir = "./assets/trained_models/d(10-2)l4u6e5000m200p.5a95t100c50.pth"     # low p cutoff model no edge features

    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=1, 
                       max_steps=500, 
                       n_nodes=6,
                       p_e=0.1, 
                       p_s=0.9,
                       tau = 100,
                       cutoff=50, 
                       logging=False,
                       savefig=False)
    
    # early termination seems to
    # improve the swapASAP varians
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	