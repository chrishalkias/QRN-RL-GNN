import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    # dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN/d(9-2)l4u6e1000m100p85a85tNcN.pth"       # high P no cutoff model
    dict_dir = 'assets/trained_models/d(18-2)l4u6e1000m100p85a95t500c200/d(18-2)l4u6e1000m100p85a95t500c200.pth'
    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=1000, 
                       max_steps=150, 
                       n_nodes=8,
                       p_e=0.1, 
                       p_s=0.95,
                       tau = 50,
                       cutoff=20, 
                       logging=False,
                       plot_actions = True,
                       savefig=False)
    
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	