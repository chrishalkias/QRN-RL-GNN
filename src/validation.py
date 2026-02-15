import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    # dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN/d(9-2)l4u6e1000m100p85a85tNcN.pth"       # high P no cutoff model
    dict_dir = 'assets/trained_models/d(15-2)l4u6e10000m50p85a95t20c20/d(15-2)l4u6e10000m50p85a95t20c20.pth'
    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=100, 
                       max_steps=200, 
                       n_nodes=6,
                       p_e=1, 
                       p_s=1,
                       tau = 10,
                       cutoff=20, 
                       logging=False,
                       plot_actions = True,
                       savefig=True)
    
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	