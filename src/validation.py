import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    # dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN/d(9-2)l4u6e1000m100p85a85tNcN.pth"       # high P no cutoff model
    dict_dir = 'assets/trained_models/Rd(12-2)l4u4e2450m80p50a99t50c15/Rd(12-2)l4u4e2450m80p50a99t50c15.pth'
    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=100, 
                       max_steps=200, 
                       n_nodes=4,
                       p_e=0.85, 
                       p_s=0.95,
                       tau = 50,
                       cutoff=30, 
                       logging=False,
                       plot_actions = True,
                       savefig=True)
    
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems
	