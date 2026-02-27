import torch
from base.agent import QRNAgent


if __name__ == '__main__':


    agent_val = QRNAgent()
    # dict_dir = "./assets/trained_models/d(9-2)l4u6e1000m100p85a85tNcN/d(9-2)l4u6e1000m100p85a85tNcN.pth"       # high P no cutoff model
    dict_dir = 'assets/trained_models/d(26-2)l6u6e3000m30p65a90t50c30/d(26-2)l6u6e3000m30p65a90t50c30.pth'
    agent_val.validate(dict_dir=dict_dir,
                       n_episodes=10, 
                       max_steps=100, 
                       n_nodes=7,
                       p_e=0.1, 
                       p_s=0.9,
                       tau = 50,
                       cutoff=40, 
                       logging=False,
                       plot_actions = False,
                       savefig=False)
    
    # the bigger max_steps the worse
    # their performance(steps). Interesting...
    # max_steps ~ cutoff it seems