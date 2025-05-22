# -*- coding: utf-8 -*-
# src/environment.py

'''
             ██████  ██████  ███    ██     ███████ ███    ██ ██    ██ 
            ██    ██ ██   ██ ████   ██     ██      ████   ██ ██    ██ 
            ██    ██ ██████  ██ ██  ██     █████   ██ ██  ██ ██    ██ 
            ██ ▄▄ ██ ██   ██ ██  ██ ██     ██      ██  ██ ██  ██  ██  
             ██████  ██   ██ ██   ████     ███████ ██   ████   ████   
                ▀▀                                                         
                                                   
                            Created Wed 02 Apr 2025
The Agent class to run the RL model on the repeater networks for the case of GNN models.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import statistics
import sys
import os
import time
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch.nn.functional as F
from torch_geometric.nn import summary
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR, CyclicLR

from repeaters import RepeaterNetwork

class Environment():
    def __init__(self,
               model: object, 
               n: int=4,
               directed: bool = False,
               geometry: str = 'chain',
               kappa: float = 1,
               tau: float = 1_000,
               p_entangle: float = 1,
               p_swap: float = 1,
               weight_decay: float = 0,
               lr: float = 1e-3,
               gamma: float = 0.9,
               epsilon: float =0.1,
               temperature: float = 0,):
        """                   
    Description:
        This class implements the Graph description of the repeater network and uses
        it to train a deep Q learning algorithm using the DQN model built with
        PyTorch.

    Methods:
        preview                > Prints the parameters of the instance and the model architecture
        get_state_vector       > Returns the state of entanglements in the network
        out-to-onehot          > Outputs a T dependent one-hot encoding of the models output
        choose_action          > Choose a random, or the best, action
        update_environment     > Execute one of the actions
        reward                 > Computes the agents reward function
        saveModel              > Saves the model to file
        test                   > Evaluate the model

    Attributes:
        lr           (float)   > Learning rate
        gamma        (float)   > Discount factor
        epsilon      (float)   > Exploration rate
        criterion    (nn.Loss) > Computes the loss function
        weight_decay (bool)
        model        (Tensor)  > Calls the DQN model
        optimizer    (obj)     > The optimizer for the model
        scheduler    (obj)     > The lr scheduler
        temperature  (float)   > The temperature to be used on one-hot
        """
        super().__init__()
        self.network = RepeaterNetwork(n, directed, geometry, kappa, tau, p_entangle, p_swap)
        self.n = self.network.n
        self.lr=lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()# nn.MSELoss()
        self.weight_decay = weight_decay
        self.memory = []
        self.model = model
        self.temperature = temperature
        self.test_dict = {}
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr, 
                                    weight_decay = self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode ='max',
            patience=10,
            factor=0.5
            )
        # self.scheduler = CyclicLR(
        #         self.optimizer,
        #         base_lr=1e-5,       # Lower bound
        #         max_lr=3e-3,        # Upper bound (start with your current LR)
        #         step_size_up=2000,  # Steps per half-cycle
        #         mode='triangular'
        #     )
        # warmup_steps = 10000
        # self.scheduler = LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lambda step: min(1.0, step / warmup_steps)
        #     )

    def preview(self) -> None:
        """Write the model params in file"""
        total_params = sum(p.numel() for p in self.model.parameters())
        attr_exclude = ['model', 'optimizer', 'memory', 'test_dict']
        infos = [f"{attr} = {value} \n" for attr, value in self.__dict__.items() if attr not in attr_exclude]
        infos.append(f'\n Model breakdown \n')
        infos.append(f'{summary(self.model, self.network.tensorState())}\n')
        infos.append(f'Total params: {total_params:,}\n')
        with open("logs/information.txt", "w") as file:
            file.write(f'Run at {datetime.now()} \n'), [file.write(info) for info in infos];


    def out_to_onehot(self, tensor: torch.tensor, temperature: float=0) -> torch.tensor:
        """
        Converts tensor to one-hot encoding with temperature-scaled probabilities.

        Args:
            tensor (torch.tensor)  : The input tensor to be one hotted
        Returns:
            one_hot (torch.tensor) : One hot encoded tensor

        """
        # Apply temperature scaling to squared values
        scaled = tensor.pow(2) / max(temperature, 1e-8)
        # Softmax to get probabilities
        probs = F.softmax(scaled, dim=1)
        # Sample one position per row
        choices = torch.multinomial(probs, num_samples=1).squeeze(1)
        # Convert to one-hot
        one_hot = torch.zeros_like(tensor)
        one_hot.scatter_(1, choices.unsqueeze(1), 1)
        return one_hot

    def binary_tensor_to_int(self, onehot:torch.tensor) -> int:
        """
        Converts a binary tensor of shape (4, N) to a unique integer.
        The left-most element is treated as the most significant bit.
        
        Args:
            tensor: torch.Tensor of shape (1, N) containing 0s and 1s
            
        Returns:
            int: Unique integer representation
        """
        onehot = onehot.view(-1)
        N = onehot.shape[0]
        # Create powers of 2 in descending order (MSB to LSB)
        powers = 2 ** torch.arange(N-1, -1, -1)
        return int((onehot * powers).sum().item())


    def choose_action(self, 
                      action_matrix: list,  
                      output: torch.tensor, 
                      use_trained_model = False, 
                      temperature: float = 0) -> list:
        """
        Choose a random action with probability epsilon, otherwise choose the best action

        Args:
            action_matrix     (list)   : A list of all the possible actions taken from self.network
            output            (tensor) : The models output (batch size, N, 4)
            temperature       (float)  : The temperature controling the stochasticity of the one-hot
            use_trained_model (bool)   : Whether to use the model or perform a random action

        Returns:
            actions           (list)   : A (4, n) list of strings of verbalized actions to be executed
        """
        explore = random.uniform(0, 1) < self.epsilon
        from_model = use_trained_model or (not explore)

        if from_model:
            with torch.no_grad():
                action_array = np.array(action_matrix)
                # Get one-hot mask
                one_hot_mask = self.out_to_onehot(output, temperature).numpy()
                # Select actions where mask is 1
                selected_actions = []
            for i in range(one_hot_mask.shape[0]):
                # Get the index of the 1 in this row
                action_idx = np.argmax(one_hot_mask[i])
                selected_actions.append(action_array[i, action_idx])
            return selected_actions
        else:
            random_action_mask = [random.randint(0, 3) for _ in range(self.n)] #4 acts to choose from
            action_array = []
            for repeater in range(self.n):
                action_array.append(self.network.globalActions()[repeater][random_action_mask[repeater]])
            return action_array
    

    def update_environment(self, chosen_list: list) -> float:
        """
        Updates the environment via a spcified (4,n) list of string actions and returns the reward.
        the list should be of the form [a,b,c,d] where the letters are actions of the form 'self.entangle(1,2)'.

        Args:
            chosen_list (list) : A list of actions ouputed by the choose_action method

        Returns:
            reward      (float) : Executes the actions on self.network and returns the reward
        """
        def insert_model(s):
            return s.replace('self.', 'self.network.')
        vectorized_insert = np.vectorize(insert_model, otypes=['<U27'])
        actions = vectorized_insert(chosen_list)
        for action in actions:
            exec(action)
        return self.reward()

    def reward(self) -> float:
        """Computes the agents reward for the current state of self.network.model"""
        bonus_reward = 0 # some function f(d, e; n)
        for (i,j), (adjecency, entanglement) in self.network.matrix.items():
            distance = j-i #chain only
            bonus_reward +=  entanglement*distance/(10*self.network.n**2) if entanglement else 0

        return 1 if self.network.endToEndCheck() else -0.1 + bonus_reward
    

    def test(self, n_test, max_steps=100, plot=True) -> int:
        """
        Performs an evaluation on a repeater chain of specific length and returns the actions
        and plots of performance. Here the trained model is tested agains heuristics (random, alternating).

        Args:
            n_test        (int)  : The length of the repeater chain to perform the test on
            max_steps     (int)  : The maximum number of test iterations
            plot          (bool) : Create fidelity and reward plots

        Returns:
            *_test_output (.txt) : A text file with all the actions taken
            test_*        (.png) : Figures of the reward and fidelity plots
        """
        self.network = RepeaterNetwork(n_test, 
                                       p_entangle=self.network.p_entangle, 
                                       p_swap=self.network.p_swap)
        self.n = self.network.n
        self.network.resetState() #start with clean slate
        state = self.network.tensorState()


        def trained_action():
            """Return the models prediction for an action"""
            return self.choose_action(self.network.globalActions(), 
                                      self.model(state), use_trained_model=True, 
                                      temperature = self.temperature)

        def random_action():
            """Perform a random action at each node"""
            waits = ['' for _ in range(self.n)]
            entangles = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
            swaps = [f'self.swapAT({i})' if (i != 0) and (i !=self.n-1) else '' for i in range(self.n)] # dont swap ad end nodes
            return [random.choice([e, s, w]) for e, s, w in zip(entangles, swaps, waits) if random.choice([e, s, w]) is not None]

        def alternating_action():
            """At even timestep entangle all and at odd swap all"""
            if (step % 2) == 0:
                return [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
            elif (step % 2) == 1:
                return [f'self.swapAT({i})' if (i != 0) and (i !=self.n-1) else '' for i in range(self.n)]
            
        def swap_asap():
            """Perform the swap asap"""
            net = self.network
            actions = []

            for i in range(net.n):
                rightlink = net.getLink(edge = (i,i+1), linkType=1) if i != net.n-1 else -1
                leftlink = net.getLink(edge = (i-1,i), linkType=1) if i != 0 else -1

                if leftlink > 0 and rightlink > 0 and i!=0 and i!=net.n:
                    actions.append(f'self.swapAT({i})')
                elif leftlink == 0 and leftlink != -1:
                    actions.append(f'self.entangle(edge={(i-1,i)})')
                elif rightlink == 0 and rightlink != -1:
                    actions.append(f'self.entangle(edge={(i,i+1)})')

            return actions

            
        os.makedirs('logs', exist_ok=True)
        for kind in ['trained', 'random', 'swapASAP', 'alternating']:
            totalReward, rewardlist, totalrewardList = 0, [], []
            fidelity, fidelityList = 0,[]    
            finalstep, timelist = 0, []
            with open(f'./logs/textfiles/{kind}_test_output.txt', 'w') as file:
                file.write(f'Action reward log for {kind} at {datetime.now()}\n\n')
                print(f'Testing {kind}')
                for step in tqdm(range(1, max_steps)):
                    if kind == 'swapASAP':
                        action = swap_asap()
                    if kind == 'alternating':
                        action = alternating_action()
                    elif kind == 'trained':
                        action = trained_action()
                    elif kind == 'random':
                        action = random_action()

                    reward = self.update_environment(action)
                    rewardlist.append(reward)
                    state = self.network.tensorState()
                    totalReward += reward
                    totalrewardList.append(totalReward)
                    fidelity += self.network.getLink((0,self.n-1),1)
                    fidelityList.append(fidelity)
                    fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
                    
                    file.write(f"\n Action: {[act[5:] for act in action]},Reward: {reward}")
                    file.write(f'\n State: {[ent for (adj, ent) in self.network.matrix.values()]}\n\n')

                    if self.network.endToEndCheck():
                        file.write(f"\n\n--Linked in {step - finalstep} steps for {kind} \n")
                        timelist.append(step-finalstep)
                        finalstep = step
                        self.network.endToEndCheck()
                        self.network.resetState()
                        file.write('\n ---Max iterations reached \n') if step == max_steps-1 else None
                file.close()

            total_links = len(timelist)
            avg_time = sum(timelist) / len(timelist) if timelist else np.inf
            std_time = statistics.stdev(timelist) if len(timelist) >= 2 else np.inf
            line0 = '-' * 50
            line1 = (f'\n >>> Total links established : {total_links}\n')
            line2 = (f'\n >>> Avg transfer time       : {avg_time:.3f} it \n')
            line3 = (f'\n >>>Typical time deviation   : {std_time:.3f} it\n')

            for line in (line0, line3, line2, line1, line0):
                with open(f'./logs/textfiles/{kind}_test_output.txt', 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(line.rstrip('\r\n') + '\n' + content)
                    f.close

            with open("logs/information.txt", "a") as f:
                f.write(f'{kind}, L={total_links}, t_avg={avg_time:.1f}, t_std={std_time:.1f}\n')
                f.close()
                
            # Gather test statistics
            (mean_reward, std_reward) = (statistics.mean(rewardlist), statistics.stdev(rewardlist))
            (mean_fidelity, std_fidelity) = (statistics.mean(fidelity_per_step), statistics.stdev(fidelity_per_step))
            self.test_dict = self.test_dict | {kind: [mean_reward, std_reward, mean_fidelity, std_fidelity]}
        

            if plot:
                fig, (ax1, ax2) = plt.subplots(2, 1)
                plot_title = f"Metrics for {kind} for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap}) over $10^{int(np.log10(max_steps))}$ steps"
                # ax1.axline((0,1),slope=0, ls='--')
                ax1.plot(totalrewardList, 'tab:orange', ls='-', label='Reward per step')
                ax1.set(ylabel=f'Log reward')
                # ax1.set_yscale("symlog")
                ax1.legend()
                ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity per step')
                ax2.legend()
                ax2.set(ylabel=f'Fidelity of resulting link')
                # ax2.set_xscale("log")
                fig.suptitle(plot_title)
                label = f'logs/plots/test_{kind}.png'
                plt.savefig(label)
                plt.xlabel('Step')

        #save the test data
        with open("logs/test_metrics.json", 'a') as f:
            json.dump(self.test_dict, f, indent=4,)