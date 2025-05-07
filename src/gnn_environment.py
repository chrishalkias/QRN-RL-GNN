# -*- coding: utf-8 -*-
# src/environment.py

'''
Created Wed 02 Apr 2025
The Agent class to run the RL model on the repeater networks.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import statistics
import sys
import os
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
from torch_geometric.nn import summary
from tqdm import tqdm
import torch.nn.functional as F

from repeaters import RepeaterNetwork

class Environment():
    def __init__(self,
               model, 
               n=4,
               directed = False,
               geometry = 'chain',
               kappa = 1,
               tau = 1_000,
               p_entangle = 1,
               p_swap = 1,
               lr=0.001,
               gamma=0.9,
               epsilon=0.1,):
      """

                                    
           ██████  ██████  ███    ██     ███████ ███    ██ ██    ██ 
          ██    ██ ██   ██ ████   ██     ██      ████   ██ ██    ██ 
          ██    ██ ██████  ██ ██  ██     █████   ██ ██  ██ ██    ██ 
          ██ ▄▄ ██ ██   ██ ██  ██ ██     ██      ██  ██ ██  ██  ██  
           ██████  ██   ██ ██   ████     ███████ ██   ████   ████   
              ▀▀                                                                                            
                                                                                               
                                             
    Description:
      This class implements the Graph description of the repeater network and uses
      it to train a deep Q learning algorithm using the DQN model built with
      PyTorch.

    Methods:
      actions()             > Creates a dict() with all the possible actions
      actionCount()         > Returns the number of possible actions
      get_state_vector()    > Returns the state of entanglements in the network
      choose_action()       > Choose a random, or the best, action
      update_environment()  > Execute one of the actions
      reward()              > Computes the agents reward function
      train()               > Trains the agent
      saveModel()           > Saves the model to file
      test()                > Evaluate the model

    Attributes:
      lr          (float)   > Learning rate
      gamma       (float)   > Discount factor
      epsilon     (float)   > Exploration rate
      criterion   (nn.Loss) > Computes the loss function
      model       (Tensor)  > Calls the DQN model
      optimizer   (Tensor)  > The optimizer for the model
      memory      (list)    > _not implemented yet_
      """
      super().__init__()
      self.network = RepeaterNetwork(n, directed, geometry, kappa, tau, p_entangle, p_swap)
      self.n = self.network.n
      self.lr=lr
      self.gamma = gamma
      self.epsilon = epsilon
      self.criterion = nn.MSELoss()
      self.memory = []
      self.model = model
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def preview(self):
        """Write the model params"""
        summary_txt = 'No summary for gnn'
        summa = [f'---Experiment parameters at {datetime.now()}---', '\n' + '-'*50 + '\n',
            f'Environment  : {self.network.__class__.__name__} \n',
            f'n            : {self.network.n} \n',
            f'directed     : {self.network.directed} \n',
            f'geometry     : {self.network.geometry} \n',
            f'kappa        : {self.network.kappa} \n',
            f'tau          : {self.network.tau} \n',
            f'p_entangle   : {self.network.p_entangle} \n',
            f'p_swap       : {self.network.p_swap} \n',
            f'lr           : {self.lr} \n',
            f'gamma        : {self.gamma} \n',
            f'epsilon      : {self.epsilon} \n',
            f'criterion    : {self.criterion.__class__.__name__}\n',
            f'optimizer    : {self.optimizer.__class__.__name__} \n'
            f'---Model architecture--- \n {summary_txt}',
            ]
        with open("logs/models/model_summary.txt", "w") as file:
            [file.write(info) for info in summa] ;
        # x = torch.randn(100, 128)
        # edge_index = torch.randint(100, size=(2, 20))
        # print(summary(self.model, x, edge_index))

    def get_state_vector(self):
        return self.network.tensorState()
    

    def out_to_onehot(self, tensor, temperature=1) -> torch.tensor:
        """
        Converts tensor to one-hot encoding with temperature-scaled probabilities.
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


    def choose_action(self, action_matrix: list,  output: torch.tensor, use_trained_model = False) -> list:
        """Choose a random action with probability epsilon, otherwise choose the best action"""
        explore = random.uniform(0, 1) < self.epsilon
        from_model = use_trained_model or (not explore)

        if from_model:
            with torch.no_grad():
                action_array = np.array(action_matrix)
                # Get one-hot mask
                one_hot_mask = self.out_to_onehot(output).numpy()
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
    

    def update_environment(self, action_list: list) -> float:
        """Updates the environment and returns the reward"""
        def insert_model(s):
            return s.replace('self.', 'self.network.')
        vectorized_insert = np.vectorize(insert_model, otypes=['<U27'])
        actions = vectorized_insert(action_list)
        for action in actions:
            exec(action)
        return self.reward()

    def reward(self) -> float:
        """Computes the agents reward"""
        self.network.endToEndCheck()
        return 1 if self.network.global_state else -.1

    def saveModel(self, filename="logs/models/model.pth"):
        """Saves the model"""
        torch.save(self.model.state_dict(), filename)

    def test(self, n_test, max_steps=100, kind='trained', plot=True):
        """Evaluate the model"""
        totalReward, rewardList = 0, []
        fidelity, fidelityList = 0,[]    
        self.network = RepeaterNetwork(n_test, p_entangle=self.network.p_entangle, p_swap=self.network.p_swap)
        self.n = self.network.n
        self.network.resetState()
        finalstep, timelist = 0, []
        state = self.get_state_vector()
        assert kind in ['trained', 'alternating', 'random'], f'Invalid option {kind}'
        os.makedirs('logs', exist_ok=True)
        with open(f'./logs/textfiles/{kind}_test_output.txt', 'w') as file:
            file.write(f'Action reward log for {kind} at {datetime.now()}\n\n')
            print(f'Testing {kind}')
            for step in tqdm(range(1, max_steps)):
                if kind == 'alternating':
                    if (step % 2) == 0:
                        action = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
                    elif (step % 2) == 1:
                        action = [f'self.swapAT({i})' for i in range(self.n)]
                elif kind == 'trained':
                    action = self.choose_action(self.network.globalActions(), self.model(state), use_trained_model=True)
                elif kind == 'random':
                    waits = ['' for _ in range(self.n)]
                    entangles = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
                    swaps = [f'self.swapAT({i})' for i in range(self.n)]
                    action = [random.choice([e, s, w]) for e, s, w in zip(entangles, swaps, waits) if random.choice([e, s, w]) is not None]
                reward = self.update_environment(action)
                state = self.get_state_vector()
                totalReward += reward
                rewardList.append(totalReward)
                fidelity += self.network.getLink((0,self.n-1),1)
                fidelityList.append(fidelity)
                fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
                file.write(f"\n Action: {[act[5:] for act in action]},Reward: {reward}")
                if reward == 1:
                    file.write(f"\n\n--Linked in {step - finalstep} steps for {kind} \n")
                    timelist.append(step-finalstep)
                    finalstep = step
                    self.network.endToEndCheck()
                    self.network.resetState()
                    file.write('\n ---Max iterations reached \n') if step == max_steps-1 else None
            file.close()
        total_links = len(timelist)
        avg_time = sum(timelist) / len(timelist) if timelist else np.inf
        std_time = statistics.stdev(timelist) if timelist else np.inf
        line0 = '-' * 50
        line1 = (f'\n >>> Total links established : {total_links}\n')
        line2 = (f'\n >>> Avg transfer time       : {avg_time:.3f} it \n')
        line3 = (f'\n >>>Typical time deviation   : {std_time:.3f} it\n')
        for line in (line0, line3, line2, line1, line0):
            with open(f'./logs/textfiles/{kind}_test_output.txt', 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_title = f"Metrics for {kind} for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap}) over {max_steps} steps"
            # ax1.axline((0,1),slope=0, ls='--')
            ax1.plot(rewardList, 'tab:orange', ls='-', label='Cummulative reward')
            ax1.set(ylabel=f'Log reward')
            ax1.set_yscale("symlog")
            ax1.legend()
            ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity per step')
            ax2.legend()
            ax2.set(ylabel=f'Fidelity of resulting link')
            ax2.set_xscale("log")
            fig.suptitle(plot_title)
            plt.savefig(f'logs/plots/test_{kind}.png')
            plt.xlabel('Step')
            return finalstep
  

    def trainQ(self, episodes=10_000, plot=True, save_model=True):
        """Trains the agent"""
        totalReward, rewardList = 0, []
        fidelity, fidelityList = 0,[]
        lossList = []
        entanglementDegree, entanglementlist = 0,[]

        for _ in tqdm(range(episodes)):
            state = self.get_state_vector()
            output = self.model(state)
            action = self.choose_action(self.network.globalActions(), output)
            reward = self.update_environment(action)
            next_state = self.get_state_vector()

            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state))
            q_value = torch.max(output)
            # q_value = torch.mean(self.model(state)) # CHANGE THIS

            loss = self.criterion(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            totalReward += reward
            rewardList.append(totalReward)
            # some extra metrics
            fidelity += self.network.getLink((0,self.n-1),1)
            fidelityList.append(fidelity)
            linkList = [self.network.getLink(node,1) for node in self.network.matrix.keys()]
            lossList.append(loss.item())
            entanglementDegree = np.mean(linkList) /self.n
            entanglementlist.append(entanglementDegree)
            self.network.resetState() if reward == 1 else None

        fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
        self.saveModel() if save_model else None
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap} over {episodes} steps)"


        # ax1.axline((0,1),slope=0, ls='--')
        # ax1.plot(lossList, ls='-', label='Loss')
        ax1.plot(rewardList,ls='-', label='Cummulative reward')
        ax1.set(ylabel=f'Log reward and loss')
        ax1.set_yscale("symlog")
        ax1.legend()
        ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
        ax2.set(ylabel=f'Fidelity of resulting links')
        ax2.set_xscale("log")
        ax2.legend()
        # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
        fig.suptitle(plot_title)
        plt.savefig('logs/plots/train_plots.png')
        plt.xlabel('Episode')
        plt.legend()

      