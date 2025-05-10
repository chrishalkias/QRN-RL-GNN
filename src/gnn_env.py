# -*- coding: utf-8 -*-
# src/environment.py

'''
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
import torch.nn.functional as F
from torch_geometric.nn import summary
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR, CyclicLR

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})

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
               weight_decay = 0,
               lr=0.001,
               gamma=0.9,
               epsilon=0.1,
               temperature=0,):
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
        preview                > Prints the parameters of the instance and the model architecture
        get_state_vector       > Returns the state of entanglements in the network
        out-to-onehot          > Outputs a T dependent one-hot encoding of the models output
        choose_action          > Choose a random, or the best, action
        update_environment     > Execute one of the actions
        reward                 > Computes the agents reward function
        train                  > Trains the agent
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
        self.line = '\n' + '-'*50 + '\n'
        self.network = RepeaterNetwork(n, directed, geometry, kappa, tau, p_entangle, p_swap)
        self.n = self.network.n
        self.lr=lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()
        self.weight_decay = weight_decay
        self.memory = []
        self.model = model
        self.temperature = temperature
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr, 
                                    weight_decay = self.weight_decay)
        
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode ='max',
            patience=10,
            factor=0.5
            )

    def preview(self):
        """Write the model params"""
        total_params = sum(p.numel() for p in self.model.parameters())
        summa = [self.line,
            f'Run information at {datetime.now()}', self.line,
            self.line, f'{" " * 10}Experiment parameters', self.line,
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
            f'optimizer    : {self.optimizer.__class__.__name__} \n',
            self.line, f'{" " * 10} Model architecture', self.line,
            f'{summary(self.model, self.network.tensorState())}\n',
            f'Total params: {total_params:,}\n',
            f'Model size: {sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2):.2f} MB',
            ]
        with open("logs/information.txt", "w") as file:
            [file.write(info) for info in summa];
        # x = torch.randn(100, 128)
        # edge_index = torch.randint(100, size=(2, 20))
        # print(summary(self.model, x, edge_index))

    def get_state_vector(self):
        return self.network.tensorState()
    

    def out_to_onehot(self, tensor, temperature=0) -> torch.tensor:
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


    def choose_action(self, action_matrix: list,  output: torch.tensor, use_trained_model = False, temperature: float = 0) -> list:
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
    

    def saveModel(self, filename="logs/model_checkpoints/GNN_model.pth"):
        """Saves the model"""
        torch.save(self.model.state_dict(), filename)

    def test(self, n_test, max_steps=100, kind='trained', plot=True):
        """
        Performs an evaluation on a repeater chain of specific length and returns the actions
        and plots of performance. Here the trained model is tested agains heuristics (random, alternating).

        Args:
            n_test        (int)  : The length of the repeater chain to perform the test on
            max_steps     (int)  : The maximum number of test iterations
            kind          (str)  : The chosen test method. 'trained' means using the trained model
            plot          (bool) : Create fidelity and reward plots

        Returns:
            *_test_output (.txt) : A text file with all the actions taken
            test_*        (.png) : Figures of the reward and fidelity plots
        """
        totalReward, rewardlist, totalrewardList = 0, [], []
        fidelity, fidelityList = 0,[]    
        self.network = RepeaterNetwork(n_test, p_entangle=self.network.p_entangle, p_swap=self.network.p_swap)
        self.n = self.network.n
        self.network.resetState()
        finalstep, timelist = 0, []
        state = self.get_state_vector()
        assert kind in ['trained', 'alternating', 'random'], f'Invalid option {kind}'
        os.makedirs('logs', exist_ok=True)
        with open("logs/information.txt", "a") as file:
            file.write(f'{self.line} Testing Information {self.line}')
            file.close()
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
                    action = self.choose_action(self.network.globalActions(), self.model(state), use_trained_model=True, temperature = self.temperature)
                elif kind == 'random':
                    waits = ['' for _ in range(self.n)]
                    entangles = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
                    swaps = [f'self.swapAT({i})' for i in range(self.n)]
                    action = [random.choice([e, s, w]) for e, s, w in zip(entangles, swaps, waits) if random.choice([e, s, w]) is not None]
                reward = self.update_environment(action)
                rewardlist.append(reward)
                state = self.get_state_vector()
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

        with open("logs/information.txt", "a") as file3:
            file3.write(f'Kind :{kind}, Links established : {total_links}, t_avg: {avg_time:.3f} it/link, t_std: {std_time:.3f} it/link')
            file3.close()
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
            plt.savefig(f'logs/plots/test_{kind}.png')
            plt.xlabel('Step')
            return finalstep
  

    def trainQ(self, episodes=10_000, plot=True, save_model=True):
        """
        This is the bread and butter of this class. Here the main loop of state action reward is iterated
        and the optimizer is moving forward on the model. This implements a very simple Q-learning procedure
        controlled by self.gamma, self.epsilon. Since the model output is a tensor the Q-value is calculated
        by the max() of it, i.e the node action with the highest confidence

        Args:
            episodes   (int)  : The number of trainig steps to perform
            plot       (bool) : Choice for plot outputs
            save_model (bool) : Choice for model checkpoints save

        Returns:
            model       (.pth) : The model checkpoints
            train_plots (.png) : The training plots
        """
        totalReward, rewardList, fidelity, fidelityList, entanglementDegree, entanglementlist, lossList = 0, [], 0, [], 0, [], []
        start_time = time.time()
        for _ in tqdm(range(episodes)):
            # set the MDP
            state = self.get_state_vector()
            output = self.model(state)
            action = self.choose_action(self.network.globalActions(), output, temperature = self.temperature)
            reward = self.update_environment(action)
            next_state = self.get_state_vector()
            
            # compute the target
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state))

            # compute reward and backpropagate
            q_value = torch.max(output)
            loss = self.criterion(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(reward)
            
            #append the metrics
            totalReward += reward
            rewardList.append(totalReward)
            fidelity += self.network.getLink((0,self.n-1),1)
            fidelityList.append(fidelity)
            linkList = [self.network.getLink(node,1) for node in self.network.matrix.keys()]
            lossList.append(loss.item())
            entanglementDegree = np.mean(linkList) /self.n
            entanglementlist.append(entanglementDegree)
            self.network.resetState() if reward == 1 else None

        train_time = time.time() - start_time
        with open("logs/information.txt", "a") as file:
            file.write(f'{self.line} {" " *10} Training information {self.line} Trained for {train_time:.3f} sec performing {episodes} steps.')
        self.saveModel() if save_model else None
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap}) over $10^{int(np.log10(episodes))}$ steps"
            ax1.plot(rewardList,ls='-', label='Cummulative reward')
            ax1.set(ylabel=f'Log reward')
            ax1.set_yscale("symlog")
            ax1.legend()

            fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
            ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
            ax2.set(ylabel=f'Fidelity of resulting links')
            ax2.set_xscale("log")
            ax2.legend()

            # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
            fig.suptitle(plot_title)
            plt.savefig('logs/plots/train_plots_reward.png')
            plt.xlabel('Episode')
            plt.legend()

            fig2, (ax12, ax22) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap}) over $10^{int(np.log10(episodes))}$ steps"
            ax12.plot(lossList,ls='-', label='Loss')
            ax12.set_xscale("log")
            ax12.set(ylabel=f'Loss')
            ax12.legend()

            fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
            ax22.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
            ax22.set_xscale("log")
            ax22.set(ylabel=f'Fidelity of resulting links')
            ax22.legend()

            # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
            fig.suptitle(plot_title)
            plt.savefig('logs/plots/train_plots_loss.png')
            plt.xlabel('Episode')
            plt.legend()


      