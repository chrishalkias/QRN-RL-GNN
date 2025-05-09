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
import sys
import os
from datetime import datetime
from torchinfo import summary
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    """A function to preview the model architecture"""
    summary_txt = summary(self.model, input_size=(1, 2, self.network.n, self.network.n))
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
            f'---Model architecture--- \n {summary_txt}',]
    with open("logs/models/model_summary.txt", "w") as file:
      [file.write(info) for info in summa] ;
        
  def get_state_vector(self) -> torch.tensor:
    """ Container for the RepeaterNetwork output state for CNN"""
    dict_state = self.network.matrix
    keys = dict_state.keys() #extract i,j indices
    all_i = [k[0] for k in keys]
    all_j = [k[1] for k in keys]
    max_dim = max(all_i + all_j) #compute maximum dimension from all i,j
    matrix_shape = (max_dim + 1, max_dim + 1, 2)
    A = np.zeros(matrix_shape, dtype=np.float64)
    for (i, j), values in dict_state.items(): #fill the matrix
      A[i, j] = values
      A[j, i] = values
      A[i, i] = np.zeros(2)
    return torch.tensor(A).permute(2, 0, 1).unsqueeze(0).float()

  def decide(self, output: np.ndarray) -> np.ndarray:
    """Chosse only one of the 4 actions from the output"""
    actions = self.network.globalActions()
    n_squared = output.size(0)
    n = int(torch.sqrt(torch.tensor(n_squared, dtype=torch.float)))
    assert n == len(actions), f'net out dim {n} =/= action space dim {len(actions)}'
    # Reshape to (n, n, 4) and average along the second dimension
    reshaped = output.view(n, n, 4)
    averaged = torch.mean(reshaped, dim=1)
    result = torch.zeros_like(averaged)
    max_indices = torch.argmax(averaged, dim=1)
    result[torch.arange(averaged.size(0)), max_indices] = 1
    binary_mask = result.numpy() if isinstance(result, torch.Tensor) else result
    # Apply the mask and flatten
    verbalized = actions[binary_mask.astype(bool)]
    return verbalized, averaged[-1]

  def choose_action(self, state: np.ndarray, use_trained_model = False) -> torch.tensor:
    """Choose a random action with probability epsilon, otherwise choose the best action"""
    explore = random.uniform(0, 1) < self.epsilon

    if use_trained_model or (not explore):
      with torch.no_grad():
        model_output = self.model(state)
        return self.decide(model_output)[0]
    else:
      random_action_mask = [random.randint(0, 3) for _ in range(self.n)] #4 acts to choose from
      action_array = []
      for repeater in range(self.n):
        action_array.append(self.network.globalActions()[repeater][random_action_mask[repeater]])
      return action_array
    

  def update_environment(self, action_list: list) -> float:
    """Updates the environment and returns the reward"""
    new_dtype = '<U27'
    def insert_model(s):
      return s.replace('self.', 'self.network.')
    vectorized_insert = np.vectorize(insert_model, otypes=[new_dtype])
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
      for step in range(1, max_steps):
        if kind == 'alternating':
          if (step % 2) == 0:
            action = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
          elif (step % 2) == 1:
            action = [f'self.swapAT({i})' for i in range(self.n)]
        elif kind == 'trained':
          action = self.choose_action(state, use_trained_model=True)
        elif kind == 'random':
          entangles = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
          swaps = [f'self.swapAT({i})' for i in range(self.n)]
          action = random.choice([entangles, swaps])
        reward = self.update_environment(action)
        state = self.get_state_vector()
        totalReward += reward
        rewardList.append(totalReward)
        fidelity += self.network.getLink((0,self.n-1),1)
        fidelityList.append(fidelity)
        file.write(f"\n Action: {[act[5:] for act in action]},Reward: {reward}")
        if reward == 1:
          file.write(f"\n\n--Linked in {step - finalstep} steps for {kind} \n")
          timelist.append(step-finalstep)
          finalstep = step
          self.network.endToEndCheck()
          self.network.resetState()
        file.write('\n ---Max iterations reached \n') if step == max_steps-1 else None
      file.write(f'---Transfer times : {timelist} \n')
      file.write (f'---Total links established : {len(timelist)}')
    if plot:
      fig, (ax1, ax2) = plt.subplots(2, 1)
      plot_title = f"Metrics for {kind} for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap}) over {max_steps} steps"
      # ax1.axline((0,1),slope=0, ls='--')
      ax1.plot(rewardList, 'tab:orange', ls='-', label='Cummulative reward')
      ax1.set(ylabel=f'Log reward')
      ax1.set_yscale("symlog")
      ax1.legend()
      ax2.plot(fidelityList, 'tab:green', ls='-', label='Total Fidelity')
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
      action = self.choose_action(state)
      reward = self.update_environment(action)
      next_state = self.get_state_vector()

      with torch.no_grad():
        target = reward + self.gamma * torch.max(self.model(next_state))
      q_value = torch.mean(self.decide(self.model(state))[1])
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


    self.saveModel() if save_model else None
    if plot:
      fig, (ax1, ax2) = plt.subplots(2, 1)
      plot_title = f"Training metrics for $(n, p_E, p_S)$= ({self.n}, {self.network.p_entangle}, {self.network.p_swap} over {episodes} episodes)"


      # ax1.axline((0,1),slope=0, ls='--')
      # ax1.plot(lossList, ls='-', label='Loss')
      ax1.plot(rewardList,ls='-', label='Cummulative reward')
      ax1.set(ylabel=f'Log reward and loss')
      ax1.set_yscale("symlog")
      ax1.legend()
      ax2.plot(fidelityList, 'tab:green', ls='-', label='Total Fidelity')
      ax2.set(ylabel=f'Fidelity of resulting link')
      ax2.set_xscale("log")
      ax2.legend()
      # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
      fig.suptitle(plot_title)
      plt.savefig('logs/plots/train_plots.png')
      plt.xlabel('Episode')
      plt.legend()

      
  def tREINFORCE(self):
    """Performs REINFORCE policy gradient update after episode ends"""
    returns = []
    G = 0
    
    # Calculate discounted returns (backwards)
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize
    
    policy_loss = []
    for log_prob, G in zip(self.log_probs, returns):
        policy_loss.append(-log_prob * G)  # negative sign for gradient ascent
    
    # Update model parameters (assuming self.model is inside choose_action)
    self.optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    self.optimizer.step()
    
    # Clear episode data
    self.log_probs = []
    self.rewards = []
    
  def run_episode(self):
    """Runs one episode and updates policy"""
    state = self.get_state_vector()
    done = False
    
    while not done:
        action, log_prob = self.choose_action(state)  # assume modified to return log_prob
        reward, done = self.update_environment(action)
        
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        state = self.get_state_vector()
    
    self.tREINFORCE()  # Update policy after episode