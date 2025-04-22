# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from repeaters import RepeaterNetwork
from models import CNN
from typing import List, Tuple

#@title AgentCNN class
class AgentDQN():
  def __init__(self, n=4,
               directed = False,
               geometry = 'chain',
               kappa = 1,
               tau = 1_000,
               p_entangle = 1,
               p_swap = 1,
               lr=0.001,
               gamma=0.9,
               epsilon=0.1):
    """        _                           _
              | |                         (_)
              | |     ___  __ _ _ __ _ __  _ _ __   __ _
              | |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
              | |___|  __/ (_| | |  | | | | | | | | (_| |
              |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                     /\                   | |        _/ |
                    /  \   __ _  ___ _ __ | |_      |___/
                   / /\ \ / _` |/ _ \ '_ \| __|
                  / ____ \ (_| |  __/ | | | |_
                 /_/    \_\__, |\___|_| |_|\__|
                           __/ |
                          |___/
    ----------------------------------------------------------------------------
    Description:
    ----------------------------------------------------------------------------
    This class implements the Graph description of the repeater network and uses
    it to train a deep Q learning algorithm using the DQN model built with
    PyTorch.

    ----------------------------------------------------------------------------
    Methods:
    ----------------------------------------------------------------------------
    actions()             > Creates a dict() with all the possible actions
    actionCount()         > Returns the number of possible actions
    get_state_vector()    > Returns the state of entanglements in the network
    choose_action()       > Choose a random, or the best, action
    update_environment()  > Execute one of the actions
    reward()              > Computes the agents reward function
    train()               > Trains the agent
    saveModel()           > Saves the model to file
    test()                > Evaluate the model

    ----------------------------------------------------------------------------
    Attributes:
    ----------------------------------------------------------------------------
    lr          (float)   > Learning rate
    gamma       (float)   > Discount factor
    epsilon     (float)   > Exploration rate
    criterion   (nn.Loss) > Computes the loss function
    model       (Tensor)  > Calls the DQN model
    optimizer   (Tensor)  > The optimizer for the model
    memory      (list)    > _not implemented yet_

    ----------------------------------------------------------------------------
    """
    super().__init__()
    self.network = RepeaterNetwork(n, directed, geometry, kappa, tau, p_entangle, p_swap)
    self.n = self.network.n
    self.lr=lr
    self.gamma = gamma
    self.epsilon = epsilon
    self.criterion = nn.MSELoss()
    self.memory = []
    self.model = CNN(
                 convolutions = 4,
                 pooling_dim = 16,
                 embeding_dim = 32,
                 hidden_dim = 32,
                 unembeding_dim = 8,
                 )
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


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
    if isinstance(result, torch.Tensor):
          binary_mask = result.numpy()
    else:
        binary_mask = result
    # Apply the mask and flatten
    verbalized = actions[binary_mask.astype(bool)]
    return verbalized, averaged

  def choose_action(self, state: np.ndarray, use_trained_model = False) -> torch.tensor:
    """
    Choose a random action with probability epsilon, otherwise choose the best action
    """
    condition_to_choose = (not use_trained_model) and (random.uniform(0, 1) < self.epsilon)
    if condition_to_choose:
      random_action_mask = [random.randint(0, 3) for _ in range(self.n)] #4 acts to choose from
      action_array = []
      for repeater in range(self.n):
        action_array.append(self.network.globalActions()[repeater][random_action_mask[repeater]])
      return action_array

    with torch.no_grad():
      q_values = self.model(state)
    return self.decide(q_values)[0]

  def update_environment(self, action_list: List) -> float:
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

  def train(self, episodes=10_000, plot=True, save_model=True):
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
      q_value = torch.mean(self.model(state)) # CHANGE THIS

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
      # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')

      fig.suptitle(plot_title)
      plt.savefig('Train.png')
      plt.xlabel('Episode')
      plt.legend()

  def saveModel(self, filename="cnn_model.pth"):
    """Saves the model"""
    torch.save(self.model.state_dict(), filename)
    print(f"Model saved to {filename}")

  def test(self, n, max_steps=100, kind='trained', plot=True):
    """Evaluate the model"""
    totalReward, rewardList = 0, []
    fidelity, fidelityList = 0,[]

    with open('test_output.txt', 'w') as file:
      self.network = RepeaterNetwork(n, p_entangle=0.85, p_swap=0.85)
      self.n = self.network.n
      self.network.resetState()
      finalstep = None
      state = self.get_state_vector()
      assert kind in ['trained', 'swap_asap', 'random'], f'Invalid option {kind}'
      for step in range(max_steps):

        if kind == 'swap_asap':
          if (step % 2) == 0:
            action = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
          elif (step % 2) == 1:
            action = [f'self.swapAT({i})' for i in range(self.n)]

        elif kind == 'trained':
          action = self.choose_action(state, use_trained_model=False)

        elif kind == 'random':
          entangles = [f'self.entangle({(i,i+1)})' for i in range(self.n-1)]
          swaps = [f'self.swapAT({i})' for i in range(self.n)]
          action = random.choice([entangles, swaps])

        state = self.get_state_vector()
        reward = self.update_environment(action)
        file.write(f"Action: {action},Reward: {reward}")
        totalReward += reward
        rewardList.append(totalReward)
        fidelity += self.network.getLink((0,self.n-1),1)
        fidelityList.append(fidelity)

        if reward == 1:
          finalstep = step
          print(f"End-to-end entanglement achieved in {step+1} steps", file=file)
          self.network.endToEndCheck()
          self.network.resetState()
        print('Max iterations reached', file=file) if step == max_steps-1 else None
        finalstep = step

      if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        plot_title = f"Metrics for {kind} for $(n, p_E, p_S)$= ({self.n}, 0.85, 0.85) over {max_steps} steps"
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
        plt.savefig('Test.png')
        plt.xlabel('Step')
    return finalstep
