#@title Agent GNN
import numpy as np
from repeaters import RepeaterNetwork
from strategies import Heuristics
from model import GNN
from buffer import Buffer
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



class AgentGNN(RepeaterNetwork):

  def __init__(self,
               n=4,
               directed = False,
               geometry = 'chain',
               tau = 1_000,
               cutoff = None,
               p_entangle = 1,
               p_swap = 1,
               lr=0.001,
               gamma=0.95,
               epsilon=1):
    
    """
    Implements the RL agent used to learn strategies on the Repeater Network.
    It also includes the heuristics tested against.

    Composed from: RepeaterNetwork

    Methods:
      get_state_vector()
      get_valid_actions()
      choose_action()
      update_environment()
      reward()
      train()
      saveModel()
      trained_action()
      random_action()
      alternating_action()
      swap_asap()
      test()
    """

    super().__init__(n, directed, geometry, cutoff, tau, p_entangle, p_swap)
    self.lr=lr
    self.gamma = gamma
    self.epsilon = epsilon
    self.criterion = nn.MSELoss()
    states = len(self.get_state_vector())
    actions = len(self.new_actions())
    self.model = GNN()
    self.target_model = type(self.model)()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.memory = Buffer(max_size = 10_000)

  def store_obs(self):
    """Store SAR pairs to use in the replay buffer"""

    state = self.get_state_vector()
    action = self.choose_action()
    reward = self.update_environment(action)

    

  def get_state_vector(self) -> torch.tensor:
    """Returns the state of entanglements in the network (pyG.Data)"""
    return self.tensorState() #Data structure with x=torch.ones and edge_attr

  def get_valid_actions(self) -> list:
    """
    If the repeater is the last of the chain it dissalows EG to the right.
    Checks if there is a left and a right connection for each repeater.
    If both exists, it allows swap on that repeater.
    Appends an even number (node*2) for EG and an odd one (node*2 +1) for swaps

    Returns:
      valid_actions (list) : 
    """
    valid_actions = []
    for node in range(self.n):
        if node < self.n - 1:  # Can entangle right
            valid_actions.append(node * 2)

        if 0 < node < self.n - 1:  # Is intermediate node
          linked_left, linked_right = False, False #init to false
          for other_node in range(self.n): #has 2 connections
            if node < other_node:
              linked_left = (self.getLink((node, other_node), 1) > 0)
            elif node > other_node:
              linked_right = (self.getLink((other_node, node), 1) > 0)
            if linked_left and linked_right:
              valid_actions.append(node * 2 + 1)
    return valid_actions

  def choose_action(self, use_trained_model=False) -> int:
    """Epsilon-greedy action choice"""
    if not use_trained_model and random.uniform(0, 1) < self.epsilon:
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions)

    with torch.no_grad():
        q_values = self.target_model(self.get_state_vector()).flatten()

        # Create a mask for valid actions
        mask = torch.full_like(q_values, float('-inf'))
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = 0.0

        # Apply mask and use softmax to ensure numerical stability
        masked_q_values = q_values + mask
        action_probs = torch.softmax(masked_q_values, dim=0)

        # Select action with highest probability (equivalent to argmax but safer)
        return torch.multinomial(action_probs, 1).item()

  def update_environment(self, action) -> float:
    """Applies the action to the environment and returns the reward.
    The action is of the form `self.entangle((i,j))` or `self.swapAT(k)`.
    See the `RepeaterNetwork` documentation for refference.
    """
    exec(self.new_actions()[action])
    return self.reward()

  def reward(self) -> float:
    """
    Computes the agents reward.
    This is based off the end-to-end condition(+1/-0.01)
    And a bonus reward that is proportional to the existing links age and length.
    """
    bonus_reward = 0 # some function f(d, e; n)
    for (i,j), (adj, entanglement) in self.matrix.items():
        bonus_reward +=  entanglement*(j-i)/(self.n**2) if entanglement else 0
    return 1 if self.endToEndCheck() else -0.01 + bonus_reward/10


  def step(self):
    """
    Perform one observation-action-reward-store step.
    > To be fed into `Q-estimate`

    Returns: The (S, A, R, S') tuple
    """
    state = self.get_state_vector()
    action = self.choose_action()
    reward = self.update_environment(action)
    next_state = self.get_state_vector()

    self.memory.add(state = state, 
            action=action, 
            reward=reward, 
            next_state=next_state)
    
    return state, action, reward, next_state


  def Q_estimate(self, state, action, reward, next_state):
    """
    Estimates the current and next q_values.
    To be fed into `backprop`

    Returns: q_value, target
    """
    q_values = self.model(state).flatten()  # Shape: [2*num_nodes]
    q_value = q_values[action]    # gradient-friendly indexing
    with torch.no_grad():
          next_q_values = self.target_model(next_state).flatten()
          target = reward + self.gamma * torch.max(next_q_values)
    return q_value, target
  
  def backprop(self, q_value, target):
    """Performs the backwards pass"""
    loss = self.criterion(q_value, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss

    
  def train(self, 
            episodes=1000, 
            plot=True, 
            save_model=True, 
            savefig=True) -> list:
    
    """Trains the agent"""
    self.reset()
    links_established, totalReward, rewardList = 0, 0, []

    for step in tqdm(range(episodes)):
      state, action, reward, next_state = self.step()
      q_value, target = self.Q_estimate(state, action, reward, next_state)
      loss = self.backprop(q_value, target)
      self.epsilon = self.epsilon * (1 - step / episodes) #epsilon decay like deepmind

      if step % 100 == 0:
        self.target_model.load_state_dict(self.model.state_dict())

      totalReward += reward
      rewardList.append(totalReward)
      wincon = self.endToEndCheck()

      if wincon:
        links_established +=1
        self.reset()


    if plot:
      print(f'Total links established = {links_established}')
      plt.plot(rewardList,ls='-', label='Cummulative reward')
      plt.title(f'Training metrics over {episodes} steps')
      plt.xlabel('Episode')
      plt.ylabel(f'Reward for $(n, p_E, p_S)$= {self.n, self.p_entangle, self.p_swap}')
      plt.yscale("symlog")
      plt.legend()
      plt.savefig('assets/train.png') if savefig else None
      plt.show()

    torch.save(self.model.state_dict(), 'assets/gnn_model.pth') if save_model else None
    return rewardList, links_established

  
  def test(self, 
           n_test=5, 
           p_entangle=1, 
           p_swap=1, 
           tau=1000, 
           cutoff = 10_000,
           max_steps=100, 
           verbose=False, 
           kind='trained') -> list:
    
    """
    
    Perform the validation of the agent against the heuristic strategies
    
    """
    super().__init__(n=n_test, tau=tau, cutoff=cutoff, p_entangle=p_entangle, p_swap=p_swap)
    totalReward, rewardList = 0, []
    fidelity, fidelityList = 0,[]
    links_established, linkList = 0, []
    linkrate = []
    self.reset()
    strategies = Heuristics(self)

    for step in range(max_steps):
      reward=0

      # allow for self.n actions per round for normalization
      if kind=='trained':
        for _ in range(self.n):
          action = self.choose_action(use_trained_model=True)
          reward += self.update_environment(action)
      elif kind=='swap_asap':
        for _ in range(self.n):
          action = strategies.swap_asap()
          exec(action)
          reward += self.reward()
      elif kind=='random':
        for _ in range(self.n):
          action = strategies.stochastic_action()
          exec(action)
          reward += self.reward()
      
      links_established +=1 if self.endToEndCheck() else 0
      self.reset() if self.endToEndCheck() else None
      print(f"Round: {step}, Reward: {reward:.3f}") if verbose else None
      totalReward += reward
      rewardList.append(totalReward)
      linkList.append(links_established)
      linkrate.append(links_established/(step+1))
    return rewardList, linkList, linkrate