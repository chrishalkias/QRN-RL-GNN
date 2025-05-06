# -*- coding: utf-8 -*-
# src/gym_env.py

'''
Created on Thu 6 Mar 2025
Contains a container for the RepeaterNetwork class creating a gym env and
an experiment class to run the RL loop (PPO) on the quantum network. 
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import gym
import os
import sys
from gym import spaces
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

#from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
# from stable_baselines3.common.summary import summary
from repeaters import RepeaterNetwork

class QuantumNetworkEnv(gym.Env, RepeaterNetwork):
  def __init__(self,
               n=5,
               directed = False,
               geometry = 'chain',
               kappa = 1,
               tau = 1_000,
               p_entangle = 1,
               p_swap = 1
              ):
    """
               ██████  ██    ██ ███    ███     ███████ ███    ██ ██    ██ 
              ██        ██  ██  ████  ████     ██      ████   ██ ██    ██ 
              ██   ███   ████   ██ ████ ██     █████   ██ ██  ██ ██    ██ 
              ██    ██    ██    ██  ██  ██     ██      ██  ██ ██  ██  ██  
               ██████     ██    ██      ██     ███████ ██   ████   ████   
                                                            
    Description:                                                                
      Creates a new QuantumNetwork gymansium environment for n repeaters to be used
      for the RL loop. This function is the proper "package" of the physical system into a reliable
      framework for training / testing etc. It is a container
      of the RepeaterNetwork class with added functionality for the Reinforcement
      Learning task (step, reward, action, reset, etc).

    Inherits from:
      gym.Env
      RepeaterNetwork

    Attributes:
      n                              (int) : Number of repeaters in the network
      RepeaterNetwork(n).__init__()  (obj) : The RepeaterNetwork class
      action_size                    (int) : The number of possible actions
      action_space                   (obj) : The action space of the env
      observation_space              (obj) : The observation space of the env
      edge_combinations              (int) : The number of edges in the graph

    Methods:
      _get_obs()                     (obj) : Return the current observation
      _get_info()                    (obj) : Return environment information
      reset()                        (obj) : Reset to an initial state
      step(action)                   (obj) : Perform one action in the env
      render()                       (obj) : Render the environment
      close()                        (obj) : Clean up resources
    """
    self.n = n
    super().__init__(n, directed, geometry, kappa, tau, p_entangle, p_swap)
    self.action_size = self.actionCount()
    self.action_space = spaces.discrete.Discrete(self.actionCount())
    self.edge_combinations = int(0.5*n*(n-1))
    assert self.edge_combinations == len(self.matrix.keys()), 'Wrong counting'
    self.observation_space = spaces.Box(
        low=0, high=1.0, shape=(self.edge_combinations,), dtype=np.float32)

  def _get_obs(self, info=0):
    """
    Return the current observation of the environment. Returns a flattened
    array consisting of the ENTANGLEMENTS E_ij in the system and not the
    adjecency list. Implement CC here!

    Args:
      info        (bool)          : Use the full matrix(1) or just the E_ij's(0)

    Returns:
      observation (Tensor) : The current observation of the environment.
    """
    raw_state = self.matrix
    array_state = [self.matrix[key][1] for key in sorted(self.matrix.keys())]
    self.entanglement_state = torch.tensor(array_state, dtype=torch.float32)
    observation =  raw_state if info else self.entanglement_state
    assert (type(observation) == torch.Tensor or dict)
    return observation


  def _get_info(self) -> dict:
    """Return environment-specific information."""
    array_state, matrix_state = [self._get_obs(info) for info in range(2)]
    infodict = {'complete state': matrix_state,
                'entanglement state': array_state,
                'other info': None}
    return infodict


  def reset(self, seed=None, options=None):
      """Reset the environment to an initial state."""
      super().reset(seed=seed)
      self.resetState()
      self.state = self._get_obs()
      return self.state, {}

  def _flatten_observation(self, observation_dict) -> list: #not implemented
      """
      Flatten the observation dictionary into a numerical array. This is for
      the GNN but maybe gets trashed with better implementations
      """
      assert type(observation_dict) == dict, 'Wrong type must be dict'
      flattened = []
      for edge in sorted(observation_dict.keys()):
          flattened.extend(observation_dict[edge])
      return np.array(flattened, dtype=np.float32)

  def step(self, action):
      """Perform one step in the environment.
      Args:
        action      (str)   : The chosen action to be performed

      Returns:
        state      (Tensor) : The new state of the environment
        reward     (float)  : The reward for the action
        terminated (bool)   : Whether the episode has terminated
        truncated  (bool)   : Whether the episode has been truncated
        info       (dict)   : Additional information about the environment
      """
      exec(self.actions()[action])
      self.endToEndCheck()
      reward = 1 if self.global_state else -.1

      #add this to promote entanglements
      reward += sum(self._get_obs().numpy())/(10*self.n)

      terminated = (self.global_state == 1)
      truncated = False
      info = {}
      self.state = self._get_obs()
      # self.state = self.matrix #FOR GNN ??
      # next_observation = self._flatten_observation(self.state)
      #return self._flatten_observation(self.state)
      return self.state, reward, terminated, truncated, info

  def render(self, save_path=None, mode = None):
    """
    Render the quantum network as a graph.

    Color coding:
      Blue: Established entanglement
      Red: Old entanglement
      Gray: Default links

    """
    lines = []
    for (i, j), values in self.matrix.items():
      assert int(values[0]) in [0,1], 'Wrong values in locality matrix'
      assert 0 <= values[1] <= 1, 'Wrong values in entanglement matrix'
      if int(values[0]) == 1:                 #local
          color='whitesmoke'
      elif int(values[0]) == 0:               # non-local
          color=None
      if values[1] > 0.5:                     # entangled
          color = 'blue'
      elif values[1] >0 and values[1] < 0.5:  #old link
          color='red'
      if (i,j) == (0,self.n-1):               #end-to-end link
        if (values[1]== 0):                    #if not entangled
          color='yellow'
        else:                                 #if 'somewhat' entangled
          color='green'
      lines.append(f"{i} {j} {values[0]} {values[1]} {color}")

    # global G, edge_colors
    G = nx.parse_edgelist(lines, nodetype=int, data=(("entanglement", float),("locality", float), ("color", str)))
    Laplacian = nx.normalized_laplacian_matrix(G)
    eigenvalues = np.linalg.eigvals(Laplacian.toarray())
    edge_colors = [d["color"] for _, _, d in G.edges(data=True)]
    pos = nx.spring_layout(G, seed=42)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    axes[0].set_title("Network graph")
    axes[0].set_xlabel('off')
    options = {'node_size':2000, 'edge_color':edge_colors,}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', width=7, ax=axes[0], **options)

    G2 = nx.Graph()
    for (i, j), (L_ij, E_ij) in self.matrix.items():
        G2.add_edge(i, j, weight=E_ij)
    L = nx.laplacian_matrix(G2).toarray()
    eigenvalues = np.linalg.eigvals(L)
    axes[1].set_title("Eigenvalue Distribution")
    axes[1].hist(eigenvalues.real, bins=10, color="blue", alpha=0.8)
    axes[1].set_xlim(0, 5)  # Limit eigenvalues between 0 and 5
    axes[1].set_xlabel("Eigenvalue")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight') if save_path else plt.show()


  def close(self):
    """Clean up, delete model file, logs etc"""
    ...




class CumulativeLossCallback(BaseCallback):
  def __init__(self, verbose=0):
    """This class is used for loss callbacks during trainning"""
    super(CumulativeLossCallback, self).__init__(verbose)
    self.losses, self.cumulative_losses = [], []

  def _on_step(self):
    #print("Logged metrics:", self.model.logger.name_to_value)
    if "train/loss" in self.model.logger.name_to_value:
      loss = self.model.logger.name_to_value["train/loss"] #self.env.step()[1]
      self.losses.append(loss)
      cumulative_loss = np.sum(self.losses)
      self.cumulative_losses.append(cumulative_loss)
    else: ...
      #print("Loss metric not found in logged metrics.")
    return True


class Experiment():
  def __init__(self,
                n=5,
                directed=False,
                geometry='chain',
                kappa=1,
                tau=1_000,
                p_entangle=1,
                p_swap=1,
                log_dir = "./logs"):
    """
    ███████ ██   ██ ██████  ███████ ██████  ██ ███    ███ ███████ ███    ██ ████████ 
    ██       ██ ██  ██   ██ ██      ██   ██ ██ ████  ████ ██      ████   ██    ██    
    █████     ███   ██████  █████   ██████  ██ ██ ████ ██ █████   ██ ██  ██    ██    
    ██       ██ ██  ██      ██      ██   ██ ██ ██  ██  ██ ██      ██  ██ ██    ██    
    ███████ ██   ██ ██      ███████ ██   ██ ██ ██      ██ ███████ ██   ████    ██    
                                                                                    
                                                                                 

    Merger class for repeaters andgym_env used to run experiments on the net.
    The display info method gives an overview of virtually all of the parameters
    in the experiment. The random sample does exactly what it sounds like. The
    train_agent method trains the agent on the environment. The test_agent
    method tests the agent on the environment and either produces a text or a
    plot output of the system.

    Attributes:
      model      (str)   : The type of model to be used
      n          (int)   : The number of repeaters in the network
      directed   (bool)  : Whether the network is directed or not
      geometry   (str)   : The networks geoometry
      kappa      (float) : The (global) link decay coeficcient
      tau :      (float) : The (global to become local) link decay coeficcient
      p_entangle (float) : The (global to become local) entnaglement probability
      p_swap     (float) : The (global to become local) swap probability

    Methods:
      display_info()  : Prints information about the experiment
      random_sample() : Randomly samples actions from the environment
      train_agent()   : Trains the agent on the environment
      test_agent()    : Tests the agent on the environment

    """
    
    # self.env = QuantumNetworkEnv(n, directed,geometry,kappa, tau, p_entangle, p_swap)
    self.env = QuantumNetworkEnv(n, directed,geometry,kappa, tau, p_entangle, p_swap)
    self.action_size = self.env.actionCount()
    self.net_type = DQN

    self.model = PPO(
      "MlpPolicy",
      self.env,
      learning_rate=1e-4,  # Adjust learning rate
      n_steps=2048,        # Increase number of steps per update
      batch_size=64,       # Adjust batch size
      gamma=0.99,          # Discount factor
      gae_lambda=0.95,     # GAE parameter
      clip_range=0.2,      # Clip range for policy updates
      ent_coef=0.1,       # Entropy coefficient (encourage exploration)
      verbose=1,
    )

  # self.policy_kwargs = kwarg_dict if kwarg_dict else None
    self.obs, self.info, self.action, self.reward, self.done, \
     self.terminated, self.truncated = (None for _ in range(7))
    # self.done = self.terminated or self.truncated

  def display_info(self):
    """Prints information about the test"""
    env= self.env
    now = datetime.now()
    with open("logs/training_information.txt", "w") as file:

      file.write(f'-> Experiment parameters at {now} \n')
      file.write(f'Environment  : {env.__class__.__name__} \n')
      file.write(f'n            : {env.n} \n')
      file.write(f'directed     : {env.directed} \n')
      file.write(f'geometry     : {env.geometry} \n')
      file.write(f'kappa        : {env.kappa} \n')
      file.write(f'tau          : {env.tau} \n')
      file.write(f'p_entangle   : {env.p_entangle} \n')
      file.write(f'p_swap       : {env.p_swap} \n')
      file.write(f'model        : {self.net_type.__name__} \n')
      file.write(f'lr           : {self.model.learning_rate} \n')
      file.write(f'gamma        : {self.model.gamma} \n')
      file.write(f'gae_lambda   : {self.model.gae_lambda} \n')
      file.write(f'ent_coef     : {self.model.ent_coef} \n')
      file.write(f'n_steps      : {self.model.n_steps} \n')

      file.write(f'\n -> Policy parameters \n')
      policy = self.model.policy
      file.write('>Shared Feature Extractor \n')
      file.write(str(policy.mlp_extractor))
      file.write('\n >Actor Network (action_net): \n')
      file.write(str(policy.action_net))
      file.write('\n >Critic Network (value_net): \n')
      file.write(str(policy.value_net))
      # print(summary(self.model.policy, input_size=(1,env.edge_combinations)), file=file)

  def random_sample(self, n_samples):
    """Randomly samples actions from the environment"""
    for sample in range(n_samples):
      if not self.done:
        action = self.env.action_space.sample() # agent.get_action(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        print("Step:", obs, reward, done, info)

  def train_agent(self,
                  total_timesteps=1000,
                  plot=True,
                  callback=False
                  ):
    """Trains the agent on the environment"""
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    loss_callback = CumulativeLossCallback() if callback else None
    self.model.learn(total_timesteps=total_timesteps,
                     callback=loss_callback,
                     progress_bar=True,
                     log_interval=total_timesteps)
    self.model.save("logs/DQN_model")
    # with open('progress.csv', 'w') as f:
    #    f.write([*loss_callback.cumulative_losses])
    if plot and callback:
      if loss_callback.cumulative_losses and loss_callback.losses:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
        axes[0].set_title("Cumulative Training Loss Over Time")
        axes[0].set_xlabel("Training Steps")
        axes[0].set_ylabel("Cumulative Loss")
        axes[0].plot(loss_callback.cumulative_losses, label='Cummulative Loss')
        axes[0].legend()
        axes[0].set_xscale("log")
        axes[1].set_title("Training Loss Over Time")
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Loss")
        axes[1].plot(loss_callback.losses, label='Loss')
        axes[1].set_xscale("log")
        plt.title("Trainng metrics")
        plt.legend()
        plt.savefig(f'{log_dir}metrics_DQN_.png')
        plt.show()
      elif (not loss_callback.cumulative_losses) or (not loss_callback.losses):
        print("no callbacks to print")
    else: ...

  def test_agent(self, max_steps =10, render:bool=False):
    """Tests the agent"""
    self.model.load(f'{self.net_type.__name__}_model')
    env = self.env
    obs, info = env.reset()
    for iter in range(max_steps):
      action, _states = self.model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      print(f'ACTIONS:{env.actions()[action][5:]}, REWARD:{reward}, STEP:{iter}')
      env.render() if render else None
      if terminated or truncated:
          obs, info = env.reset()
          break
    env.close()


if __name__ == "__main__" :
  """
                ██████  ██    ██ ███    ██ 
                ██   ██ ██    ██ ████   ██ 
                ██████  ██    ██ ██ ██  ██ 
                ██   ██ ██    ██ ██  ██ ██ 
                ██   ██  ██████  ██   ████                     
  """

  config = {'n'             : 5,
          'tau'           : 10_000,
          'p_entangle'    : 0.8,
          'p_swap'        : 0.8,
          'train_steps'   : 10_000,
          'train_agent'   : False,
          'plot_metrics'  : True,
          'callback'      : True,
          'evaluate_agent': False,
          'render_eval'   : True,
          'display'       : True}

  exp = Experiment(
      n=config['n'],
      tau=config['tau'],
      p_entangle=config['p_entangle'],
      p_swap=config['p_swap'],
      )
    
  exp.display_info() if config['display'] else None
  
  if config['train_agent']:
      exp.train_agent(
          total_timesteps=config['train_steps'], 
          plot=config['plot_metrics'], 
          callback=config['render_eval'])

  if config['evaluate_agent']:
      exp.test_agent(
          max_steps=10, 
          render=config['render_eval'])
  
  exp.env.close()
  print("Program exited with exit code 0")