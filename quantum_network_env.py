# -*- coding: utf-8 -*-

"""
Created on Thu 6 Mar 2025
The QuantumNetworkEnv class is a container for the RepeaterNetwork class, which
implements the gym.Env class. It is a reinforcement learning environment that
can be used for training and testing reinforcement learning algorithms on the
quantum network task.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import gym
from gym import spaces
from repeater_network import RepeaterNetwork
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
                ____                    _
               / __ \                  | |
              | |  | |_   _  __ _ _ __ | |_ _   _ _ __ ___
              | |  | | | | |/ _` | '_ \| __| | | | '_ ` _  \
              | |__| | |_| | (_| | | | | |_| |_| | | | | | |
             _ \___\_\\__,_|\__,_|_| |_|\__|\__,_|_| |_| |_|
            | \ | |    | |                    | |
            |  \| | ___| |___      _____  _ __| | __
            | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
            | |\  |  __/ |_ \ V  V / (_) | |  |   <
       _____|_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\            _
      |  ____|          (_)                                    | |
      | |__   _ ____   ___ _ __ ___  _ __  _ __ ___   ___ _ __ | |_
      |  __| | '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
      | |____| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_
      |______|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|


    Create a new QuantumNetwork Environment for n repeaters to be used for
    gymnasium
    ----------------------------------------------------------------------------
    This function is the proper "package" of the physical system into a reliable
    framework for training / testing etc. It should be thought of as a container
    of the RepeaterNetwork class with added functionality fo the Reinforcement
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