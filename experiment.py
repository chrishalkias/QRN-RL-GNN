# -*- coding: utf-8 -*-

"""
Created on Thu 6 Mar 2025

Includes the Experiment class which is used to run experiments on the network.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import gym
from gym import spaces
from repeater_network import RepeaterNetwork
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from graph_models import GNNPolicy, GATPolicy
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.summary import summary
from graph_models import GNNPolicy, GATPolicy
from quantum_network_env import QuantumNetworkEnv
import os

class CumulativeLossCallback(BaseCallback):
  """Uncomment lines for debugging"""
  def __init__(self, verbose=0):
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
  def __init__(self, model="DQN", n=5, directed=False,geometry='chain',
                     kappa=1, tau=1_000, p_entangle=1, p_swap=1):
    """
         ______                      _                      _
        |  ____|                    (_)                    | |
        | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_
        |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __|
        | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_
        |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|
                    | |
                    |_|

    Merger class for all the above classes used to run experiments on the net.
    The display info method gives an overview of virtually all of the parameters
    in the experiment. The random sample does exactly what it sounds like. The
    train_agent method trains the agent on the environment. The test_agent
    method tests the agent on the environment and either produces a text or a
    plot output of the system.

    It seems that 10_000 timesteps are not enough for good performance in any
    case. I believe that for large n and small p's the number of timesteps
    required to establish a policy is of the order of 1_000_000. Conjecture

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
    self.env = QuantumNetworkEnv(n, directed,geometry,kappa, tau, p_entangle, p_swap)
    self.action_size = self.env.actionCount()

    if model == "DQN":
      self.net_type = DQN
      self.model = model = PPO(
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

    elif model == 'GNN':
      self.net_type = GNNPolicy
      kwarg_dict={
        "features_extractor_class": self.net_type,
        "features_extractor_kwargs": {
            "graph": self.env._get_obs(1),
            "features_dim": 64}}
      self.model = PPO(
        "MlpPolicy",
        self.env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=kwarg_dict)

    elif model == "GAT":
      self.net_type = GATPolicy
      kwarg_dict={
        "features_extractor_class": self.net_type,
        "features_extractor_kwargs": {
            "graph": self.env._get_obs(1),
            "features_dim": 64,
            "num_heads": 4,
            "pooling": "mean"}}
      self.model = PPO(
        "MlpPolicy",
        self.env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=kwarg_dict)

  # self.policy_kwargs = kwarg_dict if kwarg_dict else None
    self.obs, self.info, self.action, self.reward, self.done, \
     self.terminated, self.truncated = (None for _ in range(7))
    # self.done = self.terminated or self.truncated

  def display_info(self):
    """Prints information about the test"""
    env= self.env
    line = '\n' + '='*50 + '\n'
    with open("output.txt", "w") as file:
      for file in [file, None]: #save and output at the same time
        print(line, file=file)
        print(f'     >>> Experiment parameters <<<', file=file)
        print(line, file=file)
        print(f'Environment  : {env.__class__.__name__}', file=file)
        print(f'n            : {env.n}', file=file)
        print(f'directed     : {env.directed}', file=file)
        print(f'geometry     : {env.geometry}', file=file)
        print(f'kappa        : {env.kappa}', file=file)
        print(f'tau          : {env.tau}', file=file)
        print(f'p_entangle   : {env.p_entangle}', file=file)
        print(f'p_swap       : {env.p_swap}', file=file)
        print(f'model        : {self.net_type.__name__}', file=file)
        print(f'lr           : {self.model.learning_rate}', file=file)
        print(f'gamma        : {self.model.gamma}', file=file)
        print(f'gae_lambda   : {self.model.gae_lambda}', file=file)
        print(f'ent_coef     : {self.model.ent_coef}', file=file)
        print(f'n_steps      : {self.model.n_steps}', file=file)

        if self.net_type in [GNNPolicy, GATPolicy] and False: #print extra info if PPO
          policy = self.model.policy
          print(line, file=file)
          print('>>> Shared Feature Extractor <<<', file=file)
          print(line, file=file)
          print(policy.mlp_extractor, file=file)
          print('\nActor Network (action_net):', file=file)
          print(policy.action_net, file=file)
          print('\nCritic Network (value_net):', file=file)
          print(policy.value_net, file=file)
          print('\n>>> Policy Network Architecture <<<', file=file)
          print(env.edge_combinations)
          # print(summary(self.model.policy, input_size=(1,env.edge_combinations)), file=file)

  def random_sample(self, n_samples):
    """Randomly samples actions from the environment"""
    for sample in range(n_samples):
      if not self.done:
        action = self.env.action_space.sample() # agent.get_action(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        print("Step:", obs, reward, done, info)

  def train_agent(self, total_timesteps=1000, plot=True, callback=False):
    """Trains the agent on the environment"""
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    loss_callback = CumulativeLossCallback() if callback else None
    self.model.learn(total_timesteps=total_timesteps,
                     callback=loss_callback,
                     progress_bar=True,
                     log_interval=total_timesteps)
    self.model.save(self.net_type.__name__)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    log_file = "./logs/progress.csv"
    if plot and loss_callback.cumulative_losses and loss_callback.losses:
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
      plt.savefig(f'metrics_{self.net_type}_.png')
      plt.show()

    elif (not loss_callback.cumulative_losses) or (not loss_callback.losses):
      print("no callbacks to print")
    elif not print: ...
    else: ...

  def test_agent(self, max_steps =10, render:bool=False):
    """Tests the agent"""
    self.model.load(self.net_type.__name__)
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