# -*- coding: utf-8 -*-
# src/repeaters.py

'''
Created Fri 09 May 2025
Test things here
'''
from repeaters import RepeaterNetwork
from models import CNN, GNN
from gnn_env import Environment as EnvG
from cnn_env import Environment as EnvC

net = RepeaterNetwork()
model = GNN()
env = EnvG(model)


# import numpy as np
# import torch
# import torch.nn as nn
# """Fix the scalar Q value with the following code"""

# def select_action(state: torch.tensor, epsilon: float, model: object, action_dims: tuple):
#     if np.random.random() < epsilon:
#         # Random multi-dimensional action (e.g., tuple of indices)
#         return tuple(np.random.randint(dim) for dim in action_dims: tuple)
#     else:
#         with torch.no_grad():
#             q_values = model(state)
#             # Get the indices of the max Q-value
#             return np.unravel_index(q_values.argmax().item(), action_dims)
        
# def compute_loss(batch, model, target_model, gamma, action_dims):
#     states, actions, rewards, next_states, dones = batch
#     # Convert actions to flat indices for gathering Q-values
#     actions_flat = np.ravel_multi_index(actions.T, action_dims)  # (batch_size,)

#     # Current Q-values (batch_size, *action_dims)
#     current_q = model(states)
#     current_q = current_q.view(len(states), -1)  # Flatten to (batch_size, num_actions)
#     current_q = current_q.gather(1, actions_flat.unsqueeze(1))  # (batch_size, 1)

#     # Target Q-values
#     with torch.no_grad():
#         next_q = target_model(next_states).view(len(next_states), -1)
#         max_next_q = next_q.max(1)[0]
#         target_q = rewards + gamma * max_next_q * (1 - dones)

#     # MSE Loss
#     loss = nn.functional.mse_loss(current_q.squeeze(), target_q)
#     return loss

# from collections import deque
# import random

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
    
# # Hyperparameters
# BATCH_SIZE = 32
# GAMMA = 0.99
# EPS_START = 1.0
# EPS_END = 0.01
# EPS_DECAY = 0.995
# TARGET_UPDATE = 10

# # Initialize
# env = YourCustomEnv()  # Replace with your environment
# action_dims = (5, 5)  # Example: 5x5 action grid
# model = DQN(input_dim=env.observation_space.shape[0], action_dims=action_dims)
# target_model = DQN(input_dim=env.observation_space.shape[0], action_dims=action_dims)
# target_model.load_state_dict(model.state_dict())
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# buffer = ReplayBuffer(10000)
# epsilon = EPS_START

# for episode in range(1000):
#     state = env.reset()
#     done = False
#     while not done:
#         # Select action
#         action = select_action(torch.FloatTensor(state), epsilon, model, action_dims)
        
#         # Take step
#         next_state, reward, done, _ = env.step(action)
        
#         # Store transition
#         buffer.push(state, action, reward, next_state, done)
        
#         # Train
#         if len(buffer) >= BATCH_SIZE:
#             batch = buffer.sample(BATCH_SIZE)
#             loss = compute_loss(batch, model, target_model, GAMMA, action_dims)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         state = next_state
    
#     # Update target network
#     if episode % TARGET_UPDATE == 0:
#         target_model.load_state_dict(model.state_dict())
    
#     # Decay epsilon
#     epsilon = max(EPS_END, epsilon * EPS_DECAY)