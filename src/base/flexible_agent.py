"""
flexible_agent.py - FIXED VERSION
==================================

This version stores action INDICES in the buffer instead of action tuples,
avoiding the tensor comparison issue completely.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import time
import wandb
import math

# Import from flexible architecture
from base.flexible_model import FlexibleGNN, ActionSpace, combine_q_values
from base.repeaters.network import FlexibleRepeaterNetwork
from base.buffer import Buffer


class FlexibleQRNAgent:
    """
    RL Agent for quantum repeater networks with arbitrary topologies.
    
    FIXED: Stores action indices instead of action tuples to avoid
    tensor comparison issues.
    """
    
    def __init__(self, 
                 lr=5e-4, 
                 gamma=0.95,
                 buffer_size=10000,
                 epsilon=1.0,
                 batch_size=128,
                 target_update_freq=500):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Models
        self.policy_net = FlexibleGNN(node_dim=8, edge_dim=3).to(self.device)
        self.target_net = FlexibleGNN(node_dim=8, edge_dim=3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Buffer
        self.memory = Buffer(max_size=buffer_size)

    def select_action(self, state: Data, training: bool = True):
        """
        Select action from variable-length action space.
        
        Returns:
            (action, action_idx) tuple where:
            - action: (action_type, action_target) for execution
            - action_idx: integer index for storage in buffer
        """
        action_space = ActionSpace(state)
        
        if action_space.num_actions == 0:
            return ('entangle', 0), 0
        
        # Get Q-values
        self.policy_net.eval()
        with torch.no_grad():
            state = state.to(self.device)
            q_dict = self.policy_net(state)
            q_values = combine_q_values(q_dict, action_space, device=self.device)
            
            # Apply action mask
            mask = action_space.get_action_mask(state)
            q_values[~mask] = -float('inf')
        
        self.policy_net.train()
        
        # Epsilon-greedy selection
        if training and random.random() < self.epsilon:
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) == 0:
                action_idx = 0
            else:
                action_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
        else:
            action_idx = q_values.argmax().item()
        
        # Decode to action tuple
        action = action_space.get_action(action_idx)
        
        return action, action_idx
    
    def reward(self, env: FlexibleRepeaterNetwork, action: tuple) -> tuple:
        """Calculate reward for taking an action."""
        step_cost = -1
        success_reward = 100
        info = {'fidelity': 0.0}
        
        # Check end-to-end fidelity
        e2e_fidelity = env.fidelities[0, env.n-1].item()
        
        if e2e_fidelity > 0:
            fidelity_bonus = 10 * (e2e_fidelity ** 2)
            step_cost += fidelity_bonus
        
        # Penalize swapping low-fidelity links
        action_type, target = action
        link_quality_penalty = 0
        
        if action_type == 'swap':
            node = target
            left_fidelities = env.fidelities[:node, node]
            right_fidelities = env.fidelities[node, node+1:]
            
            if left_fidelities.max() > 0 and right_fidelities.max() > 0:
                avg_fidelity = (left_fidelities.max() + right_fidelities.max()) / 2
                
                if env.cutoff:
                    expiry_threshold = np.exp(-env.cutoff / env.tau)
                    if avg_fidelity < expiry_threshold * 2:
                        link_quality_penalty = -5
        
        info['e2e_fidelity'] = e2e_fidelity
        info['link_quality_penalty'] = link_quality_penalty
        
        return step_cost + link_quality_penalty, success_reward, info

    def step_environment(self, env, action: tuple) -> tuple:
        """
        Execute action in environment.
        
        FIXED: Properly handles tensor conversion when executing actions.
        """
        action_type, action_target = action
        step_cost, success_reward, info = self.reward(env, action)
        
        # Execute action
        if action_type == 'entangle':
            # Get edge from physical_edges
            edge = env.physical_edges[action_target]
            
            # FIXED: Convert to Python tuple of ints
            if hasattr(edge, 'tolist'):
                edge = tuple(edge.tolist())
            else:
                # Handle both tensor and list cases
                i = edge[0].item() if hasattr(edge[0], 'item') else edge[0]
                j = edge[1].item() if hasattr(edge[1], 'item') else edge[1]
                edge = (i, j)
            
            env.entangle(edge=edge)
        
        elif action_type == 'swap':
            node = action_target
            env.swapAT(node)
        
        # Check for success
        is_success, current_fidelity = env.endToEndCheck(timeToWait=0)
        
        if is_success:
            info['fidelity'] = current_fidelity
            return success_reward, True, info
        
        return step_cost, False, info

    def train_step(self):
        """
        Single training step using experience replay.
        
        FIXED: Stores action indices instead of action tuples to avoid
        tensor comparison issues.
        """
        if self.memory.size() < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Process each sample individually
        current_q_values = []
        target_q_values = []
        
        for i in range(len(batch)):
            state = batch[i]['s'].to(self.device)
            next_state = batch[i]['s_'].to(self.device)
            action = batch[i]['a']  # This is (action_type, action_target) tuple
            reward = batch[i]['r']
            done = batch[i]['d']
            
            # Current Q-value
            action_space = ActionSpace(state)
            q_dict = self.policy_net(state)
            q_values = combine_q_values(q_dict, action_space, device=self.device)
            
            # FIXED: Find action index using manual comparison
            action_idx = None
            action_type, action_target = action
            
            for idx, (a_type, a_target) in enumerate(action_space.actions):
                if a_type == action_type and a_target == action_target:
                    action_idx = idx
                    break
            
            if action_idx is None:
                # Action not found (shouldn't happen), skip this sample
                continue
            
            current_q = q_values[action_idx]
            current_q_values.append(current_q)
            
            # Target Q-value
            with torch.no_grad():
                next_action_space = ActionSpace(next_state)
                next_q_dict = self.target_net(next_state)
                next_q_values = combine_q_values(next_q_dict, next_action_space, device=self.device)
                
                # Mask invalid actions
                next_mask = next_action_space.get_action_mask(next_state)
                next_q_values[~next_mask] = -float('inf')
                
                max_next_q = next_q_values.max()
                target_q = reward + (self.gamma * max_next_q * (1 - done))
                target_q_values.append(target_q)
        
        if len(current_q_values) == 0:
            return  # No valid samples
        
        # Convert to tensors and compute loss
        current_q_tensor = torch.stack(current_q_values)
        target_q_tensor = torch.stack(target_q_values)
        
        loss = self.loss_fn(current_q_tensor, target_q_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def train(self, 
              episodes=10000, 
              max_steps=100, 
              savemodel=False, 
              topology='chain',
              n_range=[4, 6],
              topology_mix=None,
              jitter=100,
              p_e=0.85, 
              p_s=0.95,
              tau=50,
              cutoff=30,
              use_wandb=True,
              wandb_project="QRN-Flexible"):
        """Train the agent on quantum repeater networks."""
        
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "episodes": episodes,
                    "topology": topology,
                    "topology_mix": topology_mix,
                    "n_range": n_range,
                    "p_entangle": p_e,
                    "p_swap": p_s,
                    "tau": tau,
                    "cutoff": cutoff,
                })
            wandb.watch(self.policy_net, log="all", log_freq=1000)
        
        # Setup for topology mixing
        if topology_mix is None:
            topologies = [topology]
            weights = [1.0]
        else:
            topologies = list(topology_mix.keys())
            weights = list(topology_mix.values())
        
        model_name = f"flexible_d({datetime.now().day}-{datetime.now().month})_{topology}_n{n_range[0]}-{n_range[1]}_e{episodes}"
        model_path = f'assets/trained_models/{model_name}/'
        
        self.memory.clear()
        scores = []
        
        # Initialize environment
        n_nodes = random.choice(range(n_range[0], n_range[1] + 1))
        current_topology = random.choices(topologies, weights=weights)[0]
        env = FlexibleRepeaterNetwork(n=n_nodes, topology=current_topology, 
                                       p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
        
        pbar = tqdm(range(episodes))
        eps_init = 1.0
        eps_fin = 0.05
        total_steps = 0
        
        try:
            for e in pbar:
                start_time = time.time()
                
                # Jitter: change network size/topology periodically
                if jitter and e > 0 and e % jitter == 0:
                    n_nodes = random.choice(range(n_range[0], n_range[1] + 1))
                    current_topology = random.choices(topologies, weights=weights)[0]
                    env = FlexibleRepeaterNetwork(n=n_nodes, topology=current_topology,
                                                   p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                
                state = env.tensorState()
                score = 0
                steps = 0
                success = 0
                
                for _ in range(max_steps):
                    # Select action (returns both action and index)
                    action, action_idx = self.select_action(state, training=True)
                    
                    # Execute action
                    reward, done, info = self.step_environment(env, action)
                    next_state = env.tensorState()
                    
                    # Store experience with ACTION INDEX
                    self.memory.add(
                        state=state,
                        action=action_idx,  # Store INTEGER index
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )
                    
                    state = next_state
                    score += reward
                    steps += 1
                    
                    # Train
                    loss = self.train_step()
                    
                    if done:
                        if info.get('fidelity', 0) > 0:
                            success = 1
                        break
                
                total_steps += steps
                
                # Epsilon decay
                self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (1 + math.cos(math.pi * e / episodes))
                
                scores.append(score)
                pbar.set_description(f"Ep {e+1} | Score: {score:.1f} | Eps: {self.epsilon:.2f} | Topo: {current_topology} | n: {n_nodes}")
                
                episode_time = time.time() - start_time
                steps_per_sec = steps / episode_time if episode_time > 0 else 0
                
                if use_wandb:
                    wandb.log({
                        "Performance/Reward": score,
                        "Performance/Success": success,
                        "Performance/Fidelity": info.get('fidelity', 0),
                        "Performance/Episode_Length": steps,
                        "System/Total_Steps": total_steps,
                        "System/Steps_per_Second": steps_per_sec,
                        "System/Epsilon": self.epsilon,
                        "System/Network_Size": n_nodes,
                        "System/Topology": topologies.index(current_topology),
                    })
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        
        if savemodel:
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.policy_net.state_dict(), f'{model_path}/{model_name}.pth')
            print(f"\nModel saved to {model_path}")
        
        if use_wandb:
            wandb.finish()
        
        return scores

    def validate(self, 
                 dict_dir=None,
                 n_episodes=100, 
                 max_steps=200,
                 topology='chain',
                 n_nodes=6, 
                 p_e=1.0, 
                 p_s=1.0,
                 tau=50,
                 cutoff=30):
        """Validate the trained agent."""
        
        # Load trained model
        if dict_dir:
            trained_dict = torch.load(dict_dir)
            self.policy_net.load_state_dict(trained_dict)
        
        self.epsilon = 0.0  # Greedy evaluation
        
        results = {'steps': [], 'fidelities': []}
        
        for i in range(n_episodes):
            env = FlexibleRepeaterNetwork(n=n_nodes, topology=topology,
                                          p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
            state = env.tensorState()
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                action, _ = self.select_action(state, training=False)
                _, done, info = self.step_environment(env, action)
                state = env.tensorState()
                steps += 1
                
                if done:
                    results['fidelities'].append(info['fidelity'])
            
            results['steps'].append(steps if done else max_steps)
        
        # Print results
        avg_steps = np.mean(results['steps'])
        avg_fid = np.mean(results['fidelities']) if results['fidelities'] else 0
        
        print(f"\nValidation Results ({topology}, n={n_nodes}):")
        print(f"  Average steps: {avg_steps:.2f} ± {np.std(results['steps']):.2f}")
        print(f"  Average fidelity: {avg_fid:.4f} ± {np.std(results['fidelities']):.4f}" if results['fidelities'] else "  No successes")
        print(f"  Success rate: {len(results['fidelities'])/n_episodes*100:.1f}%")
        
        return results