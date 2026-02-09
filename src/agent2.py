import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import provided modules
from base.repeaters import RepeaterNetwork
from base.buffer import Buffer
from base.model import GNN
from base.strategies import Heuristics

class QRNAgent:
    def __init__(self, 
                 lr=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.05, 
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 target_update_freq=10):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Models
        # GNN is graph-size invariant
        self.policy_net = GNN(node_dim=2, output_dim=2).to(self.device)
        self.target_net = GNN(node_dim=2, output_dim=2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Buffer
        self.memory = Buffer(max_size=buffer_size)

    def get_valid_actions_mask(self, n_nodes):
        """
        Creates a boolean mask for the flattened action space of size 2 * n_nodes.
        """
        mask = torch.zeros(n_nodes * 2, dtype=torch.bool)
        
        for i in range(n_nodes):
            # 1. Check Entangle (i, i+1) -> Valid for 0 to n-2
            if i < n_nodes - 1:
                mask[2*i] = True
            
            # 2. Check Swap at i -> Valid for 1 to n-2 (not endpoints)
            if 0 < i < n_nodes - 1:
                mask[2*i + 1] = True
                
        return mask.to(self.device)

    def select_action(self, state, n_nodes, training=True):
        """
        Selects an action using Epsilon-Greedy strategy with dynamic masking.
        """
        mask = self.get_valid_actions_mask(n_nodes)
        
        if training and random.random() < self.epsilon:
            # Random valid action
            valid_indices = torch.nonzero(mask).squeeze()
            if valid_indices.numel() == 0:
                return 0 
            
            if valid_indices.ndim == 0:
                 return valid_indices.item()
            return valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state) # Shape [N, 2]
                
                # Flatten to [N*2]
                q_flat = q_values.view(-1)
                
                # Apply mask (set invalid actions to negative infinity)
                q_flat[~mask] = -float('inf')
                
                action_idx = q_flat.argmax().item()
            self.policy_net.train()

        return action_idx

    def decode_action(self, action_idx):
        """
        Decodes flattened action index back to (node, operation)
        """
        node = action_idx // 2
        op_type = action_idx % 2 # 0 = Entangle, 1 = Swap
        return node, op_type

    def step_environment(self, env, action_idx):
        """
        Executes the action on the environment.
        Returns: reward, done, info
        """
        node, op_type = self.decode_action(action_idx)
        
        step_cost = -1
        success_reward = 100
        info = {'fidelity': 0.0}
        

        # Execute Action
        if op_type == 0: 
            env.entangle(edge=(node, node+1))
        
        elif op_type == 1: 
            env.swapAT(node)

        current_fidelity = env.getLink((0, env.n-1), 1)
        is_success = env.endToEndCheck(timeToWait=0)

        if is_success:
            info['fidelity'] = current_fidelity
            return success_reward, True, info
        
        return step_cost, False, info


    def train_step(self, n_nodes_current_batch):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        
        state_batch_list = [x['s'] for x in batch]
        next_state_batch_list = [x['s_'] for x in batch]
        
        state_batch = Batch.from_data_list(state_batch_list).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_batch_list).to(self.device)
        
        action_batch = torch.tensor([x['a'] for x in batch], device=self.device)
        reward_batch = torch.tensor([x['r'] for x in batch], device=self.device).float()
        
        # --- Current Q Values ---
        q_values_flat = self.policy_net(state_batch).view(-1)
        
        # Batch offset calculation
        batch_offset = torch.arange(self.batch_size, device=self.device) * (n_nodes_current_batch * 2)
        global_action_indices = action_batch + batch_offset
        
        current_q = q_values_flat[global_action_indices]

        # --- Target Q Values ---
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).view(-1)
            next_q_reshaped = next_q_values.view(self.batch_size, -1)
            
            mask = self.get_valid_actions_mask(n_nodes_current_batch)
            next_q_reshaped[:, ~mask] = -float('inf')
            
            max_next_q = next_q_reshaped.max(dim=1)[0]
            target_q = reward_batch + (self.gamma * max_next_q)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, 
              episodes=500, 
              max_steps=50, 
              savemodel=False, 
              plot=False, 
              savefig=False, 
              n_nodes=4, 
              p_e=1.0, 
              p_s=1.0):
        
        print(f"Starting training: {episodes} episodes | N={n_nodes} | P_ent={p_e} | P_swap={p_s}")
        
        # Clear buffer to prevent batch shape mismatch
        self.memory.clear() 
        
        scores, links = [], []
        pbar = tqdm(range(episodes))
        
        for e in pbar:
            env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s)
            state = env.tensorState()
            score = 0
            
            for _ in range(max_steps):
                action_idx = self.select_action(state, n_nodes, training=True)
                
                reward, done, _ = self.step_environment(env, action_idx)
                next_state = env.tensorState()
                
                self.memory.add(state=state, 
                                action=action_idx, 
                                reward=reward, 
                                next_state=next_state,
                                next_mask=None)
                
                state = next_state
                score += reward
                
                self.train_step(n_nodes_current_batch=n_nodes)
                
                if done:
                    links.append(1)
                    break
            
            self.epsilon = max(0.05, 1 - (e / episodes) * (1-0.05)) # decay from e=1 to e=0.05
            scores.append(score)
            pbar.set_description(f"Ep {e+1} | Score: {score:.1f} | Eps: {self.epsilon:.2f}")
        if savemodel:
            torch.save(self.target_net.state_dict(), 'assets/gnn_model.pth')

        if plot:
            plt.plot(scores,ls='-', label='Average batch-reward')
            plt.title(f'Training metrics over {episodes} steps')
            plt.xlabel('Episode')
            plt.ylabel(f'Reward for $(n, p_E, p_S)$= {n_nodes, p_e, p_s}')
            plt.legend()
            plt.savefig('assets/train.png') if savefig else None
            plt.show()

    def validate(self, 
                 n_episodes=100, 
                 max_steps=100, 
                 n_nodes=4, 
                 p_e=1.0, 
                 p_s=1.0):
        
        print(f"\n--- Validation (N={n_nodes}, P_ent={p_e}, P_swap={p_s}) ---")
        
        results = {
            'Agent': {'steps': [], 'fidelities': []},
            'SwapASAP': {'steps': [], 'fidelities': []},
            'Random': {'steps': [], 'fidelities': []}
        }
        
        # 1. Test Agent
        temp_epsilon = self.epsilon
        self.epsilon = 0.0 # Force Greedy
        
        for _ in range(n_episodes):
            env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s)
            steps = 0
            done = False
            while not done and steps < max_steps:
                state = env.tensorState()
                action_idx = self.select_action(state, n_nodes, training=False)
                
                # step_environment now handles the "peek" logic and returns fidelity in info
                _, done, info = self.step_environment(env, action_idx)
                
                if done:
                    results['Agent']['fidelities'].append(info['fidelity'])
                
                steps += 1
            results['Agent']['steps'].append(steps if done else max_steps)
            
        self.epsilon = temp_epsilon

        # 2. Test Heuristics
        heuristics_map = {'SwapASAP': 'swap_asap', 'Random': 'stochastic_action'}
        
        for name, method_name in heuristics_map.items():
            for _ in range(n_episodes):
                env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s)
                heuristic = Heuristics(env)
                steps = 0
                done = False
                
                while not done and steps < max_steps:
                    if method_name == 'swap_asap':
                        action_str = heuristic.swap_asap()
                    else:
                        action_str = heuristic.stochastic_action()
                    
                    try:
                        if action_str:
                            # HACK: Execute heuristic string on local env
                            exec(action_str.replace("self.", "env."))
                        
                        current_fid = env.getLink((0, env.n-1), 1)
                        is_success = env.endToEndCheck(timeToWait=0)

                        if is_success:
                            done = True
                            results[name]['fidelities'].append(current_fid)
                            
                    except:
                        pass
                    steps += 1
                results[name]['steps'].append(steps if done else max_steps)

        # Print Statistics
        print(f"{'Strategy':<15} | {'Avg Steps':<10} | {'Avg Fidelity':<12} | {'Success':<9} | ")
        print("-" * 60)
        for strategy, data in results.items():
            avg_steps = np.mean(data['steps'])
            std_steps = np.std(data['steps'])
            
            # Success rate calculation
            success_count = sum(1 for s in data['steps'] if s < max_steps)
            success_rate = (success_count / n_episodes) * 100
            
            # Average fidelity (only for successful runs)
            if len(data['fidelities']) > 0:
                avg_fid = np.mean(data['fidelities'])
            else:
                avg_fid = 0.0
                
            print(f"{strategy:<15} | {avg_steps:<10.2f} | {avg_fid:<12.4f} | {success_rate:<8.1f}% ")

if __name__ == "__main__":
    pe, ps = 0.1, 0.9
    agent = QRNAgent(buffer_size=10_000)
    agent.train(episodes=3000, max_steps=100, savemodel=True, n_nodes=4, p_e=pe, p_s=ps)
    agent.validate(n_episodes=100, max_steps=100, n_nodes=6, p_e=pe, p_s=ps)