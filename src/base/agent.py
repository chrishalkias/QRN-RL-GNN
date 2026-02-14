import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import Data
import re
import time
import wandb
import math

# Import locals
from base.repeaters import RepeaterNetwork
from base.buffer import Buffer
from base.model import GNN
from base.strategies import Strategies

class QRNAgent:
    def __init__(self, 
                 lr=1e-3, 
                 gamma=0.99,
                 buffer_size=10000,
                 epsilon=1.0,
                 batch_size=64,
                 target_update_freq=1000):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.lr=lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Models
        # GNN is graph-size invariant
        self.policy_net = GNN(node_dim=3, output_dim=2).to(self.device)
        self.target_net = GNN(node_dim=3, output_dim=2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Buffer
        self.memory = Buffer(max_size=buffer_size)

    def get_valid_actions_mask(self, n_nodes, state=None):
            """Creates a boolean mask for valid actions. Handles single states and batches."""
            if state is None:
                mask = torch.zeros(n_nodes * 2, dtype=torch.bool, device=self.device)
                mask[0:2*(n_nodes-1):2] = True # Entangle
                mask[3:2*(n_nodes-1):2] = True # Swap
                return mask

            # Reshape node features: works for single state (1, N, 2) or batched states (B, N, 2)
            num_graphs = state.x.shape[0] // n_nodes
            x_reshaped = state.x.view(num_graphs, n_nodes, 3) #3=num node features
            
            mask = torch.zeros((num_graphs, n_nodes, 2), dtype=torch.bool, device=self.device)
            
            # 1. Entangle valid for nodes 0 to N-2
            mask[:, :n_nodes-1, 0] = True
            
            # 2. Swap valid for nodes 1 to N-2 ONLY IF they have left AND right connections
            has_left = x_reshaped[:, :, 0] > 0
            has_right = x_reshaped[:, :, 1] > 0
            mask[:, 1:n_nodes-1, 1] = has_left[:, 1:n_nodes-1] & has_right[:, 1:n_nodes-1]
            
            # Return 1D mask for select_action, 2D mask for train_step
            if num_graphs == 1:
                return mask.view(n_nodes * 2) 
            return mask.view(num_graphs, n_nodes * 2)

    def select_action(self, state, n_nodes, training=True):
            # Pass the single state. Returns a 1D [2*n_nodes] mask
            mask = self.get_valid_actions_mask(n_nodes, state)
            
            self.policy_net.eval()
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                q_flat = q_values.view(-1).clone() 
                
                # Apply 1D mask
                q_flat[~mask] = -float('inf')
                max_q = q_flat.max().item()
            self.policy_net.train()

            if training and random.random() < self.epsilon:
                valid_indices = torch.nonzero(mask).squeeze()
                if valid_indices.numel() == 0:
                    action_idx = 0 
                elif valid_indices.ndim == 0:
                    action_idx = valid_indices.item()
                else:
                    action_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
            else:
                action_idx = q_flat.argmax().item()

            return action_idx

    def decode_action(self, action_idx):
        """Decodes flattened action index back to (node, operation)"""
        node = action_idx // 2
        op_type = action_idx % 2 # 0 = Entangle, 1 = Swap
        return node, op_type

    def step_environment(self, env, action_idx):
        """Executes the action on the environment.
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
            scaled_reward = success_reward * current_fidelity
            return success_reward, True, info
        
        return step_cost, False, info

    def _fast_batch(self, data_list, n_nodes):
            """Bypasses PyG batching overhead for uniform graph sizes."""
            # Concat node and edge features directly
            x = torch.cat([d.x for d in data_list], dim=0).to(self.device)
            edge_attr = torch.cat([d.edge_attr for d in data_list], dim=0).to(self.device)
            
            # Offset edge indices using the known, constant node count
            edge_indices = []
            offset = 0
            for d in data_list:
                edge_indices.append(d.edge_index + offset)
                offset += n_nodes
                
            edge_index = torch.cat(edge_indices, dim=1).to(self.device)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    def train_step(self, n_nodes_current_batch):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        
        state_batch_list = [x['s'] for x in batch]
        next_state_batch_list = [x['s_'] for x in batch]
        
        state_batch = self._fast_batch(state_batch_list, n_nodes_current_batch)
        next_state_batch = self._fast_batch(next_state_batch_list, n_nodes_current_batch)


        
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
            
            # Pass next_state_batch. Returns a [batch_size, 2*n_nodes] mask
            mask = self.get_valid_actions_mask(n_nodes_current_batch, next_state_batch)
            
            # Apply 2D mask directly (do not use [:, ~mask])
            next_q_reshaped[~mask] = -float('inf')
            
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
              jitter=100,
              n_range=[4,6],
              fine_tune=False,
              p_e=1.0, 
              p_s=1.0,
              tau=1000,
              cutoff=1000,
              use_wandb=True,
              wandb_project = "QRN-RL"):
        

        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "episodes": episodes,
                    "n_nodes_range": n_range,
                    "p_entangle": p_e,
                    "p_swap": p_s,
                    "tau": tau,
                    "cutoff": cutoff,
                    "fine_tune": fine_tune})
            
            wandb.watch(self.policy_net, log="all", log_freq=1000)
        
        
        model_name = f"d({datetime.now().day}-{datetime.now().month})l{n_range[0]}u{n_range[-1]}e{episodes}m{max_steps}p{f'{p_e:.2f}'[-2:]}a{f'{p_s:.2f}'[-2:]}t{tau}c{cutoff}"
        model_path = f'assets/trained_models/{model_name}/'


        self.memory.clear() 
        
        scores = []
        pbar = tqdm(range(episodes))
        n_nodes = random.choice(n_range)
        eps_init = 1.0 if not fine_tune else 0.2
        eps_fin = 0.05 if not fine_tune else 0.01

        if fine_tune: #load buffer for fine tuning
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5, eps=1e-4)
            env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
            state = env.tensorState()
            for _ in range(self.batch_size * 5):
                action = self.select_action(state, env.n, training=True)
                r, done, info = self.step_environment(env, action)
                next_state = env.tensorState()
                self.memory.add(state, action, r, next_state, None)
                state = next_state
                if done: 
                    env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                    state = env.tensorState()
        
        total_steps = 0
        try:
            for e in pbar:
                start_time = time.time()
                env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                state = env.tensorState()
                score = 0
                steps, success, swaps, entangles, q_value_list = 0, 0, 0,0, []

                if jitter and not e%jitter:
                    new_n = np.random.choice(n_range)
                    if new_n != n_nodes: #clear the buffer to avoid shape mismatch
                        self.memory.clear()
                    new_env = RepeaterNetwork(n=new_n, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                    env=new_env
                    n_nodes = new_n
                    state = env.tensorState()

                for _ in range(max_steps):
                    action_idx = self.select_action(state, n_nodes, training=True)
                    if action_idx % 2 == 0:
                        entangles += 1
                    else:
                        swaps +=1
                    
                    reward, done, info = self.step_environment(env, action_idx)
                    next_state = env.tensorState()
                    
                    self.memory.add(state=state, 
                                    action=action_idx, 
                                    reward=reward, 
                                    next_state=next_state,
                                    next_mask=None)
                    
                    state = next_state
                    score += reward
                    steps += 1
                    
                    self.train_step(n_nodes_current_batch=n_nodes)
                    
                    if done:
                        if info.get('fidelity', 0) > 0:
                            success = steps
                        break

                total_steps += steps
                if not fine_tune:
                    self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (1 + math.cos(math.pi * e / episodes))
                elif fine_tune:
                    self.epsilon = 0.2 * np.exp(-10 * e / episodes)
                scores.append(score)
                pbar.set_description(f"Ep {e+1} | Score: {score:.1f} | Eps: {self.epsilon:.2f}")

                episode_time = time.time() - start_time
                steps_per_sec = steps / episode_time if episode_time > 0 else 0
                action_total = swaps + entangles
                swap_ratio = swaps / action_total if action_total > 0 else 0

                if use_wandb:
                    wandb.log({
                        "Performance/Reward": score,
                        "Performance/Success_Rate": success, # W&B can smooth this for you
                        "Performance/Terminal_Fidelity": info.get('fidelity', 0),
                        "Performance/Episode_Length": steps,
                        "Policy/Swap_Ratio": swap_ratio,
                        "System/total_steps": total_steps,
                        "System/Steps_per_Second": steps_per_sec,
                        "System/Epsilon": self.epsilon})       
                
        except KeyboardInterrupt:
            print("\n\nTraining Interrupted by User (Ctrl+C)!")
            print("Attempting to save current model state...")

       
        if savemodel:
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.policy_net.state_dict(), f'{model_path}/{model_name}.pth')

        if plot:
            plt.plot(scores,ls='-', label='Average batch-reward')
            plt.title(f'{"Training" if not fine_tune else "Fine-tuning"} metrics over {episodes} episodes')
            plt.xlabel('Episode')
            n_label = f'{n_range[0]}-{n_range[-1]}' if jitter else n_range[0]
            label = f'Reward for {r'$n, p_E, p_S, T, c$'}= {n_label}, {p_e}, {p_s}, {tau}, {cutoff}'
            plt.ylabel(label)
            plt.legend()
            os.makedirs(f'assets/trained_models/{model_name}/', exist_ok=True)
            plt.savefig(f'assets/trained_models/{model_name}/train.png') if savefig else None
            plt.show()

        if use_wandb:
            wandb.finish()

    def validate(self, 
                    dict_dir=None,
                    n_episodes=100, 
                    max_steps=100, 
                    n_nodes=4, 
                    p_e=1.0, 
                    p_s=1.0,
                    tau=1000,
                    cutoff=None, 
                    logging=True,
                    plot_actions=True,
                    savefig=True):
            

            def log(msg):
                """Used to print on STDOUT and log to file"""
                print(msg)
                if logging:
                    with open("./assets/validation_results/validation_results.txt", "a") as f:
                        f.write(msg + "\n")
            
            # --- Helper: Action Parser for Plotting ---
            def parse_action_label(action_raw, is_agent=False):
                """Converts raw action (int or string) to shorthand label E(x) or S(x)."""
                if action_raw is None: return "None"
                
                if is_agent:
                    # Agent uses int index: even=Entangle, odd=Swap
                    node = action_raw // 2
                    op_type = action_raw % 2 
                    if op_type == 0: return f"E({node})"
                    return f"S({node})"
                else:
                    # Heuristics use strings
                    if "entangle" in action_raw:
                        # Extract first number from "self.entangle((X, Y))"
                        match = re.search(r"\((\d+),", action_raw)
                        if match: return f"E({match.group(1)})"
                    elif "swapAT" in action_raw:
                        # Extract number from "self.swapAT(X)"
                        match = re.search(r"\((\d+)\)", action_raw)
                        if match: return f"S({match.group(1)})"
                    return "None"

            # --- Helper: Plotting Function ---
            # --- Helper: Plotting Function (MODIFIED) ---
            def plot_action_timeline(action_history):
                strategies = list(action_history.keys())
                
                # 1. Collect all unique labels
                unique_labels = set()
                for seq in action_history.values():
                    for act in seq:
                        if act != "Done": unique_labels.add(act)
                
                # Sort labels: Entangles first, then Swaps
                sorted_labels = sorted(list(unique_labels), key=lambda x: (x[0], int(x[2:-1])))
                
                # 2. Create Color Map
                colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_labels)))
                color_map = {label: colors[i] for i, label in enumerate(sorted_labels)}
                color_map["Done"] = (0, 0, 0, 1) # Black for termination
                
                # 3. Build Grid for Colors
                max_len = max(len(seq) for seq in action_history.values())
                grid = np.zeros((len(strategies), max_len, 4)) # RGBA grid
                grid.fill(1.0) # White background
                
                for i, strat in enumerate(strategies):
                    seq = action_history[strat]
                    for j, action in enumerate(seq):
                        if action in color_map:
                            grid[i, j] = color_map[action]

                # 4. Plot
                fig, ax = plt.subplots(figsize=(12, len(strategies) * 0.8))
                
                # A) Plot the colors
                ax.imshow(grid, aspect='auto', interpolation='nearest')
                
                # B) Overlay Hatching for 'Swap' (S) operations
                # We iterate through the grid and place a hatched rectangle over Swaps
                for i, strat in enumerate(strategies):
                    seq = action_history[strat]
                    for j, action in enumerate(seq):
                        if action.startswith("S"): # Identify Swaps
                            # Create a rectangle centered at (j, i)
                            # xy is bottom-left corner, so (j-0.5, i-0.5)
                            rect = mpatches.Rectangle(
                                (j - 0.5, i - 0.5), 1, 1, 
                                fill=False,         # Transparent background
                                hatch='///',         # Grid pattern
                                edgecolor='black',  # Hatch color
                                linewidth=0,        # No border around the cell
                                alpha=0.5           # Make the hatch subtle
                            )
                            ax.add_patch(rect)

                # Formatting
                ax.set_yticks(np.arange(len(strategies)))
                ax.set_yticklabels(strategies)
                ax.set_xlabel("Time Step")
                ax.set_title(f"Policy Action Timeline (N:{n_nodes}, Pe:{p_e}, Ps:{p_s}, T:{tau}, c:{cutoff})")
                ax.grid(False) 
                
                # 5. Custom Legend
                patches = []
                for l in sorted_labels:
                    # Check if action is a Swap to apply hatching
                    is_swap = l.startswith("S")
                    
                    patches.append(mpatches.Patch(
                        facecolor=color_map[l], 
                        label=l,
                        # Apply grid pattern '++' only for Swaps
                        hatch='///' if is_swap else None,
                        # Hatch color is controlled by edgecolor (must be distinct from facecolor)
                        edgecolor='black' if is_swap else None 
                    ))
                
                patches.append(mpatches.Patch(color='black', label='Terminated'))
                
                # Place legend outside
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Actions")
                
                plt.savefig(f'assets/validation_results/policy_comparison.png') if savefig else None
                plt.show()

            # -----------------------------

            log(f"\n--- Validation (N:{n_nodes}, Pe:{p_e}, Ps:{p_s}, tau:{tau}, cutoff:{cutoff}) ---")
            log(f"--- Max_steps: {max_steps}, n_episodes:{n_episodes}, Model: {dict_dir.split('/')[-1][:-4]} ---")
            
            results = {
                'Agent': {'steps': [], 'fidelities': []},
                'Frontier': {'steps': [], 'fidelities': []},
                'FN_Swap': {'steps': [], 'fidelities': []},
                'SN_Swap': {'steps': [], 'fidelities': []},
                'SwapASAP': {'steps': [], 'fidelities': []},
                # 'Doubling': {'steps': [], 'fidelities': []},
                'Random': {'steps': [], 'fidelities': []},
            }
            
            # Dictionary to store the FIRST episode action sequence for each strategy
            action_timeline = {k: [] for k in results.keys()}

            # --- Load model ---
            if dict_dir == None:
                raise RuntimeWarning(
                    ' No trained dic_dir. Pass a dictionary as kwarg `validate(trained_dict= ...)`')
            
            trained_dict = torch.load(dict_dir)
            self.policy_net.load_state_dict(trained_dict)

            # --- Test Agent ---
            self.epsilon = 0.0 # Force greedy
            
            for i in range(n_episodes):
                env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                steps = 0
                done = False
                
                while not done and steps < max_steps:
                    state = env.tensorState()
                    action_idx = self.select_action(state, n_nodes, training=False)
                    
                    # RECORD ACTION (Only for first episode)
                    if plot_actions and i == 0:
                        label = parse_action_label(action_idx, is_agent=True)
                        action_timeline['Agent'].append(label)

                    _, done, info = self.step_environment(env, action_idx)
                    
                    if done:
                        results['Agent']['fidelities'].append(info['fidelity'])
                        if plot_actions and i == 0:
                            action_timeline['Agent'].append("Done")
                    
                    steps += 1
                results['Agent']['steps'].append(steps if done else max_steps)
                
            # --- Test Heuristics ---
            heuristics_map = {
                'Frontier': 'Frontier',
                'FN_Swap': 'FN_swap',
                'SN_Swap': 'SN_swap',
                'SwapASAP': 'swap_asap', 
                # 'Doubling': 'doubling',
                'Random': 'stochastic_action',
            }
                 
            pbar = tqdm(heuristics_map.items())
            for name, method_name in pbar:
                pbar.set_description(f"Agent VS {name}")
                for i in range(n_episodes):
                    env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                    heuristic = Strategies(env) 
                    steps = 0
                    done = False
                    
                    while not done and steps < max_steps:
                        if method_name == 'Frontier':
                            action_str = heuristic.create_and_propagate()
                        elif method_name == 'swap_asap':
                            action_str = heuristic.swap_asap()
                        elif method_name == 'FN_swap':
                            action_str = heuristic.FN_swap()
                        elif method_name == 'SN_swap':
                            action_str = heuristic.SN_swap()
                        elif method_name == 'doubling':
                            action_str = heuristic.doubling_swap()
                        else:
                            action_str = heuristic.stochastic_action()
                        
                        # RECORD ACTION (Only for first episode)
                        if plot_actions and i == 0:
                            if action_str:
                                label = parse_action_label(action_str, is_agent=False)
                                action_timeline[name].append(label)

                        try:
                            if action_str:
                                # HACK: execute heuristic string on local env
                                exec(action_str.replace("self.", "env."))
                            
                            current_fid = env.getLink((0, env.n-1), 1)
                            is_success = env.endToEndCheck(timeToWait=0)

                            if is_success:
                                done = True
                                results[name]['fidelities'].append(current_fid)
                                if plot_actions and i == 0:
                                    action_timeline[name].append("Done")
                        except:
                            pass
                        steps += 1
                    results[name]['steps'].append(steps if done else max_steps)

            # Generate Plot if requested
            if plot_actions:
                plot_action_timeline(action_timeline)

            # Calculate Baseline (Agent) Averages first
            agent_steps_avg = np.mean(results['Agent']['steps'])
            if len(results['Agent']['fidelities']) > 0:
                agent_fid_avg = np.mean(results['Agent']['fidelities'])
            else:
                agent_fid_avg = 0.0

            log("=" * 70)
            # Updated Header for Ratios
            log(f"{'Strategy':<12} | {'Avg Steps (std)':<19} | {'Avg Fidelity (std)':<17} | {'S%':<4} | {'F%':<5}")
            log("-" * 70)
            
            for strategy, data in results.items():
                avg_steps = np.mean(data['steps'])
                std_steps = np.std(data['steps'])
                
                # Average fidelity (only for successful runs)
                if len(data['fidelities']) > 0:
                    avg_fid = np.mean(data['fidelities'])
                    std_fid = np.std(data['fidelities'])
                else:
                    avg_fid, std_fid = 0.0, 0.0
                
                # --- RATIO CALCULATION ---
                step_ratio = (avg_steps / agent_steps_avg) * 100 if agent_steps_avg > 0 else 0.0
                
                if agent_fid_avg > 1e-9:
                    fid_ratio = (avg_fid / agent_fid_avg) * 100
                elif avg_fid > 1e-9: 
                    fid_ratio = float('inf') 
                else:
                    fid_ratio = 100.0 
                
                pm = u"\u00B1"
                log(f"{strategy:<12} | {avg_steps:<9.2f} ({pm}{std_steps:<6.2f}) | {avg_fid:<8.4f} ({pm}{std_fid:.4f}) | {f'{step_ratio:.0f}%':<4} | {f'{fid_ratio:.0f}%':<5}")