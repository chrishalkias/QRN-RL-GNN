import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import re
import time
import wandb
import math

# Import locals
from base.repeaters import RepeaterNetwork
from base.buffer import Buffer
from base.model import GNN
from base.strategies import Strategies



class Binder:
        def __init__(self):
            """Decouples the physical environment from the agent."""
            ...

        @staticmethod
        def get_valid_actions_mask(state: Data, device) -> torch.Tensor:
            """
            Returns the action mask for each state.
            
            Entangle: Valid for nodes WITHOUT existing right connections (except last node)
            Swap: Valid for nodes WITH both left AND right connections (except endpoints)
            """
            if not hasattr(state, 'ptr'):
                # Handle single unbatched state
                ptr = torch.tensor([0, state.x.shape[0]], device=device)
            else:
                ptr = state.ptr

            num_nodes_total = state.x.shape[0]
            mask = torch.zeros((num_nodes_total, 2), dtype=torch.bool, device=device)

            first_nodes = ptr[:-1]
            last_nodes = ptr[1:] - 1

            # Extract connection indicators
            has_left = state.x[:, 0] > 0
            has_right = state.x[:, 1] > 0

            # Entangle: valid for nodes WITHOUT right connections, except last node
            valid_entangle = ~has_right
            valid_entangle[last_nodes] = False
            mask[:, 0] = valid_entangle

            # Swap: requires BOTH left AND right connections, not at endpoints
            valid_swap = has_left & has_right
            valid_swap[first_nodes] = False
            valid_swap[last_nodes] = False
            mask[:, 1] = valid_swap

            return mask.view(-1)  # 1D mask aligning with flattened Q-values
        
        @staticmethod
        def _decode_action(action_idx: int)-> tuple:
            """
            ## Action_idx -> (node, bool)

            ###Decription:
                Decodes flattened action index back to (node, operation)
            """
            node = action_idx // 2
            op_type = action_idx % 2 # 0 = Entangle, 1 = Swap
            return node, op_type
        
        @staticmethod
        def reward() -> float:
            """
            Defines the succsess reward and time penalty
            """
            step_cost = -1
            success_reward = 100
            info = {'fidelity': 0.0}
            return step_cost, success_reward, info
    
        @classmethod
        def step_environment(self, env: RepeaterNetwork, action_idx: int) -> tuple: 
            """
            Executes the (integer) action on the environment.
            
            Args:
                env: RepeaterNetwork instance
                action_idx: Integer encoded action
                
            Returns:
                tuple: (reward, done, info)
            """
            node, op_type = self._decode_action(action_idx)
            step_cost, success_reward, info = self.reward()

            # Execute Action
            if op_type == 0: 
                env.entangle(edge=(node, node+1))
            elif op_type == 1: 
                env.swapAT(node)

            # Check for success FIRST, which returns fidelity before zeroing
            # (Requires modifying endToEndCheck to return fidelity - see repeaters fix)
            is_success, current_fidelity = env.endToEndCheck(timeToWait=0)

            if is_success:
                info['fidelity'] = current_fidelity
                return success_reward, True, info
            
            return step_cost, False, info
        

class QRNAgent:
    def __init__(self, 
                 lr=5e-4, 
                 gamma=0.95,
                 buffer_size=50_000,
                 epsilon=1.0,
                 batch_size=128,
                 target_update_freq=500):
        
        #TODO decouple the agent from the action selection
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.lr=lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0


        # Models
        self.policy_net = GNN(node_dim=2, output_dim=2).to(self.device)
        self.target_net = GNN(node_dim=2, output_dim=2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Buffer
        self.memory = Buffer(max_size=buffer_size)

    def select_action(self, state:Data,
                      training:bool=True
                      ) -> int:
            """
            ## Q-value -> Action_idx
            ### Description:
                Selects an action based on the q-values from `self.policy_net()`.
                Applies a mask to illegal actions via `self.get_valid_actions_mask()`

            ### Returns:
                The integer-encoded action via the `tensor.argmax()` function
            """
            mask = Binder.get_valid_actions_mask(state, self.device)
            
            self.policy_net.eval()
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                q_flat = q_values.view(-1).clone() 
                
                # Apply 1D mask
                q_flat[~mask] = -float('inf')
                # max_q = q_flat.max().item() #In case peeking into the q-val is needed
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
    

    def train_step(self):
        """            
            1. Sample from the Buffer and create batches
            2. Get the model's Q-values
            3. Offset batch to match Q-values
            4. Compute next Q-values, mask them and compute target Q
            5. Perform a backwards pass and step the optimizer
            6. Copy the params to target net (if step condition matches)
            """
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        state_batch_list = [x['s'] for x in batch]
        next_state_batch_list = [x['s_'] for x in batch]

        state_batch = Batch.from_data_list(state_batch_list).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_batch_list).to(self.device)

        
        action_batch = torch.tensor([x['a'] for x in batch], device=self.device)
        reward_batch = torch.tensor([x['r'] for x in batch], device=self.device).float()
        done_batch = torch.tensor([x['d'] for x in batch], device=self.device).float()

        # --- Current Q Values ---
        q_values_flat = self.policy_net(state_batch).view(-1)

        # Dynamic batch offset using graph boundaries (ptr)
        batch_offset = state_batch.ptr[:-1] * 2
        global_action_indices = action_batch + batch_offset

        current_q = q_values_flat[global_action_indices]

        # --- Target Q Values ---
        with torch.no_grad():
            next_q_values_flat = self.target_net(next_state_batch).view(-1)

            # Returns a 1D mask of size [Total_Nodes * 2]
            mask = Binder.get_valid_actions_mask(next_state_batch, self.device)

            # Apply 1D mask
            next_q_values_flat[~mask] = -float('inf')

            # Extract the max action value per node. Shape: [Total_Nodes]
            next_q_nodes = next_q_values_flat.view(-1, 2)
            max_q_per_node = next_q_nodes.max(dim=1)[0] 

            # Pool the maximum node value per graph to align with the batch. Shape: [batch_size]
            max_next_q = global_max_pool(max_q_per_node, next_state_batch.batch)
            
            # Mask the target Q value if the state is terminal
            target_q = reward_batch + (self.gamma * max_next_q * (1 - done_batch))

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    def get_curriculum_n_range(self, episode: int, total_episodes: int) -> list:
        """
        Progressive curriculum: start small, gradually increase size range.
        
        Phase 1 (0-25%): Train on small networks (n=3,4)
        Phase 2 (25-50%): Medium networks (n=4,5,6)
        Phase 3 (50-75%): Larger networks (n=5,6,7,8)
        Phase 4 (75-100%): Full range (n=4,5,6,7,8,9,10)
        """
        progress = episode / total_episodes
        
        if progress < 0.25:
            return [3, 4]
        elif progress < 0.50:
            return [4, 5, 6]
        elif progress < 0.75:
            return [5, 6, 7, 8]
        else:
            return [4, 5, 6, 7, 8, 9, 10]


    def train(self, 
              episodes=500, 
              max_steps=50, 
              savemodel=False, 
              plot=False, 
              curriculum=False, 
              jitter=100,
              n_range=[4,6],
              fine_tune=False,
              p_e=1.0, 
              p_s=1.0,
              tau=1000, # TODO tau
              cutoff=1000,
              use_wandb=True,
              wandb_project = "QRN-RL"):
        
        """
        ## The main trainning loop

        Trains the agent. Performs the following algorithm

        Args:
        
            episodes:
                The maximum number of episodes to train for
                generally somewhere in the range of 1000-50.000.
            max_steps:
                The maximum number of steps until episode termination.
                The agent is givem this amount of possible operations 
                until the episode terminates and the systems restarts.
                savemodel 
                Save the model .pth file to be used for validation and diagnostics
            plot:
                Plot the reward curves for training (redundand if using wandb)
            savefig:
                Save the figure produced by plot
            jitter:
                If non-zero, specifies the number of episodes for which the system
                is reinitialized to a new value of n (chosen at random from n_rang).
                If the new_n is different from the old, the Buffer is reinitialized.
            n_range:
                The range for which to choose n's from. If `jitter=0` then the first
                element of the list is always passed as the n to be trained on.
            fine_tune:
                If true, the parameters of the agent are changed so that the training run
                counts as fine tuning. `self.lr` is lower, `self.epsilon` starts from a
                smaller value and the agent expects a trained dict to be uploaded into *both*
                the `self.policy_net` and `self.value_net`.
            p_e, p_s, tau, cutoff:
                The training env parameters
            use_wandb:
                If true, pushes the run to wandb to monitor the training performance, model
                weights and compare with other training runs.
            wandb_project:
                The name of the wandb project
        """
        

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
        
        scores, losses = [], []
        pbar = tqdm(range(episodes))
        n_nodes = random.choice(n_range)
        eps_init = 1.0 if not fine_tune else 0.2
        eps_fin = 0.05 if not fine_tune else 0.01

        if fine_tune: #load buffer for fine tuning # REVIEW to check the logic
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5, eps=1e-4)
            env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
            state = env.tensorState()
            for _ in range(self.batch_size * 5):
                action = Binder.select_action(state, training=True)
                r, done, info = Binder.step_environment(env, action)
                next_state = env.tensorState()
                self.memory.add(state, action, r, next_state, done)
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

                if jitter and not e % jitter: # REVIEW check the jitter logic
                    if curriculum:
                        new_n = np.random.choice(self.get_curriculum_n_range(episode = e, total_episodes=episodes))
                    else:
                        new_n = np.random.choice(n_range)
                    if new_n != n_nodes:
                        self.memory.clear()
                        if use_wandb:
                            wandb.log({"System/New_N": new_n})
                    new_env = RepeaterNetwork(n=new_n, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                    env = new_env
                    n_nodes = new_n
                    state = env.tensorState()

                for _ in range(max_steps):
                    action_idx = self.select_action(state, training=True)
                    if action_idx % 2 == 0: # HACK This has to be pulled to the Binder somehow
                        entangles += 1
                    else:
                        swaps +=1
                    
                    reward, done, info = Binder.step_environment(env, action_idx)
                    next_state = env.tensorState()
                    
                    self.memory.add(state=state, 
                                    action=action_idx, 
                                    reward=reward, 
                                    next_state=next_state,
                                    done=done)
                    
                    state = next_state
                    score += reward
                    steps += 1
                    
                    loss = self.train_step()
                    
                    if done:
                        if info.get('fidelity', 0) > 0:
                            success = steps
                        break

                total_steps += steps
                if not fine_tune: # QUESTION Is there a better choice here?
                    self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (1 + math.cos(math.pi * e / episodes))
                elif fine_tune:
                    self.epsilon = 0.2 * np.exp(-10 * e / episodes)

                scores.append(score)
                losses.append(0 if loss == None else loss.detach().item())
                pbar.set_description(f"Ep {e+1} | Score: {score:.1f} | Eps: {self.epsilon:.2f}")

                episode_time = time.time() - start_time
                steps_per_sec = steps / episode_time if episode_time > 0 else 0
                action_total = swaps + entangles
                swap_ratio = swaps / action_total if action_total > 0 else 0

                if use_wandb:
                    wandb.log({
                        "Performance/Reward": score,
                        "Performance/Loss": loss,
                        "Performance/Success_Rate": success, # W&B can smooth this
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
            average_reward = TestHelpers.running_average(scores, 10)
            average_loss = TestHelpers.running_average(losses, 10)

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(scores, 'b', ls='-', label='Batch-reward', alpha=0.3)
            ax1.plot(average_reward, 'b', ls='-', label='Running average')
            ax1.set_title(f'Average batch reward')
            ax1.set_xlabel('Episode')
            n_label = f'{n_range[0]}-{n_range[-1]}' if jitter else n_range[0]
            ax1.legend()

            ax2.plot(losses, 'r', ls='-', label='Batch-loss', alpha=0.3)
            ax2.plot(average_loss, 'r', ls='-', label='Running average')
            ax2.set_xlabel('Episode')
            ax2.set_title(f'Average Batch loss')
            ax2.set_yscale('log')
            # ax2.grid(axis='y', which='both', linewidth='0.5')
            ax2.legend()
            label = f'{"Training" if not fine_tune else "Fine-tuning"} metrics for {r'$n, p_E, p_S, T, c$'}= {n_label}, {p_e}, {p_s}, {tau}, {cutoff}'
            fig.suptitle(label)
            fig.tight_layout()
            plt.show()
            os.makedirs(f'assets/trained_models/{model_name}/', exist_ok=True)
            fig.savefig(f'assets/trained_models/{model_name}/train.svg')

        if use_wandb:
            wandb.finish()

    def validate(self, 
                    dict_dir=None,
                    n_episodes=100, 
                    max_steps=100, 
                    n_nodes=4, 
                    p_e=1.0, 
                    p_s=1.0,
                    tau=1000, # NOTE tau
                    cutoff=None, 
                    logging=True,
                    plot_actions=True,
                    savefig=True):
            
            """
            The main testing loop
            The agent is tested for its performance (avg steps) and (avg fidelity) on a 
            quantum repeater network from the `RepeaterNetwork` class

        Args:
            dict_fir:
                The directory of the trained model dict to be used for validation.
            n_episodes: 
                The number of testing episodes. It terminates if either the end-to-end
            state is reached or if `max_steps` is reached.
            n_nodes:
                The number of testing nodes. This can be different than the number of 
            nodes used in the training loop.
            p_e, p_s, tau, cutoff: 
                The testing parameters. 
            loging:
                If true, logs the results of the validation (avg/std steps, avg/std fidelity).
            plot_actions:
                If true plots the actions of the **first episode** of the validation run. This
                is used to interpret the agents actions. Colored blocks are utilized to differentiate
                the actions of the agent (and the strategies) used.
                savefig. Only used in conjunction with `plot_actions=True`. Saves the resulting plot
        """
            
            # REVIEW check this function again
            # TODO Tidy this function up

            def log(msg):
                """Used to print on STDOUT and log to file"""
                print(msg)
                if logging:
                    with open("./assets/validation_results/validation_results.txt", "a") as f:
                        f.write(msg + "\n")
            


            log(f"\n--- Validation (N:{n_nodes}, Pe:{p_e}, Ps:{p_s}, tau:{tau}, cutoff:{cutoff}) ---")
            log(f"--- Max_steps: {max_steps}, n_episodes:{n_episodes}, Model: {dict_dir.split('/')[-1][:-4]} ---")
            
            results = {
                'Agent': {'steps': [], 'fidelities': []},
                'Frontier': {'steps': [], 'fidelities': []},
                'FN_Swap': {'steps': [], 'fidelities': []},
                'SN_Swap': {'steps': [], 'fidelities': []},
                'SwapASAP': {'steps': [], 'fidelities': []},
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
                    action_idx = self.select_action(state, training=False)
                    
                    # RECORD ACTION (Only for first episode)
                    if plot_actions and i == 0:
                        label = TestHelpers.parse_action_label(action_idx, is_agent=True)
                        action_timeline['Agent'].append(label)

                    _, done, info = Binder.step_environment(env, action_idx)
                    
                    if done:
                        results['Agent']['fidelities'].append(info['fidelity'])
                        if plot_actions and i == 0:
                            action_timeline['Agent'].append("Done")
                    
                    steps += 1
                results['Agent']['steps'].append(steps if done else max_steps)
                
            # --- Test Heuristics ---
            heuristic_methods = [key for key in results.keys() if key != 'Agent']
            pbar = tqdm(heuristic_methods)
            for method_name in pbar:
                pbar.set_description(f"Agent VS {method_name}")
                for i in range(n_episodes):
                    env = RepeaterNetwork(n=n_nodes, p_entangle=p_e, p_swap=p_s, tau=tau, cutoff=cutoff)
                    heuristic = Strategies(env) 
                    steps = 0
                    done = False
                    
                    while not done and steps < max_steps:
                        if method_name == 'Frontier':
                            action_str = heuristic.frontier()
                        elif method_name == 'SwapASAP':
                            action_str = heuristic.swap_asap()
                        elif method_name == 'FN_Swap':
                            action_str = heuristic.FN_swap()
                        elif method_name == 'SN_Swap':
                            action_str = heuristic.SN_swap()
                        elif method_name == 'doubling':
                            action_str = heuristic.doubling_swap()
                        else:
                            action_str = heuristic.stochastic_action()
                        
                        # RECORD ACTION (Only for first episode)
                        if plot_actions and i == 0:
                            if action_str:
                                label = TestHelpers.parse_action_label(action_str, is_agent=False)
                                action_timeline[method_name].append(label)

                        try:
                            if action_str:
                                exec(action_str.replace("self.", "env.")) # HACK: execute heuristic string on local env
                            
                            is_success, current_fid = env.endToEndCheck(timeToWait=0)

                            if is_success:
                                done = True
                                results[method_name]['fidelities'].append(current_fid)
                                if plot_actions and i == 0:
                                    action_timeline[method_name].append("Done")
                        except:
                            raise RuntimeError(f's action execution went wrong')
                        
                        steps += 1
                    results[method_name]['steps'].append(steps if done else max_steps)

            # Generate Plot if required
            if plot_actions:
                TestHelpers.plot_action_timeline(action_timeline, n_nodes, p_e, p_s, tau, cutoff, dict_dir, savefig)

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











class TestHelpers:
    def __init__(self):
        """
        A plotting and an action parser function that aid in validation.
        Separated here to keep the validation logic intact
        """
        pass


    # --- Helper: Plotting Function ---
    @staticmethod
    def plot_action_timeline(action_history, n_nodes, p_e, p_s, tau, cutoff, dict_dir, savefig):
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
        save_dir =  dict_dir.rsplit('/', 1)[0] + '/'
        plt.savefig(save_dir + 'validation.png') if savefig else None
        plt.show()


    # --- Helper: Action Parser for Plotting ---
    @staticmethod
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
        
    # --- Helper: Running average calculator ---
    @staticmethod
    def running_average(list, window):
            running_average = []
            for ind in range(len(list) - window + 1):
                running_average.append(np.mean(list[ind:ind+window]))
            return running_average

