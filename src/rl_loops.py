# -*- coding: utf-8 -*-
# src/rl_loops.py

'''
             ████████ ██████   █████  ██ ███    ██ ███████ ██████  
                ██    ██   ██ ██   ██ ██ ████   ██ ██      ██   ██ 
                ██    ██████  ███████ ██ ██ ██  ██ █████   ██████  
                ██    ██   ██ ██   ██ ██ ██  ██ ██ ██      ██   ██ 
                ██    ██   ██ ██   ██ ██ ██   ████ ███████ ██   ██    

                            Created Sat 10 May 2025
                        Implements the different RL loops.
'''
import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from gnn_env import Environment
from torch.nn import functional as F


class QTrainer(Environment):
    
    def __init__(self, experiment: object):
        """
        The ~pokemon~ trainer class. Here all the RL loops are included, seperated from the
        environment class for clarity and debugging. This is done through class composition.

        Attributes:
            experiment (obj) : The experiment object

        Methods:
            saveModel (NaN) : Saves the model chackpoint files
            trainQ    (NaN) : Deployes the Q-learning algorithm
        """
        assert isinstance(experiment, Environment), f'Arg has to be an instance of an Environment'
        self.experiment = experiment


    def saveModel(self, filename="logs/model_checkpoints/GNN_model.pth"):
        """Saves the model"""
        model = self.experiment.model
        torch.save(model.state_dict(), filename)


    def trainQ_scalar(self, episodes=10_000, plot=True, save_model=True):
        """
        This is the bread and butter of this class. Here the main loop of state action reward is iterated
        and the optimizer is moving forward on the model. This implements a very simple Q-learning procedure
        controlled by self.gamma, self.epsilon. Since the model output is a tensor the Q-value is calculated
        by the max() of it, i.e the node action with the highest confidence

        Args:
            episodes   (int)  : The number of trainig steps to perform
            plot       (bool) : Choice for plot outputs
            save_model (bool) : Choice for model checkpoints save

        Returns:
            model       (.pth) : The model checkpoints
            train_plots (.png) : The training plots
        """
        exp = self.experiment
        (totalReward, fidelity, entanglementDegree) = (0,0,0)
        (rewardList, fidelityList, entanglementlist, lossList) = ([],[],[],[])
        start_time = time.time()

        for _ in tqdm(range(episodes)):

            # set the MDP
            state = exp.network.tensorState()
            output = exp.model(state)
            action = exp.choose_action(exp.network.globalActions(), 
                                       output=output, 
                                       temperature = exp.temperature,
                                       choose_one=True)
            q_value = torch.max(output)
            reward = exp.update_environment(action)
            next_state = exp.network.tensorState()
            next_q_value = torch.max(exp.model(next_state))
            
            # compute the target
            with torch.no_grad():
                target = reward + exp.gamma * next_q_value 

            # compute reward and backpropagate
            q_value = q_value #torch.max(output)
            loss = exp.criterion(q_value, target)
            exp.optimizer.zero_grad()
            loss.backward()
            exp.optimizer.step()
            exp.scheduler.step(reward)
            
            #append the metrics
            totalReward += reward
            rewardList.append(totalReward)
            fidelity += exp.network.getLink((0,exp.n-1),1)
            fidelityList.append(fidelity)
            linkList = [exp.network.getLink(node,1) for node in exp.network.matrix.keys()]
            lossList.append(loss.item())
            entanglementDegree = np.mean(linkList) /exp.n
            entanglementlist.append(entanglementDegree)
            exp.network.resetState() if reward == 1 else None

        train_time = time.time() - start_time
        with open("logs/information.txt", "a") as file:
            file.write(f'\n {" " *10} Training information (Q-learning) \n Trained for {train_time:.3f} sec performing {episodes} steps.\n')
        self.saveModel() if save_model else None
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({exp.n}, {exp.network.p_entangle}, {exp.network.p_swap}) over $10^{int(np.log10(episodes))}$ steps"
            ax1.plot(rewardList,ls='-', label='Cummulative reward')
            ax1.set(ylabel=f'Log reward')
            ax1.set_yscale("symlog")
            ax1.legend()

            fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
            ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
            ax2.set(ylabel=f'Fidelity of resulting links')
            ax2.set_xscale("log")
            ax2.legend()

            # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
            fig.suptitle(plot_title)
            plt.savefig('logs/plots/train_plots_reward.png')
            plt.xlabel('Episode')
            plt.legend()

            fig2, (ax12, ax22) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({exp.n}, {exp.network.p_entangle}, {exp.network.p_swap}) over $10^{int(np.log10(episodes))}$ steps"
            ax12.plot(lossList,ls=(0,(1,10)), label='Loss')
            ax12.set_xscale("log")
            ax12.set(ylabel=f'Loss')
            ax12.legend()

            fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
            ax22.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
            ax22.set_xscale("log")
            ax22.set(ylabel=f'Fidelity of resulting links')
            ax22.legend()

            # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
            fig2.suptitle(plot_title)
            plt.savefig('logs/plots/train_plots_loss.png')
            plt.xlabel('Episode')
            plt.legend()

    def trainQ_tensor(self, episodes=10_000, plot=True, save_model=True):
        """
        Same as scalar but now the MSE is calculated on the difference
        of the current and next q_value tensors.

        Args:
            episodes   (int)  : The number of trainig steps to perform
            plot       (bool) : Choice for plot outputs
            save_model (bool) : Choice for model checkpoints save

        Returns:
            model       (.pth) : The model checkpoints
            train_plots (.png) : The training plots
        """
        exp = self.experiment
        (totalReward, fidelity, entanglementDegree) = (0,0,0)
        (rewardList, fidelityList, entanglementlist, lossList) = ([],[],[],[])
        start_time = time.time()

        for _ in tqdm(range(episodes)):
            # set the MDP
            dims = 4 * exp.network.n
            state = exp.network.tensorState()
            output = exp.model(state)
            action = exp.choose_action(exp.network.globalActions(), 
                                    output, 
                                    temperature = exp.temperature)
            reward = exp.update_environment(action)
            q_values = output.view(-1, dims)[0]
            next_state = exp.network.tensorState()
            next_q_values = exp.model(next_state).view(-1, dims)[0]
            
            # compute the target
            with torch.no_grad():
                target = reward*torch.ones_like(next_q_values) + exp.gamma * next_q_values

            # compute reward and backpropagate
            loss = exp.criterion(q_values, target)
            exp.optimizer.zero_grad()
            loss.backward()
            exp.optimizer.step()
            exp.scheduler.step(reward)
            
            #append the metrics
            totalReward += reward
            rewardList.append(totalReward)
            fidelity += exp.network.getLink((0,exp.n-1),1)
            fidelityList.append(fidelity)
            linkList = [exp.network.getLink(node,1) for node in exp.network.matrix.keys()]
            lossList.append(loss.item())
            entanglementDegree = np.mean(linkList) /exp.n
            entanglementlist.append(entanglementDegree)
            exp.network.resetState() if reward == 1 else None

        train_time = time.time() - start_time
        with open("logs/information.txt", "a") as file:
            file.write(f'\n-Training information (Q-learning) \n Trained for {train_time:.3f} sec performing {episodes} steps.\n')
        self.saveModel() if save_model else None

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            plot_title = f"Training metrics for $(n, p_E, p_S)$= ({exp.n}, {exp.network.p_entangle}, {exp.network.p_swap}) over $10^{int(np.log10(episodes))}$ steps (DQN)"
            ax1.plot(rewardList,ls='-', label='Cummulative reward')
            ax1.set(ylabel=f'Log reward')
            ax1.set_yscale("symlog")
            ax1.legend()

            fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
            ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
            ax2.set(ylabel=f'Fidelity of resulting links')
            ax2.set_xscale("log")
            ax2.legend()

            # ax2.plot(entanglementlist*self.n,'tab:green', ls='-', label=r'Average Entanglement')
            fig.suptitle(plot_title)
            plt.savefig('logs/plots/(DQN)_train_reward.png')
            plt.xlabel('Episode')
            plt.legend()


    # def train_PPO(self, 
    #               num_steps: int= 2048, 
    #               epochs: int = 4, 
    #               batch_size: int = 64, 
    #               gamma: float = 0.99, 
    #               clip_epsilon: float = 0.2,
    #               plot: bool = True) -> None:
    #     """
    #     Implements the PPO algorithm from scratch.

    #     Args:
    #         num_steps    (int)   > The number of training steps
    #         epochs       (int)   > The total number of epochs
    #         batch_size   (int)   > The batch size
    #         gamma        (float) > The discount factor
    #         clip_epsilon (float) > The policy clipping
    #         plot         (int)   > If plot the metrics

    #     Returns:
    #         information        (.txt) > Appends the training information
    #         train_plot_loss    (.png) > Returns the loss and fidelity plots
    #         train_plots_reward (.png) > Returns the reward and fidelity plots
    #     """
        
    #     (fidelityList, losslist) = ([],[])
        
    #     exp = self.experiment
    #     states, actions, old_log_probs, rewards = [], [], [], []
    #     current_state = exp.network.tensorState()

    #     # Data collection phase
    #     print(f'Collecting data')
    #     for _ in tqdm(range(num_steps)):
    #         with torch.no_grad():
    #             output = exp.model(current_state)
    #             action_probs = F.softmax(output, dim=-1)
                
    #             # Create distribution and sample actions for each node
    #             dists = [torch.distributions.Categorical(logits=node_logits) 
    #                     for node_logits in output]
    #             action_labels = torch.stack([d.sample() for d in dists])
    #             log_probs = torch.stack([dists[i].log_prob(action_labels[i]) 
    #                         for i in range(exp.n)]).sum()

    #             # Convert to environment's action format
    #             action = exp.choose_action(
    #                                 exp.network.globalActions(), 
    #                                 action_probs, 
    #                                 use_trained_model=True,
    #                                 temperature=exp.temperature
    #                                 )

    #         reward = exp.update_environment(action)
            
    #         states.append(current_state)
    #         actions.append(action_labels)
    #         old_log_probs.append(log_probs)
    #         rewards.append(reward)
    #         current_state = exp.network.tensorState()

    #     # Convert to tensors
    #     actions = torch.stack(actions)  # [num_steps, n]
    #     old_log_probs = torch.stack(old_log_probs)  # [num_steps]

    #     # Calculate discounted returns
    #     returns = []
    #     discounted = 0
    #     for r in reversed(rewards):
    #         discounted = r + gamma * discounted
    #         returns.insert(0, discounted)
    #     returns = torch.tensor(returns, dtype=torch.float32)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    #     # Optimization loop
    #     print(f'Optimizing')
    #     for epoch in tqdm(range(epochs)):
    #         indices = torch.randperm(num_steps)
    #         for start in range(0, num_steps, batch_size):
    #             batch_idx = indices[start:start+batch_size]
    #             batch_states = [states[i] for i in batch_idx.tolist()]
    #             batch_actions = actions[batch_idx]
    #             batch_old_log_probs = old_log_probs[batch_idx]
    #             batch_returns = returns[batch_idx]

    #             # Get new policy probabilities
    #             current_log_probs = []
    #             for state in batch_states:
    #                 output = exp.model(state)
    #                 dists = [torch.distributions.Categorical(logits=node_logits)
    #                         for node_logits in output]
    #                 log_probs = torch.stack([dists[i].log_prob(batch_actions[:,i][j]) 
    #                                     for j, i in enumerate(range(exp.n))]).sum()
                    
    #                 current_log_probs.append(log_probs)
                    
    #             current_log_probs = torch.stack(current_log_probs)

    #             # Calculate policy loss
    #             ratio = (current_log_probs - batch_old_log_probs).exp()
    #             surr1 = ratio * batch_returns
    #             surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * batch_returns
    #             policy_loss = -torch.min(surr1, surr2).mean()
    #             losslist.append(policy_loss.item())

    #             # Update model
    #             exp.optimizer.zero_grad()
    #             policy_loss.backward()
    #             exp.optimizer.step()

    #     if plot:
    #         logstep = int(np.log10(epochs * num_steps / batch_size))
    #         params = (exp.n, exp.network.p_entangle, exp.network.p_swap)
    #         plot_title = f"Training metrics for $(n, p_E, p_S)$= {params} over $10^{logstep}$ steps (PPO)"
    #         plt.plot(losslist,ls='-', label='Loss')
    #         plt.ylabel('Policy Loss')
    #         plt.title(plot_title)
    #         plt.legend()
    #         plt.savefig("./logs/plots/(PPO)_PolicyLoss.png")
            
    #         # fig, (ax1, ax2) = plt.subplots(2, 1)
    #         # plot_title = f"Training metrics for $(n, p_E, p_S)$= ({exp.n}, {exp.network.p_entangle}, {exp.network.p_swap}) over $10^{int(np.log10(epochs))}$ epochs (PPO)"
    #         # ax1.plot(losslist,ls='-', label='Loss')
    #         # ax1.set(ylabel=f'Policy Loss')
    #         # ax1.set_yscale("symlog")
    #         # plt.savefig("./logs/plots/(PPO)_PolicyLoss.png")
    #         # ax1.legend()

    #         # fidelity_per_step = [val/(i+1) for i, val in enumerate(fidelityList)]
    #         # ax2.plot(fidelity_per_step, 'tab:green', ls='-', label='Average Fidelity')
    #         # ax2.set(ylabel=f'Fidelity of resulting links')
    #         # ax2.set_xscale("log")
    #         # ax2.legend()