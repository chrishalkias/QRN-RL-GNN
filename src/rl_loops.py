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
                                       output, 
                                       temperature = exp.temperature)
            q_value = exp.binary_tensor_to_int(exp.out_to_onehot(output))
            reward = exp.update_environment(action)
            next_state = exp.network.tensorState()
            next_q_value = exp.binary_tensor_to_int(exp.out_to_onehot(exp.model(next_state)))
            
            # compute the target
            with torch.no_grad():
                target = reward + exp.gamma * next_q_value #torch.argmax(exp.model(next_state))

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
            q_value = torch.max(q_values)
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

