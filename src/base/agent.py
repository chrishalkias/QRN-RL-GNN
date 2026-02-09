#@title Agent GNN
import numpy as np
from base.repeaters import RepeaterNetwork
from base.strategies import Heuristics
from torch_geometric.data import Batch
from base.model import GNN
from base.buffer import Buffer
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.data import Data
np.set_printoptions(legacy='1.25')



class AgentGNN(RepeaterNetwork):

  def __init__(self,
               n=4,
               directed = False,
               geometry = 'chain',
               tau = 1_000,
               cutoff = None,
               p_entangle = 1,
               p_swap = 1,
               lr=0.005,
               gamma=0.93,
               epsilon=1):
    
    """
    Implements the RL agent used to learn strategies on the Repeater Network.
    It also includes the heuristics tested against.

    Composed from: RepeaterNetwork

    Methods:
      get_state_vector()
      get_valid_actions()
      choose_action()
      update_environment()
      reward()
      train()
      saveModel()
      trained_action()
      random_action()
      alternating_action()
      swap_asap()
      test()
    """

    super().__init__(n, cutoff, tau, p_entangle, p_swap)
    self.lr=lr
    self.gamma = gamma
    self.epsilon = epsilon
    self.criterion = nn.MSELoss()
    states = len(self.get_state_vector())
    actions = len(self.all_actions())
    self.model = GNN()
    self.target_model = type(self.model)()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.memory = Buffer(max_size = 10_000)

  def reinitialize(self, n=None, tau=None, cutoff=None, p_entangle=None, p_swap=None) ->None:
    """Reinitialize the agent with different parameters"""
    new_n = n if n is not None else self.n
    new_tau = tau if tau is not None else self.tau
    new_p_entangle = p_entangle if p_entangle is not None else self.p_entangle
    new_p_swap = p_swap if p_swap is not None else self.p_swap
    new_cutoff = cutoff if cutoff is not None else self.cutoff
    super().__init__(
        n=new_n, 
        tau=new_tau, 
        cutoff=new_cutoff, 
        p_entangle=new_p_entangle, 
        p_swap=new_p_swap
    )


  def get_state_vector(self) -> Data:
    return self.tensorState() 

  def get_valid_actions(self) -> list:
    """
    If the repeater is the last of the chain it dissalows EG to the right.
    Checks if there is a left and a right connection for each repeater.
    If both exists, it allows swap on that repeater.
    Appends an even number (node*2) for EG and an odd one (node*2 +1) for swaps

    Returns:
      valid_actions (list) : 
    """
    valid_actions = []
    for node in range(self.n):
        if node != self.n - 1:  # Can entangle right
            valid_actions.append(node * 2)

        if 0 < node < self.n - 1:  # Is intermediate node
          linked_left, linked_right = False, False #init to false
          for other_node in range(self.n): #has 2 connections
            if node < other_node:
              linked_left = (self.getLink((node, other_node), 1) > 0)
            elif node > other_node:
              linked_right = (self.getLink((other_node, node), 1) > 0)

            if linked_left and linked_right:
              valid_actions.append(node * 2 + 1)
    return valid_actions

  def choose_action(self, use_trained_model=False) -> int:
    """Epsilon-greedy action choice"""
    if not use_trained_model and random.uniform(0, 1) < self.epsilon:
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions)

    with torch.no_grad():
        q_values = self.target_model(self.get_state_vector()).flatten()

        # Create a mask for valid actions
        mask = torch.full_like(q_values, float('-inf'))
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = 0.0

        # Apply mask and use softmax to ensure numerical stability
        masked_q_values = q_values + mask
        action_probs = torch.softmax(masked_q_values, dim=0)

        if use_trained_model:
           return torch.argmax(action_probs).item()
        else: # TODO: check if this actually helps the agent learn or not
        # Select action with highest probability (equivalent to argmax but safer)
          return torch.multinomial(action_probs, 1).item()

  def update_environment(self, action) -> float:
        """ Apply action"""
        action_string = self.all_actions()[action]
        exec(action_string)
        return self.reward()

  def reward(self) -> float:
    return 1 if self.endToEndCheck() else -0.01 

  def get_valid_mask(self) -> torch.Tensor:
        """
        Converts the list of valid actions into a tensor mask 
        where invalid actions are -inf and valid actions are 0.0.
        """
        # Total possible actions = 2 per node (rntangle, swap)
        total_actions = self.n * 2 
        mask = torch.full((total_actions,), float('-inf'))
        
        valid_indices = self.get_valid_actions()
        if valid_indices:
            mask[valid_indices] = 0.0
            
        return mask
  
  def step(self):
    """
    Perform one observation-action-reward-store step.
    Store the observation in the buffer
    > To be fed into `Q-estimate`

    Returns: The (S, A, R, S', D) tuple
    """
    state = self.get_state_vector()
    action = self.choose_action()
    reward = self.update_environment(action)
    done = self.endToEndCheck()
    next_state = self.get_state_vector()

    # Calculate the mask for the state we just arrived in
    next_mask = self.get_valid_mask()

    self.memory.add(state=state, 
            action=action, 
            reward=reward, 
            next_state=next_state,
            next_mask=next_mask)
    
    return done

#=========================================================================================================

  def Q_estimate(self, state_batch, action_batch, reward_batch, next_state_batch, next_mask_batch):
      """
      Calculates Q-values and Targets for a BATCH of data.
      """
      BATCH_SIZE = len(reward_batch)

      #--------Get Q-values for current states
      q_out = self.model(state_batch)  # Shape: [BATCH_SIZE * self.n, 2]
      
      # Reshape to [Batch, Nodes, 2] (batch indexing)
      q_out_reshaped = q_out.view(BATCH_SIZE, self.n, 2)
      
      # Flatten to [Batch, Total_Actions] (Total_Actions = Nodes * 2)
      # The layout is: [e0, s0, e1, s1...]
      q_flat = q_out_reshaped.view(BATCH_SIZE, -1)

      # Gather the specific Q-value for the action taken in each batch item
      q_value = q_flat.gather(1, action_batch.unsqueeze(1)).squeeze(1) # Shape: [Batch, 1]

      #--------Calculate Target Q-values
      with torch.no_grad():
        # online to select
        next_q_online = self.model(next_state_batch).view(BATCH_SIZE, -1)
        masked_next_q_online = next_q_online + next_mask_batch
        best_actions = torch.argmax(masked_next_q_online, dim=1)

        # target to eval
        next_q_target = self.target_model(next_state_batch).view(BATCH_SIZE, -1)
        q_target_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        target = reward_batch + self.gamma * q_target_values

      return q_value, target
  
#=========================================================================================================

  def stacker(self, samples):
    """Manual stacking of the data into Batches"""

    # a. Create Graph Batches for states
    state_list = [item['s'] for item in samples]
    next_state_list = [item['s_'] for item in samples]
    state_batch = Batch.from_data_list(state_list)
    next_state_batch = Batch.from_data_list(next_state_list)

    # b. Stack simple tensors
    # torch.stack creates [Batch, ...]
    action_batch = torch.tensor([item['a'] for item in samples], dtype=torch.long)
    reward_batch = torch.tensor([item['r'] for item in samples], dtype=torch.float)
    mask_batch = torch.stack([item['m_'] for item in samples]) # Shape [Batch, Action_Size]
    avg_batch_reward = reward_batch.mean().item()
    return state_batch, action_batch, reward_batch, next_state_batch, mask_batch, avg_batch_reward


  def backprop(self, q_value, target):
    """Performs the backwards pass"""
    loss = self.criterion(q_value, target)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # failsafe prevents "exploding gradient" crashes
    self.optimizer.step()
    return loss

    
  def train(self, 
            episodes=1000,
            batch_size=64, 
            plot=True, 
            jitter = None,
            n_range = [4,6],
            save_model=True, 
            savefig=True) -> list:
    
    """Trains the agent"""
    self.reset()
    total_links, total_batch_reward, rewardList = 0, 0, []

    for step in tqdm(range(episodes)):
      
      #jitter condition
      if jitter and not step%jitter:
          new_n = np.random.choice(n_range)
          if new_n != self.n: # clear the buffer (cannot stack observations of different size)
            self.memory.clear()
          self.reinitialize(n=new_n)
          

      done = self.step() # run one step and store obs in buffer
      total_links += 1 if done else 0

      if self.memory.size() > batch_size:
          
        # Sample TensorDicts
        samples = self.memory.sample(batch_size)
        # Stack samples
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, avg_batch_reward = self.stacker(samples)
        #run bellman
        q_value, target = self.Q_estimate(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        # backprop
        loss = self.backprop(q_value, target)

        # Decay epsilon
        self.epsilon = max(0.05, 1 - (step / episodes) * (1-0.05)) # decay from e=1 to e=0.05

        # Polyak Averaging: soft update (1% transfer per step)
        tau_update = 0.01 
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
          target_param.data.copy_(tau_update * local_param.data + (1.0 - tau_update) * target_param.data)

        total_batch_reward = avg_batch_reward
        rewardList.append(total_batch_reward)
      
    if plot:
      plt.plot(rewardList,ls='-', label='Average batch-reward')
      plt.title(f'Training metrics over {episodes} steps')
      plt.xlabel('Episode')
      plt.ylabel(f'Reward for $(n, p_E, p_S)$= {n_range, self.p_entangle, self.p_swap}')
      plt.yscale("symlog")
      plt.legend()
      plt.savefig('assets/train.png') if savefig else None
      plt.show()

    torch.save(self.model.state_dict(), 'assets/gnn_model.pth') if save_model else None
    print(total_links)
    return rewardList

  
  def test(self, 
           n_test=5, 
           p_entangle=1, 
           p_swap=1, 
           tau=1000, 
           cutoff = 10_000,
           max_steps=100, 
           verbose=False, 
           kind='trained') -> list:
    
    """
    
    Perform the validation of the agent against the heuristic strategies
    
    """
    links_established, linkList = 0, []

    self.reinitialize(n=n_test, tau=tau, cutoff=cutoff, p_entangle=p_entangle, p_swap=p_swap)
    strategies = Heuristics(self)

    for _ in tqdm(range(max_steps)):

      if kind=='trained':
          action = self.choose_action(use_trained_model=True)
          self.update_environment(action)
      elif kind=='swap_asap':
          action = strategies.swap_asap()
          exec(action)
      elif kind=='random':
          action = strategies.stochastic_action()
          exec(action)
      

         
      wincon = self.getLink((0,self.n-1)) > 0

      if wincon:
        links_established +=1 
        self.reset() 

      linkList.append(links_established)

    return linkList
  

  def action_to_string(self, action):
    """DEBUGGING"""
    node_index = action // 2
    action_type = action % 2 

    if action_type == 0:
        if node_index < self.n - 1: #kinda redundant since mask handles this
            return f'self.entangle(({node_index}, {node_index + 1}))'
            
    elif action_type == 1:
        return f'self.swapAT({node_index})'