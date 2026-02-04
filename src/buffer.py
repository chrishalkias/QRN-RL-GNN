from torchrl.data import ReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
import numpy as np
from repeaters import RepeaterNetwork
from agent import AgentGNN
import random
from tensordict import TensorDict
import torch



class ExperienceBuffer():
    def __init__(self, max_size):
        """
        Implements a simple list replay buffer for the agent

        Methods:
            size()
            add()
            sample()
        """
        self.max_size = max_size
        self.buffer_list = ReplayBuffer(storage=ListStorage(max_size), 
                           collate_fn=lambda x: x)
        
    def size(self) -> int:
        return len(self.buffer_list)

    def add(self, state, action, reward, next_state) -> None:
        data = TensorDict({
            's' : state,
            'a' : action,
            'r' : reward,
            's_' : next_state})
        self.buffer_list.add(data)

    def sample(self, samples = 1) -> list:
        return self.buffer_list.sample(samples)





if __name__ == '__main__':
    agent = AgentGNN(n=4)
    buffer = ExperienceBuffer(max_size = 10_000)

    # collect data
    state = agent.get_state_vector()
    action = agent.choose_action()
    reward = agent.update_environment(action)
    next_state = agent.get_state_vector()

    buffer.add(state = state, 
            action=action, 
            reward=reward, 
            next_state=next_state)

    x = buffer.sample(3)

    # print the (S,A,R) tuple
    print(x[0]['s'])
    print(x[0]['a'])
    print(x[0]['r'])
    print(x[0]['s_'])

    # pseudo algorithm implementation
    step = int()
    k = int()
    s,a,r,s_ = None, None, None, None
    if step < buffer.max_size:
        buffer.add(s,a,r,s_)
    else:
        buffer.sample(samples=k)