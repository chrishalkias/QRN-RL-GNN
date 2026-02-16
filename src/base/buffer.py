from torchrl.data import ReplayBuffer
from torchrl.data import ListStorage
import numpy as np
from base.repeaters import RepeaterNetwork
import random
from tensordict import TensorDict
import torch



class Buffer():
    def __init__(self, max_size):
        """
        Implements a simple list replay buffer for the agent

        Methods:
            size()
            add()
            sample()
        """
        self.max_size = max_size # 5e4-1e6
        self.buffer_list = ReplayBuffer(storage=ListStorage(max_size), 
                           collate_fn=lambda x: x)
        

    def add(self, state, action, reward, next_state, done) -> None:
            """
            Adds a SARSM tuple to the buffer. Also stores the validity mask
            for the next state and the done flag to prevent hallucination.
            """
            data = TensorDict({
                's' : state,
                'a' : action,
                'r' : reward,
                's_' : next_state,
                'd'  : done
                })
            self.buffer_list.add(data)

    def sample(self, samples = 1) -> list:
        return self.buffer_list.sample(samples)

    def size(self) -> int:
        return len(self.buffer_list)
    
    def clear(self) -> None:
         self.buffer_list = ReplayBuffer(storage=ListStorage(self.max_size), 
                           collate_fn=lambda x: x)





