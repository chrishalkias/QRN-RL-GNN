# -*- coding: utf-8 -*-
# test.py

'''
Created Wed 22 Apr 2025
This is a unittesting file
'''

import sys
from pathlib import Path
import unittest
sys.path.append(str(Path(__file__).parent.parent))
from src.repeaters import RepeaterNetwork as rn
from src.gym_env import QuantumNetworkEnv as qenv
from src.gym_env import Experiment as exp
from src.agent import AgentDQN as agent

class TestReepater(unittest.TestCase):
    """Test the Quantum repeater network class"""
    def setUp(self):
        self.renet = rn
        return super().setUp()
    
    def test_nesting(self):
        for n in range(1,100,1):
            renet = self.renet(n)
            self.assertEqual(renet.n, n)
    def test_deterministic_swap_asap(self):
        for n in range(100):
            for p_entangle in range(0,1,10):
                for p_swap in range(0,1,10):
                    n=10
                    net=rn(n=n, p_entangle=p_entangle, p_swap=p_swap)
                    initialMatrix = net.matrix
                    net.endToEndCheck()
                    self.assertFalse(net.global_state)


    def test_state_dict(self):
        return self.assertTrue(type(self.renet().matrix) == dict)

    def tearDown(self):
        return super().tearDown()

class TestAgent(unittest.TestCase):
    """Test the Agent class"""
    pass

class TestModel(unittest.TestCase):
    """Test the neural networks"""
    pass

if __name__ == '__main__':
    unittest.main()
