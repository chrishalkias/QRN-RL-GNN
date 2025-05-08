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
from src.repeaters import RepeaterNetwork
from src.mlp_gym_env import QuantumNetworkEnv
from src.cnn_environment import Environment

class TestRepater(unittest.TestCase):
    """Test the Quantum repeater network class"""
    def setUp(self):
        self.net = RepeaterNetwork
        return super().setUp()
    
    def test_nesting(self):
        """Test to see if num nodes is always consistent"""
        for n in range(1,100,1):
            net = self.net(n)
            data = net.tensorState()
            self.assertEqual(net.n, n)
            self.assertEqual(data.num_nodes, n)

    def test_deterministic_swap_asap(self):
        """Test links"""
        for n in range(100):
            for p_entangle in range(0,1,10):
                for p_swap in range(0,1,10):
                    n=10
                    net=self.net(n=n, p_entangle=p_entangle, p_swap=p_swap)
                    initialMatrix = net.matrix
                    net.endToEndCheck()
                    self.assertFalse(net.global_state)
                    for (i,j) in net.matrix.keys():
                        net.entangle(edge = (i,j))
                        self.assertEqual(net.matrix[(i,j)][1], 1)


    def test_state_dict(self):
        return self.assertTrue(type(self.net().matrix) == dict)

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
