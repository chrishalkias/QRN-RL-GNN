import unittest
import numpy as np
import random
from base.agent import AgentGNN
from base.model import GNN
from generators import *

class TestAgentGNN(unittest.TestCase):
    """
    Test AgentGNN class

    Includes:
        test_agent_initialization_parameters()
        test_get_valid_actions_different_sizes()
        test_choose_action_probabilistic()
        test_choose_action_probabilistic()
    """

    def setUp(self):
        self.num_inits = 10
        self.reward_consistency_checks = 20
        self.action_tests = 20
        self.n_range = range(3,20)
        self.epsilon_values = np.linspace(0,1, 20)
    
    def test_agent_initialization_parameters(self):
        """Test agent initialization with different parameters"""
        test_params = []
        for _ in range(self.num_inits):
            test_params.append(
                {'n': generateRandom_N(), 
                 'p_entangle': generateRandom_pe(), 
                 'p_swap': generateRandom_ps(), 
                 'tau': generateRandom_tau(),
                 'cutoff': bool(random.getrandbits(1)),
                 'lr': random.uniform(0.0005, 0.01),
                 'gamma': random.uniform(0.80, 0.99),
                 })
        
        for params in test_params:
            with self.subTest(**params):
                agent = AgentGNN(**params)
                self.assertEqual(agent.n, params['n'])
                self.assertEqual(agent.p_entangle, params['p_entangle'])
                self.assertEqual(agent.p_swap, params['p_swap'])
                self.assertEqual(agent.tau, params['tau'])
                self.assertEqual(agent.lr, params['lr'])
                self.assertEqual(agent.gamma, params['gamma'])
                self.assertIsInstance(agent.model, GNN)
    
    def test_get_valid_actions_different_sizes(self):
        """Test valid action identification for different network sizes"""

        for n in self.n_range:
            with self.subTest(n=n):
                agent = AgentGNN(n=n, p_entangle=1.0, p_swap=1.0)
                valid_actions = agent.get_valid_actions()
                
                self.assertIsInstance(valid_actions, list)
                self.assertTrue(len(valid_actions) > 0)
                
                # Check that all valid actions are within bounds
                max_possible_actions = len(agent.new_actions())
                for action in valid_actions:
                    self.assertGreaterEqual(action, 0)
                    self.assertLess(action, max_possible_actions)
    
    def test_reward_function_consistency(self):
        """Test reward function consistency across different states"""
        test_cases = []
        for _ in range(self.reward_consistency_checks):
            test_cases.append({
                'n': generateRandom_N(),
                'p_entangle': generateRandom_pe(), 
                'p_swap': generateRandom_ps()
            })
        
        for params in test_cases:
            with self.subTest(**params):
                agent = AgentGNN(**params)
                agent.reset()
                
                # Test reward for initial state (should be negative)
                initial_reward = agent.reward()
                self.assertLess(initial_reward, 0)
                
                # Test reward for end-to-end state (should be positive)
                agent.setLink(linkType=1, edge=(0, agent.n-1), newValue=1.0)
                agent.endToEndCheck()  # This sets global_state to True
                end_reward = agent.reward()
                self.assertEqual(end_reward, 1.0)
    
    def test_choose_action_probabilistic(self):
        """Test action selection with different epsilon values"""
        
        for epsilon in self.epsilon_values:
            with self.subTest(epsilon=epsilon):
                agent = AgentGNN(n=generateRandom_N(), 
                                 p_entangle=generateRandom_pe(), 
                                 p_swap=generateRandom_ps(), 
                                 epsilon=epsilon)
                
                # Test multiple action selections
                actions = []
                for _ in range(self.action_tests):
                    action = agent.choose_action(use_trained_model=False)
                    actions.append(action)
                    self.assertIn(action, range(len(agent.new_actions())))

def test_replay_buffer(self): #TODO: include tests
    ...
