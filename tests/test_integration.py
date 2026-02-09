import unittest
import numpy as np
import torch
import random
from base.agent import AgentGNN
from torch_geometric.data import Data
from generators import *

class TestIntegration(unittest.TestCase):
    """
    Integration tests with different parameters

    Includes:
        setUp()
        test_complete_episode_different_parameters()
        test_action_execution_robustness()
        test_swap_asap_policy_different_sizes()
    """

    def setUp(self):
        self.action_probs = np.linspace(0.5, 1, 5)
        self.swapasap_sizes = range(3, 10)
        self.episode_completion_checks = 20
    
    def test_complete_episode_different_parameters(self):
        """Test complete episodes with different parameters"""
        test_cases = []
        for _ in range(self.episode_completion_checks):
            test_cases.append({
                'n': generateRandom_N(),
                'p_entangle': generateRandom_pe(), 
                'p_swap': generateRandom_ps()
            })
        
        for params in test_cases:
            with self.subTest(**params):
                agent = AgentGNN(**params)
                
                # Perform actions to try to reach end-to-end
                max_attempts = 50
                success = False
                
                for attempt in range(max_attempts):
                    # Try to entangle all adjacent pairs
                    for i in range(agent.n - 1):
                        agent.entangle((i, i+1))
                    
                    # Try to swap at intermediate nodes
                    for i in range(1, agent.n - 1):
                        agent.swapAT(i)
                    
                    if agent.endToEndCheck():
                        success = True
                        break
                    else:
                        agent.reset()
                
                # With perfect operations, should always succeed
                if params['p_entangle'] == 1.0 and params['p_swap'] == 1.0:
                    self.assertTrue(success, f"Should reach end-to-end with perfect operations for n={params['n']}")
    
    def test_action_execution_robustness(self):
        """Test action execution robustness with different probabilities"""
        probabilities = self.action_probs
        
        for p in probabilities:
            with self.subTest(p_entangle=p, p_swap=p):
                agent = AgentGNN(n=generateRandom_N(), 
                                 p_entangle=p, 
                                 p_swap=p)
                actions = agent.new_actions()
                
                # Execute first few actions using agent's method instead of exec()
                for i in range(min(3, len(actions))):
                    try:
                        # Use the agent's update_environment method to execute actions
                        action_index = i
                        reward = agent.update_environment(action_index)
                        
                        # Basic checks
                        self.assertIsInstance(reward, (int, float))
                        self.assertTrue(agent.time >= 0)
                        
                    except Exception as e:
                        self.fail(f"Action execution failed for action {i} ({actions[i]}) with error: {e}")
    
    def test_swap_asap_policy_different_sizes(self):
        """Test swap-asap policy generation for different network sizes"""
        sizes = self.swapasap_sizes
        
        for n in sizes:
            with self.subTest(n=n):
                agent = AgentGNN(n=n, p_entangle=0.8, p_swap=0.8)
                actions = agent.swap_asap()
                
                self.assertIsInstance(actions, list)
                
                # Check that all actions are valid strings
                for action in actions:
                    self.assertIsInstance(action, str)
                    self.assertTrue(action.startswith('self.entangle') or 
                                  action.startswith('self.swapAT'))



class TestPerformanceScaling(unittest.TestCase):
    """
    Performance and scaling tests
    
    Includes:
        test_memory_usage_different_sizes()
        test_training_stability()
        test_state_representation_consistency()
    """

    def setUp(self):
        self.stability_consistency_tests = 30
    
    def test_memory_usage_different_sizes(self):
        """Test that models can handle different network sizes"""
        sizes = range(10, 50)
        
        for n in sizes:
            with self.subTest(n=n):
                # This should not raise memory errors
                agent = AgentGNN(n=n, p_entangle=0.8, p_swap=0.8)
                state = agent.get_state_vector()
                
                # Model should be able to process the state
                with torch.no_grad():
                    output = agent.model(state)
                
                self.assertEqual(output.shape, (n, 2))
    
    def test_training_stability(self):
        """Test that training doesn't crash with different parameters"""
        test_params = []
        for _ in range(self.stability_consistency_tests):
            test_params.append({
                'n': generateRandom_N(),
                'p_entangle': generateRandom_pe(),
                'p_swap': generateRandom_ps(),
                'lr': random.uniform(0.0001, 0.05)
            })
        
        for params in test_params:
            with self.subTest(**params):
                agent = AgentGNN(**params)
                
                # Perform a few training steps without actual training loop
                # Just test that the components work together
                state = agent.get_state_vector()
                action = agent.choose_action()
                reward = agent.update_environment(action)
                next_state = agent.get_state_vector()
                
                # Basic sanity checks
                self.assertIsInstance(state, Data)
                self.assertIsInstance(action, int)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(next_state, Data)
    
    def test_state_representation_consistency(self):
        """Test that state representation is consistent across parameters"""

        test_cases = []
        for _ in range(self.stability_consistency_tests):
            test_cases.append({
                'n': generateRandom_N(),
                'p_entangle': generateRandom_pe(),
                'p_swap': generateRandom_ps(),
            })

        for params in test_cases:
            with self.subTest(**params):
                agent = AgentGNN(**params)
                state = agent.get_state_vector()
                
                # Check state properties
                self.assertIsInstance(state, Data)
                self.assertEqual(state.x.shape[0], params['n'])  # Number of nodes
                self.assertEqual(state.x.shape[1], 2)  # Node features
                self.assertEqual(state.edge_index.shape[0], 2)  # Edge indices
                self.assertEqual(state.edge_index.shape[1], params['n'] - 1)  # Number of edges


