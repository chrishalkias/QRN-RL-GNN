import unittest
import numpy as np
import torch
import random
from repeaters import RepeaterNetwork
from agent import AgentGNN
from model import GNN
from torch_geometric.data import Data

class TestRepeaterNetwork(unittest.TestCase):
    """
    Tests for the RepeaterNetwork class

    Includes:
        setUp()
        test_initialization_parameters()
        test_connect_chain_different_sizes()
        test_entanglement_generation_probabilities()
        test_swap_operation_probabilities()
        test_link_decay_different_tau()
        test_end_to_end_different_sizes()
    """
    
    def setUp(self):

        self.num_inits = 10
        self.connect_sizes = range(3, 30)

        self.EG_tests = 100
        self.EG_probs = np.linspace(0.01, 1.0, 100)

        self.S_tests = 100
        self.S_probs = np.linspace(0.3, 1, 100)

        self.decay_steps = 10
        self.tau_values = np.linspace(50, 50_000, 200)

        self.endtoend_sizes = range(3, 40)
    
    def test_initialization_parameters(self):
        """Test network initialization with different parameters"""
        test_params = []
        for _ in range(self.num_inits):
            test_params.append(
                {'n': random.randint(3,30), 
                 'p_entangle': random.uniform(0.001, 0.99), 
                 'p_swap': random.uniform(0.1, 0.99), 
                 'tau': random.randint(100, 1_000),
                 'cutoff': bool(random.getrandbits(1)),
                 })

        
        for params in test_params:
            with self.subTest(**params):
                net = RepeaterNetwork(**params)
                self.assertEqual(net.n, params['n'])
                self.assertEqual(net.p_entangle, params['p_entangle'])
                self.assertEqual(net.p_swap, params['p_swap'])
                self.assertEqual(net.tau, params['tau'])
                self.assertFalse(net.global_state)
    
    def test_connect_chain_different_sizes(self):
        """Test chain connectivity for different network sizes"""

        for n in self.connect_sizes:
            with self.subTest(n=n):
                net = RepeaterNetwork(n=n, p_entangle=1.0, p_swap=1.0)
                
                # Check that only neighboring nodes are connected
                for (i, j), (adj, ent) in net.matrix.items():
                    if abs(i - j) == 1:  # Neighbors
                        self.assertEqual(adj, 1, f"Neighbors ({i},{j}) should be connected in n={n}")
                    else:  # Non-neighbors
                        self.assertEqual(adj, 0, f"Non-neighbors ({i},{j}) should not be connected in n={n}")
    
    def test_entanglement_generation_probabilities(self):
        """Test entanglement generation with different probabilities"""
        probabilities = self.EG_probs
        n_tests = self.EG_tests  # Number of tests per probability
        
        for p_entangle in probabilities:
            with self.subTest(p_entangle=p_entangle):
                success_count = 0
                net = RepeaterNetwork(
                    n=random.randint(3, 20), 
                    p_entangle=p_entangle, 
                    p_swap=1.0, 
                    tau=random.randint(100, 1_000)
                    )
                
                for _ in range(n_tests):
                    net.reset()
                    net.entangle((0, 1))
                    if net.getLink((0, 1), 1) > 0:
                        success_count += 1
                
                # Check that success rate is approximately equal to probability
                # Allow some tolerance for randomness
                success_rate = success_count / n_tests
                tolerance = 0.30  # 30% tolerance
                self.assertAlmostEqual(success_rate, 
                                    p_entangle, 
                                    delta=tolerance,
                                    msg=f"Tested success rate {success_rate} vs true ({p_entangle})")
    
    def test_swap_operation_probabilities(self):
        """Test swap operations with different probabilities"""
        for p_swap in self.S_probs:
            with self.subTest(p_swap = p_swap):
                success_count = 0
                net = RepeaterNetwork(n=random.randint(4, 20), 
                                    p_entangle=1.0, 
                                    p_swap=p_swap, 
                                    tau=random.randint(100, 1_000))
                
                for _ in range(self.S_tests):
                    i= random.randint(1,net.n-2)
                    net.reset()
                    net.entangle(edge=(i-1, i))
                    net.entangle(edge=(i, i+1))
                    
                    net.swap((i-1, i), (i, i+1))
                    
                    # Check if swap was successful
                    if net.getLink(edge=(i-1, i+1), linkType=1) > 0:
                        success_count += 1
                
                success_rate = success_count / self.S_tests
                tolerance = 0.2  # 20% tolerance for small sample size
                self.assertAlmostEqual(success_rate, 
                                    p_swap, 
                                    delta=tolerance,
                                    msg=f"Tested success rate {success_rate} vs true ({p_swap})")
    
    def test_link_decay_different_tau(self):
        """Test link decay with different tau values"""
        for tau in self.tau_values:
            with self.subTest(tau=tau):
                net = RepeaterNetwork(n=random.randint(3, 20), 
                                      p_entangle=1.0, 
                                      p_swap=1.0, 
                                      tau=tau)
                timestep = random.randint(1, 5000)
                i = random.randint(1,net.n-1)
                net.entangle((i-1, i))
                initial_entanglement = net.getLink(edge=(i-1, i), linkType=1)
                
                # Advance time
                net.tick(timestep)
                
                decayed_entanglement = net.getLink(edge=(i-1, i), linkType=1)
                expected_decay = initial_entanglement * np.exp(-timestep / tau)
                
                self.assertAlmostEqual(decayed_entanglement, 
                                       expected_decay, 
                                       places=5,
                                       msg=f"Decay incorrect for tau={tau}")
                
    def test_link_discard_with_cutoff(self):
        for tau in self.tau_values:
            with self.subTest(tau=tau):
                cutoff = random.uniform(tau, 1000 * tau)
                net = RepeaterNetwork(n=random.randint(3, 20), 
                                      p_entangle=1.0, 
                                      p_swap=1.0, 
                                      cutoff=cutoff,
                                      tau=tau)
                node = random.randint(1, net.n-1)
                net.entangle(edge=(node-1, node))
                net.tick(cutoff)
                # Check that link is discarded after the cutoff time
                self.assertTrue(net.getLink(edge=(node-1, node), linkType=1) == 0)
    
    def test_end_to_end_different_sizes(self):
        """Test end-to-end entanglement for different network sizes"""
        sizes = self.endtoend_sizes
        
        for n in sizes:
            with self.subTest(n=n):
                net = RepeaterNetwork(n=n, p_entangle=1.0, p_swap=1.0)
                
                # Initially should not be end-to-end entangled
                self.assertFalse(net.endToEndCheck())

                # Create end-to-end link directly
                net.setLink(linkType=1, edge=(0, n-1), newValue=1.0)
                
                # Should now detect end-to-end entanglement
                self.assertTrue(net.endToEndCheck())
                self.assertTrue(net.global_state)


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
                {'n': random.randint(3,20), 
                 'p_entangle': random.uniform(0.001, 0.99), 
                 'p_swap': random.uniform(0.1, 0.99), 
                 'tau': random.randint(100, 1_000),
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
                'n': random.randint(3, 10),
                'p_entangle': random.uniform(0.01, 0.90), 
                'p_swap': random.uniform(0.1, 0.99)
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
                agent = AgentGNN(n=random.randint(3, 10), 
                                 p_entangle=random.uniform(0.1, 0.8), 
                                 p_swap=random.uniform(0.5, 0.99), 
                                 epsilon=epsilon)
                
                # Test multiple action selections
                actions = []
                for _ in range(self.action_tests):
                    action = agent.choose_action(use_trained_model=False)
                    actions.append(action)
                    self.assertIn(action, range(len(agent.new_actions())))


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
                'n': random.randint(3, 10),
                'p_entangle': random.uniform(0.01, 0.90), 
                'p_swap': random.uniform(0.1, 0.99)
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
                agent = AgentGNN(n=random.randint(3, 10), 
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
        sizes = range(3, 50)
        
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
                'n': random.randint(3, 10),
                'p_entangle': random.uniform(0.1, 0.9),
                'p_swap': random.uniform(0.4, 0.95),
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
                'n': random.randint(3, 10),
                'p_entangle': random.uniform(0.1, 0.9),
                'p_swap': random.uniform(0.4, 0.95),
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



def run_parameterized_tests():
    """Run all parameterized tests and return the test summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentGNN))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceScaling))
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"    Tests run: {result.testsRun}")
    print(f"    Failures: {len(result.failures)}")
    print(f"    Errors: {len(result.errors)}")
    print(f"    Success: {result.wasSuccessful()}")
    print(f"{'='*50}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Now running tests...")
    print("=" * 60)
    
    success = run_parameterized_tests()
    
    if success:
        print("\n🎉 All parameterized tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)