import unittest
import numpy as np
import torch
from repeaters import RepeaterNetwork
from agent import AgentGNN
from model import GNN
from torch_geometric.data import Data

class TestRepeaterNetwork(unittest.TestCase):
    
    def setUp(self):
        """Set up a test network before each test"""
        self.n = 4
        self.net = RepeaterNetwork(n=self.n, p_entangle=1.0, p_swap=1.0, tau=1000)
    
    def test_initialization(self):
        """Test that the network initializes correctly"""
        self.assertEqual(self.net.n, self.n)
        self.assertFalse(self.net.global_state)
        self.assertEqual(self.net.time, 0)
        self.assertEqual(len(self.net.matrix), 6)  # 4 nodes -> 6 undirected edges
    
    def test_connect_chain(self):
        """Test chain connectivity"""
        # Check that only neighboring nodes are connected
        for (i, j), (adj, ent) in self.net.matrix.items():
            if abs(i - j) == 1:  # Neighbors
                self.assertEqual(adj, 1, f"Neighbors ({i},{j}) should be connected")
            else:  # Non-neighbors
                self.assertEqual(adj, 0, f"Non-neighbors ({i},{j}) should not be connected")
    
    def test_entanglement_generation(self):
        """Test entanglement generation between adjacent nodes"""
        initial_time = self.net.time
        self.net.entangle((0, 1))
        
        # Check that time advanced
        self.assertEqual(self.net.time, initial_time + 1)
        
        # Check that entanglement was created
        entanglement = self.net.getLink((0, 1), linkType=1)
        self.assertEqual(entanglement, 1.0)
    
    def test_swap_operation(self):
        """Test entanglement swapping"""
        # Create two adjacent entangled links
        self.net.entangle((0, 1))
        self.net.entangle((1, 2))
        
        # Perform swap
        self.net.swap((0, 1), (1, 2))
        
        # Check that original links are destroyed and new link is created
        self.assertEqual(self.net.getLink((0, 1), 1), 0.0)
        self.assertEqual(self.net.getLink((1, 2), 1), 0.0)
        self.assertTrue(self.net.getLink(edge=(0, 2), linkType=1) > 0)
    
    def test_swap_at_node(self):
        """Test swapAT method"""
        # Create two adjacent entangled links
        self.net.entangle((0, 1))
        self.net.entangle((1, 2))
        
        # Perform swap at node 1
        self.net.swapAT(1)
        
        # Check results
        self.assertEqual(self.net.getLink((0, 1), 1), 0.0)
        self.assertEqual(self.net.getLink((1, 2), 1), 0.0)
        self.assertTrue(self.net.getLink(edge=(0, 2), linkType=1) > 0)
    
    def test_link_decay(self):
        """Test that links decay over time"""
        self.net.entangle((0, 1))
        initial_entanglement = self.net.getLink((0, 1), 1)
        
        # Advance time
        self.net.tick(10)
        
        # Check that entanglement decayed
        decayed_entanglement = self.net.getLink((0, 1), 1)
        self.assertLess(decayed_entanglement, initial_entanglement)
    
    def test_end_to_end_check(self):
        """Test end-to-end entanglement detection"""
        # Initially should not be end-to-end entangled
        self.assertFalse(self.net.endToEndCheck())
        
        # Create end-to-end link directly
        self.net.setLink(linkType=1, edge=(0, self.n-1), newValue=1.0)
        
        # Now should detect end-to-end entanglement
        self.assertTrue(self.net.endToEndCheck())
        self.assertTrue(self.net.global_state)
    
    def test_reset(self):
        """Test network reset functionality"""
        # Create some entanglements
        self.net.entangle((0, 1))
        self.net.entangle((1, 2))
        
        # Reset
        self.net.reset()
        
        # Check all entanglements are 0
        for (i, j), (adj, ent) in self.net.matrix.items():
            self.assertEqual(ent, 0.0, f"Edge ({i},{j}) should be reset to 0")
    
    def test_tensor_state(self):
        """Test conversion to PyG Data object"""
        data = self.net.tensorState()
        
        # Check data types and shapes
        self.assertIsInstance(data, Data)
        self.assertEqual(data.x.shape, (self.n, 2))  # Node features
        self.assertEqual(data.edge_index.shape, (2, self.n-1))  # Edges
        self.assertEqual(data.edge_attr.shape, (len(self.net.matrix),))  # Edge features

class TestGNNModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test GNN model"""
        self.model = GNN()
        self.n = 4
        self.test_net = RepeaterNetwork(n=self.n)
    
    def test_model_initialization(self):
        """Test that GNN initializes correctly"""
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.decoder)
    
    def test_forward_pass(self):
        """Test GNN forward pass"""
        data = self.test_net.tensorState()
        
        with torch.no_grad():
            output = self.model(data)
        
        # Check output shape: [num_nodes, 2] for (entangle, swap) q-values per node
        self.assertEqual(output.shape, (self.n, 2))
        
        # Check that output contains finite values
        self.assertTrue(torch.isfinite(output).all())

class TestAgentGNN(unittest.TestCase):
    
    def setUp(self):
        """Set up test agent"""
        self.agent = AgentGNN(n=4, p_entangle=1.0, p_swap=1.0, tau=1000, epsilon=0.0)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsInstance(self.agent.model, GNN)
        self.assertIsInstance(self.agent.target_model, GNN)
        self.assertIsNotNone(self.agent.optimizer)
    
    def test_get_state_vector(self):
        """Test state vector retrieval"""
        state = self.agent.get_state_vector()
        self.assertIsInstance(state, Data)
    
    def test_get_valid_actions(self):
        """Test valid action identification"""
        valid_actions = self.agent.get_valid_actions()
        self.assertIsInstance(valid_actions, list)
        
        # Initially, only entanglement actions should be valid
        for action in valid_actions:
            # Even indices are entanglement actions
            self.assertEqual(action % 2, 0)
    
    def test_choose_action(self):
        """Test action selection"""
        # Test random action (with epsilon=0 in setUp, but we'll test both)
        action = self.agent.choose_action(use_trained_model=False)
        self.assertIn(action, self.agent.get_valid_actions())
        
        # Test trained action
        action = self.agent.choose_action(use_trained_model=True)
        self.assertIn(action, range(len(self.agent.new_actions())))
    
    def test_reward_function(self):
        """Test reward calculation"""
        # Test negative reward for non-end state
        reward = self.agent.reward()
        self.assertLess(reward, 0)
        
        # Test positive reward for end state
        self.agent.setLink(linkType=1, edge=(0, self.agent.n-1), newValue=1.0)
        self.agent.endToEndCheck()  # This sets global_state to True
        reward = self.agent.reward()
        self.assertEqual(reward, 1.0)
    
    def test_swap_asap_policy(self):
        """Test swap-asap policy generation"""
        actions = self.agent.swap_asap()
        self.assertIsInstance(actions, list)
        
        # All actions should be valid Python commands
        for action in actions:
            self.assertTrue(action.startswith('self.entangle') or 
                          action.startswith('self.swapAT'))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_complete_episode(self):
        """Test a complete episode from start to end-to-end entanglement"""
        agent = AgentGNN(n=4, p_entangle=1.0, p_swap=1.0, tau=1000)
        
        # Manually perform actions to reach end-to-end
        agent.entangle((0, 1))
        agent.entangle((1, 2))
        agent.entangle((2, 3))
        agent.swapAT(1)  # Swap at node 1
        agent.swapAT(2)  # Swap at node 2
        
        # Should now be end-to-end entangled
        self.assertTrue(agent.endToEndCheck())
    
    # def test_action_execution(self):
    #     """Test that actions can be executed via exec()"""
    #     agent = AgentGNN(n=4, p_entangle=1.0, p_swap=1.0, tau=1000)
    #     actions = agent.new_actions()
        
    #     # Execute first action (entangle (0,1))
    #     exec(actions[0])
        
    #     # Check that action was executed
    #     self.assertEqual(agent.getLink((0, 1), 1), 1.0)

def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestGNNModel))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentGNN))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Quantum Repeater Network Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)