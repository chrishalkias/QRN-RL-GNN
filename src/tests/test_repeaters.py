import unittest
import numpy as np
import random
from generators import *
from base.repeaters import RepeaterNetwork

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
                {'n': generateRandom_N(), 
                 'p_entangle': generateRandom_pe(), 
                 'p_swap': generateRandom_ps(), 
                 'tau': generateRandom_tau,
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
                    n=generateRandom_N(), 
                    p_entangle=p_entangle, 
                    p_swap=1.0, 
                    tau=generateRandom_tau()
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
                net = RepeaterNetwork(n=generateRandom_N(), 
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
                net = RepeaterNetwork(n=generateRandom_N(), 
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
                self.assertEqual(net.endToEndCheck(timeToWait=0), True)
                self.assertTrue(net.global_state)


def run_parameterized_tests():
    """Run all parameterized tests and return the test summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)  
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_parameterized_tests()
    
    if success:
        print("\n🎉 All parameterized tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)
