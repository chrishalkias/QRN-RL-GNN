import unittest
import numpy as np
import random
from base.repeaters import RepeaterNetwork
import random

def generateRandom_N():
    return random.randint(3,30)

def generateRandom_pe():
    return random.uniform(0.001, 0.99)

def generateRandom_ps():
    return random.uniform(0.1, 0.99)

def generateRandom_tau():
    return random.randint(100, 1000)

def generateRandom_cutoff():
    return random.randint(100, 1000)


def net_init(n=None, p_entangle=None, p_swap=None, tau=None, cutoff=None):
    """
    Initializes RepeaterNetwork. If an argument is None, generates a random value.
    Otherwise, uses the provided argument.
    """
    # Use the passed value if it exists, otherwise generate a random one
    n_val = n if n is not None else generateRandom_N()
    p_e_val = p_entangle if p_entangle is not None else generateRandom_pe()
    p_s_val = p_swap if p_swap is not None else generateRandom_ps()
    tau_val = tau if tau is not None else generateRandom_tau()
    cutoff_val = cutoff if cutoff is not None else generateRandom_cutoff()

    return RepeaterNetwork(
        n=n_val,
        p_entangle=p_e_val,
        p_swap=p_s_val,
        tau=tau_val,
        cutoff=cutoff_val
    )


class TestRepeaterNetwork_CoreTests(unittest.TestCase):

    def setUP(self):
        """
        Test the core functionality of the class

        Includes:
        """
        return
    
    def test_correct_Data_representations(self):

        NUMBER_OF_INITIALIZATIONS = 20
        TESTS_PER_INITIALIZATION = 100

        for _ in range(NUMBER_OF_INITIALIZATIONS):
            net = net_init()
            for _ in range(TESTS_PER_INITIALIZATION):
                n1= random.randint(0, net.n-2)
                net.entangle(edge=(n1, n1+1))
                state = net.tensorState()
                self.assertEqual(state.x[n1][1], state.x[n1+1][0])

    

class TestRepeaterNetwork_SanityChecks(unittest.TestCase):

    
    def setUp(self):
        """
        Sanity checks for the RepeaterNetwork class.
        Includes:
            test_initialization_parameters()
            test_connect_chain_different_sizes()
            test_entanglement_generation_probabilities()
            test_swap_operation_probabilities()
            test_link_decay_different_tau()
            test_end_to_end_different_sizes()
        """
        return
    
    def test_initialization_parameters(self):
            """Test network initialization with different parameters"""

            NUMBER_OF_RANDOM_INITIALIZATIONS = 10

            test_params = []
            for _ in range(NUMBER_OF_RANDOM_INITIALIZATIONS):
                test_params.append(
                    {'n': generateRandom_N(), 
                    'p_entangle': generateRandom_pe(), 
                    'p_swap': generateRandom_ps(), 
                    'tau': generateRandom_tau(),
                    'cutoff': bool(random.getrandbits(1)),
                    })
        
            for params in test_params:
                with self.subTest(**params):
                    net = RepeaterNetwork(**params)
                    self.assertEqual(net.n, params['n'])
                    self.assertEqual(net.p_entangle, params['p_entangle'])
                    self.assertEqual(net.p_swap, params['p_swap'])
                    self.assertEqual(net.tau, params['tau'])
    
    def test_connect_chain_different_sizes(self):
        """Test chain connectivity for different network sizes"""

        SIZE_RANGE = range(3, 30)

        for n in SIZE_RANGE:
            with self.subTest(n=n):
                net = net_init(n=n, p_entangle=1.0, p_swap=1.0)
                
                # Check that only neighboring nodes are connected
                for (i, j), (adj, ent) in net.matrix.items():
                    if abs(i - j) == 1:
                        self.assertEqual(adj, 1, f"{i,j} non-local at n={n}")
                    else: 
                        self.assertEqual(adj, 0, f"{i,j} local at n={n}")
    
    def test_entanglement_generation_probabilities(self):
        """Test entanglement generation with different probabilities"""

        NUMBER_OF_TESTS = 100
        PROBABILITY_RANGE = np.linspace(0.01, 1.0, 100)

        for p_entangle in PROBABILITY_RANGE:
            with self.subTest(p_entangle=p_entangle):
                success_count = 0
                net = net_init(p_swap=1)
                
                for _ in range(NUMBER_OF_TESTS):
                    net.reset()
                    net.entangle((0, 1))
                    if net.getLink((0, 1), 1) > 0:
                        success_count += 1
                
                # Check that success rate is approximately equal to probability
                # Allow some tolerance for randomness
                success_rate = success_count / NUMBER_OF_TESTS
                tolerance = 0.30  # 30% tolerance
                self.assertAlmostEqual(success_rate, 
                                    p_entangle, 
                                    delta=tolerance,
                                    msg=f"Tested success rate {success_rate} vs true ({p_entangle})")
    
    def test_swap_operation_probabilities(self):
        """Test swap operations with different probabilities"""

        NUMBER_OF_TESTS = 100
        PROBABILITY_RANGE = np.linspace(0.3, 1, 100)

        for p_swap in PROBABILITY_RANGE:
            with self.subTest(p_swap = p_swap):
                success_count = 0
                net = net_init(p_entangle=1, p_swap=p_swap)
                
                for _ in range(NUMBER_OF_TESTS):
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
        return
        #TODO: fix this

        TAU_RANGE = np.linspace(50, 5_000, 200)

        for tau in TAU_RANGE:
            with self.subTest(tau=tau):
                net = net_init(p_entangle=1.0, p_swap=1.0, tau=tau)
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
        """Tests whether links are discarded after cutoff time"""
        return
        #TODO: Fix this

        TAU_RANGE = np.linspace(50, 50_000, 200)

        for tau in TAU_RANGE:
            with self.subTest(tau=tau):
                cutoff = random.uniform(tau, 100 * tau)
                net = net_init(p_entangle=1.0, 
                                p_swap=1.0, 
                                cutoff=cutoff,
                                tau=tau)
                node = random.randint(1, net.n-1)
                net.entangle(edge=(node-1, node))
                net.tick(cutoff+1)
                # Check that link is discarded after the cutoff time
                self.assertTrue(net.getLink(edge=(node-1, node), linkType=1) == 0)
    
    def test_end_to_end_different_sizes(self):
        """Test end-to-end entanglement for different network sizes"""
        SIZES = range(4, 10)
        sizes = self.endtoend_sizes
        
        for n in sizes:
            with self.subTest(n=n):
                net = net_init(n=n, p_entangle=1.0, p_swap=1.0)
                
                # Initially should not be end-to-end entangled
                self.assertFalse(net.endToEndCheck())

                # Create end-to-end link directly
                net.setLink(linkType=1, edge=(0, n-1), newValue=1.0)
                
                # Should now detect end-to-end entanglement
                self.assertEqual(net.endToEndCheck(timeToWait=0), True)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork_CoreTests))
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork_SanityChecks))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)  
    success = result.wasSuccessful()
    exit(0 if success else 1)
