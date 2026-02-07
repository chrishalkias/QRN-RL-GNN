import unittest
from base.repeaters import RepeaterNetwork
from base.agent import AgentGNN
from base.model import GNN
from torch_geometric.data import Data
from generators import *
from test_repeaters import TestRepeaterNetwork
from test_agent import TestAgentGNN
from test_integration import TestIntegration, TestPerformanceScaling


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
    print(f"    Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
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