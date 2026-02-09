import unittest
from generators import *
from test_repeaters import TestRepeaterNetwork_SanityChecks
from test_agent import TestAgentGNN
from test_integration import TestIntegration, TestPerformanceScaling


def run_all_parameterized_tests():
    """Run all parameterized tests and return the test summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork_SanityChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentGNN))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceScaling))
    
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':

    success = run_all_parameterized_tests()
    
    if success:
        print("\n🎉 All parameterized tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)