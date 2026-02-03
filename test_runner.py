#!/usr/bin/env python3
"""
Test runner for the agent memory system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytest


def run_tests():
    """Run all tests."""
    
    print("🧪 Running Tests for Agent Memory System")
    print("=" * 60)
    
    # Define test modules to run
    test_modules = [
        'tests/test_scoring.py',
        'tests/test_forgetting.py',
        'tests/test_memory_manager.py'
    ]
    
    # Run tests
    results = []
    for test_module in test_modules:
        print(f"\n📋 Running {test_module}")
        result = pytest.main([test_module, '-v'])
        results.append((test_module, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_module, result in results:
        status = "✅ PASS" if result == 0 else "❌ FAIL"
        print(f"{test_module}: {status}")
    
    # Check if all passed
    all_passed = all(result == 0 for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed.")
    
    return all_passed


def run_simple_tests():
    """Run simple tests without ChromaDB dependency."""
    
    print("🧪 Running Simple Tests (No ChromaDB)")
    print("=" * 60)
    
    # Import and run scoring tests
    from tests.test_scoring import TestMemoryScorer
    from tests.test_forgetting import TestForgettingMechanism
    
    import unittest
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryScorer)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestForgettingMechanism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 Simple Tests Result:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    print("Select test mode:")
    print("1. Run all tests (including ChromaDB)")
    print("2. Run simple tests (no ChromaDB)")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        success = run_tests()
        sys.exit(0 if success else 1)
    elif choice == '2':
        success = run_simple_tests()
        sys.exit(0 if success else 1)
    else:
        print("Exiting...")
        sys.exit(0)