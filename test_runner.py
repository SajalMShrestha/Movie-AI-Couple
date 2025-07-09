#!/usr/bin/env python3
"""
Comprehensive test runner for the Movie Recommendation System.

This script runs all tests with proper reporting and coverage analysis.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_all_tests():
    """Run all tests and generate a comprehensive report."""
    
    print("🎬 Movie Recommendation System - Test Suite")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Load all test modules
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Custom test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream, 
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    print(f"📍 Running tests from: {start_dir}")
    print(f"🔍 Test pattern: test_*.py")
    print("-" * 60)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Get the output
    test_output = stream.getvalue()
    
    # Print results
    print("📊 TEST RESULTS")
    print("-" * 60)
    
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n🚨 FAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"❌ {test}")
            print(f"   {traceback.strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"💥 {test}")
            print(f"   {traceback.strip()}")
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! 🎉")
        success_rate = 100.0
    else:
        total_issues = len(result.failures) + len(result.errors)
        success_rate = ((result.testsRun - total_issues) / result.testsRun) * 100
        print(f"⚠️  Some tests failed. Success rate: {success_rate:.1f}%")
    
    # Module-by-module breakdown
    print("\n📋 TEST COVERAGE BY MODULE:")
    print("-" * 40)
    
    modules = [
        ("movie_search", "🔍 Movie Search & Fuzzy Matching"),
        ("narrative_analysis", "📖 Narrative Analysis & Text Processing"), 
        ("franchise_detection", "🎭 Franchise Detection & Limiting"),
        ("feedback_system", "📊 Feedback System & Google Sheets"),
        ("movie_scoring", "🎯 Movie Scoring & Recommendations"),
        ("utils", "🛠️  Utility Functions & Constants")
    ]
    
    for module, description in modules:
        test_file = f"test_{module}.py"
        test_path = os.path.join(start_dir, test_file)
        
        if os.path.exists(test_path):
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - MISSING TEST FILE")
    
    return result.wasSuccessful()

def run_specific_module(module_name):
    """Run tests for a specific module."""
    
    print(f"🎬 Running tests for module: {module_name}")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    test_file = f"tests/test_{module_name}.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        suite = loader.loadTestsFromName(f'test_{module_name}')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"❌ Error loading tests: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    
    print("🔧 Checking Dependencies")
    print("-" * 40)
    
    required_packages = [
        ('unittest', 'unittest'),
        ('unittest.mock', 'unittest.mock'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('requests', 'requests'),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT FOUND")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies satisfied!")
    return True

def main():
    """Main function to run tests based on command line arguments."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--deps":
            return check_dependencies()
        elif sys.argv[1] == "--help":
            print("🎬 Movie Recommendation System Test Runner")
            print("\nUsage:")
            print("  python test_runner.py                 - Run all tests")
            print("  python test_runner.py --deps          - Check dependencies")
            print("  python test_runner.py <module_name>   - Run specific module tests")
            print("\nAvailable modules:")
            print("  movie_search, narrative_analysis, franchise_detection")
            print("  feedback_system, movie_scoring, utils")
            return True
        else:
            # Run specific module
            module_name = sys.argv[1]
            return run_specific_module(module_name)
    else:
        # Check dependencies first
        if not check_dependencies():
            print("\n❌ Cannot run tests due to missing dependencies")
            return False
        
        print()  # Add spacing
        
        # Run all tests
        return run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)