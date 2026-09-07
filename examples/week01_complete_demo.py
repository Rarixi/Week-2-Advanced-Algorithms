"""
Complete Week 1 demonstration: From theory to practice.

This script demonstrates:
1. Algorithm implementation with proper documentation
2. Comprehensive testing
3. Performance benchmarking
4. Complexity analysis
5. Professional visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sorting.basic_sorts import bubble_sort, selection_sort, insertion_sort
from src.utils.benchmark import AlgorithmBenchmark
import matplotlib.pyplot as plt
import time

def demonstrate_correctness():
    """Demonstrate that our algorithms work correctly."""
    print("🔍 CORRECTNESS DEMONSTRATION")
    print("=" * 50)
    
    # Test cases that cover edge cases and typical scenarios
    test_cases = {
        "Empty array": [],
        "Single element": [42],
        "Already sorted": [1, 2, 3, 4, 5],
        "Reverse sorted": [5, 4, 3, 2, 1],
        "Random order": [3, 1, 4, 1, 5, 9, 2, 6],
        "All same": [7, 7, 7, 7],
        "Negative numbers": [-3, -1, -4, -1, -5],
        "Mixed positive/negative": [3, -1, 4, 0, -2]
    }
    
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort
    }
    
    all_passed = True
    
    for test_name, test_array in test_cases.items():
        print(f"\n📝 Test case: {test_name}")
        print(f"   Input: {test_array}")
        
        expected = sorted(test_array)
        print(f"   Expected: {expected}")
        
        for algo_name, algorithm in algorithms.items():
            try:
                result = algorithm(test_array.copy())
                
                # Verify correctness
                if result == expected:
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                    all_passed = False
                
                print(f"   {algo_name:15}: {result} {status}")
                
            except Exception as e:
                print(f"   {algo_name:15}: ❌ ERROR - {e}")
                all_passed = False
    
    print(f"\n🎯 Overall result: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed

def demonstrate_efficiency():
    """Demonstrate efficiency analysis and comparison."""
    print("\n\n⚡ EFFICIENCY DEMONSTRATION")
    print("=" * 50)
    
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort
    }
    
    # Test on different input sizes
    sizes = [50, 100, 200, 500]
    
    benchmark = AlgorithmBenchmark()
    
    print("🔬 Running performance benchmarks...")
    print("This may take a moment...\n")
    
    # Test on different data types
    data_types = ["random", "sorted", "reverse"]
    
    for data_type in data_types:
        print(f"📊 Testing on {data_type.upper()} data:")
        results = benchmark.benchmark_suite(
            algorithms=algorithms,
            sizes=sizes,
            data_types=[data_type],
            runs=3
        )
        
        # Show complexity analysis
        print(f"\n🧮 Complexity Analysis for {data_type} data:")
        for algo_name, result_list in results.items():
            if result_list:
                analysis = benchmark.analyze_complexity(result_list, algo_name)
                print(f"  {algo_name}: {analysis['best_fit_complexity']} "
                      f"(R² = {analysis['best_fit_r_squared']:.3f})")
        
        # Create visualization
        benchmark.plot_comparison(
            results, 
            f"Performance on {data_type.title()} Data"
        )
        print()

def demonstrate_best_vs_worst_case():
    """Demonstrate best vs worst case performance."""
    print("📈 BEST VS WORST CASE ANALYSIS")
    print("=" * 40)
    
    size = 500
    print(f"Testing with {size} elements:\n")
    
    # Test insertion sort on different data types (most sensitive to input order)
    test_cases = {
        "Best case (sorted)": list(range(size)),
        "Average case (random)": AlgorithmBenchmark().generate_test_data(size, "random"),
        "Worst case (reverse)": list(range(size, 0, -1))
    }
    
    print("🔄 Insertion Sort Performance:")
    times = {}
    
    for case_name, test_data in test_cases.items():
        # Time the algorithm
        start_time = time.perf_counter()
        result = insertion_sort(test_data.copy())
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times[case_name] = elapsed
        
        print(f"  {case_name:20}: {elapsed:.6f} seconds")
    
    # Calculate ratios
    best_time = times["Best case (sorted)"]
    worst_time = times["Worst case (reverse)"]
    avg_time = times["Average case (random)"]
    
    print(f"\n📊 Performance Ratios:")
    print(f"  Worst/Best ratio:    {worst_time/best_time:.1f}x")
    print(f"  Average/Best ratio:  {avg_time/best_time:.1f}x")
    print(f"  Worst/Average ratio: {worst_time/avg_time:.1f}x")
    
    print(
        f"\nInsight: Insertion sort is {worst_time/best_time:.0f}x slower "
        "on reverse-sorted data!"
    )

def main():
    """Run the complete Week 1 demonstration."""
    print("🚀 ADVANCED ALGORITHMS - WEEK 1 COMPLETE DEMONSTRATION")
    print("=" * 60)
    print("This demo covers:")
    print("• Algorithm correctness verification")
    print("• Performance benchmarking and analysis") 
    print("• Best/worst case behavior")
    print("•
