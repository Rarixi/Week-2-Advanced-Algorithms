"""
Professional benchmarking framework for algorithm analysis.
"""
import time
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    algorithm_name: str
    input_size: int
    average_time: float
    std_deviation: float
    min_time: float
    max_time: float
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = None

class AlgorithmBenchmark:
    """
    Professional algorithm benchmarking and analysis toolkit.
    
    Features:
    - Multiple run averaging with statistical analysis
    - Memory usage tracking
    - Complexity validation
    - Beautiful visualizations
    - Export capabilities
    """
    
    def __init__(self, warmup_runs: int = 2, precision: int = 6):
        self.warmup_runs = warmup_runs
        self.precision = precision
        self.results: List[BenchmarkResult] = []
        
    def generate_test_data(self, size: int, data_type: str = "random", 
                          seed: int = None) -> List[int]:
        """
        Generate various types of test data for algorithm testing.
        
        Args:
            size: Number of elements to generate
            data_type: Type of data to generate
            seed: Random seed for reproducibility
        
        Returns:
            List of test data
        """
        if seed is not None:
            random.seed(seed)
            
        generators = {
            "random": lambda: [random.randint(1, 1000) for _ in range(size)],
            "sorted": lambda: list(range(1, size + 1)),
            "reverse": lambda: list(range(size, 0, -1)),
            "nearly_sorted": self._generate_nearly_sorted,
            "duplicates": lambda: [random.randint(1, size // 10) for _ in range(size)],
            "single_value": lambda: [42] * size,
            "mountain": self._generate_mountain,
            "valley": self._generate_valley,
        }
        
        if data_type not in generators:
            raise ValueError(f"Unknown data type: {data_type}")
            
        if data_type in ["nearly_sorted", "mountain", "valley"]:
            return generators[data_type](size)
        else:
            return generators[data_type]()
    
    def _generate_nearly_sorted(self, size: int) -> List[int]:
        """Generate nearly sorted data with a few random swaps."""
        arr = list(range(1, size + 1))
        num_swaps = max(1, size // 20)  # 5% of elements
        for _ in range(num_swaps):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    def _generate_mountain(self, size: int) -> List[int]:
        """Generate mountain-shaped data (increases then decreases)."""
        mid = size // 2
        left = list(range(1, mid + 1))
        right = list(range(mid, 0, -1))
        return left + right
    
    def _generate_valley(self, size: int) -> List[int]:
        """Generate valley-shaped data (decreases then increases)."""
        mid = size // 2
        left = list(range(mid, 0, -1))
        right = list(range(1, size - mid + 1))
        return left + right
    
    def time_algorithm(self, algorithm: Callable, data: List[Any], 
                      runs: int = 5, verify_correctness: bool = True) -> BenchmarkResult:
        """
        Time an algorithm with multiple runs and statistical analysis.
        
        Args:
            algorithm: Function to benchmark
            data: Input data
            runs: Number of runs to average
            verify_correctness: Whether to verify output correctness
        
        Returns:
            BenchmarkResult with timing statistics
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            test_data = data.copy()
            algorithm(test_data)
        
        # Actual timing runs
        times = []
        for _ in range(runs):
            test_data = data.copy()
            
            start_time = time.perf_counter()
            result = algorithm(test_data)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Verify correctness on first run
            if verify_correctness and len(times) == 1:
                if not self._verify_sorting_correctness(data, result):
                    raise ValueError(f"Algorithm {algorithm.__name__} produced incorrect result")
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        return BenchmarkResult(
            algorithm_name=algorithm.__name__,
            input_size=len(data),
            average_time=round(avg_time, self.precision),
            std_deviation=round(std_time, self.precision),
            min_time=round(min_time, self.precision),
            max_time=round(max_time, self.precision)
        )
    
    def _verify_sorting_correctness(self, original: List, result: List) -> bool:
        """Verify that a sorting algorithm produced correct output."""
        if result is None:
            return False
        
        # Check if result is sorted
        if not all(result[i] <= result[i+1] for i in range(len(result)-1)):
            return False
        
        # Check if result contains same elements as original
        return sorted(original) == sorted(result)
    
    def benchmark_suite(self, algorithms: Dict[str, Callable], 
                       sizes: List[int], data_types: List[str] = None,
                       runs: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmarks across multiple algorithms and conditions.
        
        Args:
            algorithms: Dictionary of {name: function}
            sizes: List of input sizes to test
            data_types: List of data types to test
            runs: Number of runs per test
        
        Returns:
            Dictionary mapping algorithm names to their results
        """
        if data_types is None:
            data_types = ["random"]
        
        all_results = defaultdict(list)
        total_tests = len(algorithms) * len(sizes) * len(data_types)
        current_test = 0
        
        print(f"Running {total_tests} benchmark tests...")
        print("-" * 60)
        
        for data_type in data_types:
            print(f"\n📊 Testing on {data_type.upper()} data:")
            
            for size in sizes:
                print(f"\n  Input size: {size:,}")
                test_data = self.generate_test_data(size, data_type)
                
                for name, algorithm in algorithms.items():
                    current_test += 1
                    try:
                        result = self.time_algorithm(algorithm, test_data, runs)
                        all_results[name].append(result)
                        
                        # Progress indicator
                        progress = current_test / total_tests * 100
                        print(
                            f"    {name:20}: {result.average_time:8.6f}s "
                            f"± {result.std_deviation:.6f}s [{progress:5.1f}%]"
                        )
                        
                    except Exception as e:
                        print(f"    {name:20}: ERROR - {e}")
        
        self.results.extend([result for results in all_results.values() for result in results])
        return dict(all_results)
    
    def plot_comparison(self, results: Dict[str, List[BenchmarkResult]], 
                       title: str = "Algorithm Performance Comparison",
                       log_scale: bool = True, save_path: str = None):
        """
        Create professional visualization of benchmark results.
        
        Args:
            results: Results from benchmark_suite
            title: Plot title
            log_scale: Whether to use log scale for better visualization
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Color palette for algorithms
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (name, data), color in zip(results.items(), colors):
            if not data:  # Skip empty results
                continue
                
            sizes = [r.input_size for r in data]
            times = [r.average_time for r in data]
            stds = [r.std_deviation for r in data]
            
            # Plot line with error bars
            plt.plot(sizes, times, 'o-', label=name, color=color, 
                    linewidth=2, markersize=6)
            plt.errorbar(sizes, times, yerr=stds, color=color, 
                        alpha=0.3, capsize=3)
        
        plt.xlabel("Input Size (n)", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            
        # Add complexity reference lines
        if log_scale and len(results) > 0:
            sample_sizes = sorted({
                r.input_size
                for results_list in results.values()
                for r in results_list
            })
            if len(sample_sizes) >= 2:
                min_size, max_size = min(sample_sizes), max(sample_sizes)
                
                # Add O(n), O(n log n), O(n²) reference lines
                ref_sizes = np.logspace(np.log10(min_size), np.log10(max_size), 50)
                base_time = 1e-8  # Arbitrary base time for scaling
                
                plt.plot(ref_sizes, base_time * ref_sizes, '--', alpha=0.5, 
                        color='gray', label='O(n)')
                plt.plot(ref_sizes, base_time * ref_sizes * np.log2(ref_sizes), '--', 
                        alpha=0.5, color='orange', label='O(n log n)')
                plt.plot(ref_sizes, base_time * ref_sizes**2, '--', alpha=0.5, 
                        color='red', label='O(n²)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_complexity(self, results: List[BenchmarkResult], 
                          algorithm_name: str = None) -> Dict[str, Any]:
        """
        Analyze empirical complexity from benchmark results.
        
        Args:
            results: List of benchmark results for a single algorithm
            algorithm_name: Name of algorithm being analyzed
        
        Returns:
            Dictionary with complexity analysis
        """
        if len(results) < 3:
            return {"error": "Need at least 3 data points for complexity analysis"}
        
        # Sort results by input size
        sorted_results = sorted(results, key=lambda r: r.input_size)
        sizes = np.array([r.input_size for r in sorted_results])
        times = np.array([r.average_time for r in sorted_results])
        
        # Try to fit different complexity curves
        complexity_fits = {}
        
        # Linear: O(n)
        try:
            linear_fit = np.polyfit(sizes, times, 1)
            linear_pred = np.polyval(linear_fit, sizes)
            linear_r2 = 1 - np.sum((times - linear_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n)'] = {'r_squared': linear_r2, 'coefficients': linear_fit}
        except:
            pass
        
        # Quadratic: O(n²)
        try:
            quad_fit = np.polyfit(sizes, times, 2)
            quad_pred = np.polyval(quad_fit, sizes)
            quad_r2 = 1 - np.sum((times - quad_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n²)'] = {'r_squared': quad_r2, 'coefficients': quad_fit}
        except:
            pass
        
        # Linearithmic: O(n log n)
        try:
            log_sizes = sizes * np.log2(sizes)
            nlogn_fit = np.polyfit(log_sizes, times, 1)
            nlogn_pred = np.polyval(nlogn_fit, log_sizes)
            nlogn_r2 = 1 - np.sum((times - nlogn_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n log n)'] = {'r_squared': nlogn_r2, 'coefficients': nlogn_fit}
        except:
            pass
        
        # Find best fit
        best_fit = max(complexity_fits.items(), key=lambda x: x[1]['r_squared'])
        
        # Calculate doubling ratios for additional insight
        doubling_ratios = []
        for i in range(1, len(sorted_results)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            if size_ratio > 1:  # Only meaningful if size actually increased
                doubling_ratios.append(time_ratio / size_ratio)
        
        avg_ratio = np.mean(doubling_ratios) if doubling_ratios else 0
        
        return {
            'algorithm': algorithm_name or 'Unknown',
            'best_fit_complexity': best_fit[0],
            'best_fit_r_squared': best_fit[1]['r_squared'],
            'all_fits': complexity_fits,
            'average_doubling_ratio': avg_ratio,
            'interpretation': self._interpret_complexity(
                best_fit[0], best_fit[1]['r_squared'], avg_ratio
            )
        }
    
    def _interpret_complexity(
        self, complexity: str, r_squared: float, doubling_ratio: float
    ) -> str:
        """Provide human-readable interpretation of complexity analysis."""
        interpretation = f"Best fit: {complexity} (R² = {r_squared:.3f})\n"
        
        if r_squared > 0.95:
            interpretation += "Excellent fit - high confidence in complexity estimate."
        elif r_squared > 0.85:
            interpretation += "Good fit - reasonable confidence in complexity estimate."
        else:
            interpretation += "Poor fit - complexity may be more complex or need more data points."
        
        if complexity == 'O(n)' and 0.8 < doubling_ratio < 1.2:
            interpretation += "\nDoubling ratio confirms linear behavior."
        elif complexity == 'O(n²)' and 1.8 < doubling_ratio < 2.2:
            interpretation += "\nDoubling ratio confirms quadratic behavior."
        elif complexity == 'O(n log n)' and 1.0 < doubling_ratio < 1.5:
            interpretation += "\nDoubling ratio suggests linearithmic behavior."
        
        return interpretation
    
    def export_results(self, filename: str, format: str = 'csv'):
        """Export benchmark results to file."""
        if not self.results:
            print("No results to export")
            return
        
        if format == 'csv':
            import pandas as pd
            df = pd.DataFrame([
                {
                    'algorithm': r.algorithm_name,
                    'input_size': r.input_size,
                    'average_time': r.average_time,
                    'std_deviation': r.std_deviation,
                    'min_time': r.min_time,
                    'max_time': r.max_time
                }
                for r in self.results
            ])
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            raise ValueError(f"Unsupported format: {format}")
