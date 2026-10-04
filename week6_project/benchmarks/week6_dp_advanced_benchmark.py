"""
week6_dp_advanced_benchmark.py
---------------------------------
Benchmarks and visualizes the four Week 6 algorithms:
    - Matrix Chain Multiplication: recursive vs. memo vs. tabulation
    - 0/1 Knapsack: full 2D table vs. space-optimized 1D array
    - Floyd-Warshall: single-algorithm time-vs-size scaling
    - Traveling Salesman: brute force vs. bitmask (Held-Karp) DP

What this script does, per algorithm:
    1. Runs each variant across a range of input sizes, measuring mean wall-clock TIME
       (time.perf_counter, averaged over a few repeats) and PEAK MEMORY (tracemalloc).
    2. Any exponential/factorial variant (MCM's naive recursion, TSP's brute force) is
       only run up to a small, safe cutoff size -- going further isn't practical, which
       is exactly the point each comparison plot is making.
    3. Exports every row to results/comparison_table.csv.
    4. Saves one PNG per algorithm via src/utils/visualization.py.

This file defines its own small time_execution/measure_peak_memory helpers rather than
importing them from a shared timer module -- unlike Week 5, this week's utils/ only
holds matrix_utils.py, bitmask_utils.py, and visualization.py.

Run with:
    python3 week6_dp_advanced_benchmark.py

(Needs matplotlib in addition to the stdlib.)
"""

import csv
import os
import random
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ^ This file lives in benchmarks/, one level below the project root where src/
#   actually is -- ".." goes up to that root so the imports below can resolve.

from src.dp_advanced.matrix_chain_multiplication import mcm_recursive, mcm_memo, mcm_tabulation
from src.dp_advanced.space_optimized_knapsack import knapsack_2d, knapsack_1d
from src.dp_advanced.floyd_warshall import floyd_warshall
from src.dp_advanced.bitmask_traveling_salesman import tsp_brute_force, tsp_bitmask_dp
from src.utils.matrix_utils import generate_chain_dimensions, generate_distance_matrix, generate_sparse_weighted_matrix
from src.utils.visualization import (
    save_mcm_figure,
    save_knapsack_space_figure,
    save_floyd_warshall_figure,
    save_tsp_figure,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

TIME_REPEATS = 3

# ---- Per-algorithm input-size configuration used by main() ----
# (run_all_benchmarks accepts overrides so tests can pass small/fast sizes instead.)
MCM_SIZES = [3, 5, 8, 10, 12, 20, 40, 80, 120, 200]
MCM_RECURSIVE_CUTOFF = 12

KNAPSACK_N_ITEMS = 30
KNAPSACK_CAPACITIES = [10, 50, 100, 500, 1000, 5000, 10000]

FLOYD_WARSHALL_SIZES = [5, 10, 20, 40, 80, 120, 160]

TSP_SIZES = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
TSP_BRUTE_FORCE_CUTOFF = 9


def time_execution(fn, repeats=TIME_REPEATS):
    """Mean wall-clock time (seconds) of `fn()` over `repeats` calls. `fn` takes no
    arguments -- callers wrap whatever they're timing in a small lambda/closure."""
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        total += time.perf_counter() - start
    return total / repeats


def measure_peak_memory(fn):
    """Peak traced memory (bytes) of ONE call to fn(), measured in its own tracemalloc
    session, kept separate from time_execution so tracemalloc's own bookkeeping doesn't
    inflate the timing numbers."""
    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def benchmark_variant(fn, algorithm, variant, input_size, rows, verbose=True):
    """Runs one (algorithm, variant, input_size) combination and appends its row to
    `rows`. `fn` is a zero-argument callable that runs the algorithm once."""
    mean_time = time_execution(fn)
    peak_mem = measure_peak_memory(fn)
    row = {
        "algorithm": algorithm,
        "variant": variant,
        "input_size": input_size,
        "time_us": round(mean_time * 1e6, 3),
        "peak_memory_kb": round(peak_mem / 1024, 3),
    }
    rows.append(row)
    if verbose:
        print(
            f"{algorithm:16s} {variant:12s} n={input_size:6d}  "
            f"time={row['time_us']:12.2f}us  mem={row['peak_memory_kb']:10.2f}KB"
        )


def run_mcm_benchmarks(rows, sizes=MCM_SIZES, recursive_cutoff=MCM_RECURSIVE_CUTOFF, seed=1, verbose=True):
    if verbose:
        print("\n### Matrix Chain Multiplication: recursive vs. memo vs. tabulation ###")
    rng = random.Random(seed)
    for n in sizes:
        dims = generate_chain_dimensions(n, rng)
        if n <= recursive_cutoff:
            benchmark_variant(lambda d=dims: mcm_recursive(d), "mcm", "recursive", n, rows, verbose)
        benchmark_variant(lambda d=dims: mcm_memo(d), "mcm", "memo", n, rows, verbose)
        benchmark_variant(lambda d=dims: mcm_tabulation(d), "mcm", "tabulation", n, rows, verbose)


def run_knapsack_space_benchmarks(rows, capacities=KNAPSACK_CAPACITIES, n_items=KNAPSACK_N_ITEMS, seed=7, verbose=True):
    if verbose:
        print("\n### 0/1 Knapsack: full 2D table vs. space-optimized 1D array ###")
    rng = random.Random(seed)
    weights = [rng.randint(1, 50) for _ in range(n_items)]
    values = [rng.randint(1, 100) for _ in range(n_items)]
    for capacity in capacities:
        benchmark_variant(lambda w=weights, v=values, c=capacity: knapsack_2d(w, v, c),
                           "knapsack_space", "2d", capacity, rows, verbose)
        benchmark_variant(lambda w=weights, v=values, c=capacity: knapsack_1d(w, v, c),
                           "knapsack_space", "1d", capacity, rows, verbose)


def run_floyd_warshall_benchmarks(rows, sizes=FLOYD_WARSHALL_SIZES, seed=13, verbose=True):
    if verbose:
        print("\n### Floyd-Warshall: time vs. graph size ###")
    rng = random.Random(seed)
    for n in sizes:
        matrix = generate_sparse_weighted_matrix(n, rng)
        benchmark_variant(lambda m=matrix: floyd_warshall(m), "floyd_warshall", "tabulation", n, rows, verbose)


def run_tsp_benchmarks(rows, sizes=TSP_SIZES, brute_force_cutoff=TSP_BRUTE_FORCE_CUTOFF, seed=21, verbose=True):
    if verbose:
        print("\n### TSP: brute force vs. bitmask DP (Held-Karp) ###")
    rng = random.Random(seed)
    for n in sizes:
        dist = generate_distance_matrix(n, rng)
        if n <= brute_force_cutoff:
            benchmark_variant(lambda d=dist: tsp_brute_force(d), "tsp", "brute_force", n, rows, verbose)
        benchmark_variant(lambda d=dist: tsp_bitmask_dp(d), "tsp", "bitmask_dp", n, rows, verbose)


def run_all_benchmarks(
    mcm_sizes=MCM_SIZES, mcm_cutoff=MCM_RECURSIVE_CUTOFF,
    knap_capacities=KNAPSACK_CAPACITIES, knap_n_items=KNAPSACK_N_ITEMS,
    fw_sizes=FLOYD_WARSHALL_SIZES,
    tsp_sizes=TSP_SIZES, tsp_cutoff=TSP_BRUTE_FORCE_CUTOFF,
    verbose=True,
):
    """Runs every algorithm's benchmark sweep and returns the combined rows list.
    Factored out from main() so tests (and anything else) can run a small, fast
    configuration without also writing CSV/PNG files to disk."""
    rows = []
    run_mcm_benchmarks(rows, mcm_sizes, mcm_cutoff, verbose=verbose)
    run_knapsack_space_benchmarks(rows, knap_capacities, knap_n_items, verbose=verbose)
    run_floyd_warshall_benchmarks(rows, fw_sizes, verbose=verbose)
    run_tsp_benchmarks(rows, tsp_sizes, tsp_cutoff, verbose=verbose)
    return rows


def export_csv(rows, path):
    fieldnames = ["algorithm", "variant", "input_size", "time_us", "peak_memory_kb"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = run_all_benchmarks()

    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    export_csv(rows, csv_path)
    print(f"\nWrote {csv_path}")

    save_mcm_figure(rows, os.path.join(RESULTS_DIR, "mcm_performance.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'mcm_performance.png')}")

    save_knapsack_space_figure(rows, os.path.join(RESULTS_DIR, "knapsack_space_comparison.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'knapsack_space_comparison.png')}")

    save_floyd_warshall_figure(rows, os.path.join(RESULTS_DIR, "floyd_warshall_scaling.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'floyd_warshall_scaling.png')}")

    save_tsp_figure(rows, os.path.join(RESULTS_DIR, "tsp_bitmask_runtime.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'tsp_bitmask_runtime.png')}")

    print(f"\nAll results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
