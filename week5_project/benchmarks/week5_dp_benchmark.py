"""
week5_dp_benchmark.py
----------------------
Part 4 deliverable: benchmarks and visualizes the naive recursive vs. memoized vs.
tabulated implementations of Fibonacci (Part 1), 0/1 Knapsack (Part 2), and LCS (Part 3).

What this script does, per algorithm:
    1. Runs the naive recursive version and both DP versions (memo, tabulation) across a
       range of input sizes, measuring:
         - mean wall-clock TIME (utils.timer.time_execution)
         - PEAK MEMORY (utils.timer.measure_peak_memory)
         - CALL COUNT and max RECURSION DEPTH (utils.timer.count_calls_and_depth)
    2. The naive recursive version is exponential for all three problems, so it's only
       run up to a small, safe cutoff size; the DP versions are run across a much wider
       range to also demonstrate their own scalability once recursion is out of the
       picture (this mirrors how the Week 5 test suite only exercises *_recursive at
       small n, for the same reason).
    3. Exports every row to results/dp_vs_recursive_table.csv.
    4. Saves one three-panel comparison figure per algorithm -- time vs. size (log-log),
       calls vs. size (log-log), and recursive-vs-DP speedup factor -- via
       utils/visualization.py: fibonacci_comparison.png, knapsack_performance.png,
       lcs_performance.png.

Run with:
    python3 week5_dp_benchmark.py

(Needs matplotlib in addition to the stdlib.)
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ^ This file lives in benchmarks/, one level below the project root where src/ and
#   utils/ actually are -- ".." goes up to that root so the imports below can resolve.

sys.setrecursionlimit(6000)
# knapsack_memo/lcs_memo recurse roughly as deep as their input size (number of items,
# or combined string length); the largest sizes benchmarked below need headroom well
# beyond Python's default recursion limit of 1000.

from utils.timer import time_execution, measure_peak_memory, count_calls_and_depth
from utils.visualization import save_comparison_figure
from src.dp.fibonacci import fibonacci_recursive, fibonacci_memo, fibonacci_tabulation
from src.dp.knapsack import knapsack_recursive, knapsack_memo, knapsack_tabulation
from src.dp.lcs import lcs_recursive, lcs_memo, lcs_tabulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

TIME_REPEATS = 3

# ---- Per-algorithm input-size configuration used by main() ----
# (run_all_benchmarks accepts overrides so tests can pass small/fast sizes instead.)
FIBONACCI_SIZES = [5, 10, 15, 20, 24, 30, 50, 100, 200, 500, 1000]
FIBONACCI_RECURSIVE_CUTOFF = 24

KNAPSACK_SIZES = [5, 10, 15, 18, 25, 50, 100, 300, 500, 1000]
KNAPSACK_RECURSIVE_CUTOFF = 18
KNAPSACK_CAPACITY = 200

LCS_SIZES = [4, 6, 8, 10, 12, 20, 50, 100, 300, 500]
LCS_RECURSIVE_CUTOFF = 12


def make_knapsack_instance(n_items, rng):
    weights = [rng.randint(1, 50) for _ in range(n_items)]
    values = [rng.randint(1, 100) for _ in range(n_items)]
    return weights, values


def make_lcs_instance(length, rng, alphabet="ACGT"):
    x = "".join(rng.choice(alphabet) for _ in range(length))
    y = "".join(rng.choice(alphabet) for _ in range(length))
    return x, y


def benchmark_variant(fn, algorithm, variant, input_size, rows, verbose=True):
    """Runs one (algorithm, variant, input_size) combination and appends its row to
    `rows`. `fn` is a zero-argument callable that runs the algorithm once."""
    mean_time = time_execution(fn, repeats=TIME_REPEATS)
    peak_mem = measure_peak_memory(fn)
    _, calls, max_depth = count_calls_and_depth(fn)
    row = {
        "algorithm": algorithm,
        "variant": variant,
        "input_size": input_size,
        "time_us": round(mean_time * 1e6, 3),
        "peak_memory_kb": round(peak_mem / 1024, 3),
        "calls": calls,
        "max_depth": max_depth,
    }
    rows.append(row)
    if verbose:
        print(
            f"{algorithm:10s} {variant:12s} n={input_size:6d}  "
            f"time={row['time_us']:12.2f}us  mem={row['peak_memory_kb']:9.2f}KB  "
            f"calls={calls:9d}  depth={max_depth:5d}"
        )


def run_fibonacci_benchmarks(rows, sizes=FIBONACCI_SIZES, recursive_cutoff=FIBONACCI_RECURSIVE_CUTOFF, verbose=True):
    if verbose:
        print("\n### Fibonacci: recursive vs. memo vs. tabulation ###")
    for n in sizes:
        if n <= recursive_cutoff:
            benchmark_variant(lambda n=n: fibonacci_recursive(n), "fibonacci", "recursive", n, rows, verbose)
        benchmark_variant(lambda n=n: fibonacci_memo(n), "fibonacci", "memo", n, rows, verbose)
        benchmark_variant(lambda n=n: fibonacci_tabulation(n), "fibonacci", "tabulation", n, rows, verbose)


def run_knapsack_benchmarks(rows, sizes=KNAPSACK_SIZES, recursive_cutoff=KNAPSACK_RECURSIVE_CUTOFF,
                             capacity=KNAPSACK_CAPACITY, seed=7, verbose=True):
    if verbose:
        print("\n### 0/1 Knapsack: recursive vs. memo vs. tabulation ###")
    rng = random.Random(seed)
    for n in sizes:
        weights, values = make_knapsack_instance(n, rng)
        if n <= recursive_cutoff:
            benchmark_variant(lambda w=weights, v=values: knapsack_recursive(w, v, capacity),
                               "knapsack", "recursive", n, rows, verbose)
        benchmark_variant(lambda w=weights, v=values: knapsack_memo(w, v, capacity),
                           "knapsack", "memo", n, rows, verbose)
        benchmark_variant(lambda w=weights, v=values: knapsack_tabulation(w, v, capacity),
                           "knapsack", "tabulation", n, rows, verbose)


def run_lcs_benchmarks(rows, sizes=LCS_SIZES, recursive_cutoff=LCS_RECURSIVE_CUTOFF, seed=13, verbose=True):
    if verbose:
        print("\n### LCS: recursive vs. memo vs. tabulation ###")
    rng = random.Random(seed)
    for n in sizes:
        x, y = make_lcs_instance(n, rng)
        if n <= recursive_cutoff:
            benchmark_variant(lambda x=x, y=y: lcs_recursive(x, y), "lcs", "recursive", n, rows, verbose)
        benchmark_variant(lambda x=x, y=y: lcs_memo(x, y), "lcs", "memo", n, rows, verbose)
        benchmark_variant(lambda x=x, y=y: lcs_tabulation(x, y), "lcs", "tabulation", n, rows, verbose)


def run_all_benchmarks(
    fib_sizes=FIBONACCI_SIZES, fib_cutoff=FIBONACCI_RECURSIVE_CUTOFF,
    knap_sizes=KNAPSACK_SIZES, knap_cutoff=KNAPSACK_RECURSIVE_CUTOFF, knap_capacity=KNAPSACK_CAPACITY,
    lcs_sizes=LCS_SIZES, lcs_cutoff=LCS_RECURSIVE_CUTOFF,
    verbose=True,
):
    """Runs every algorithm's benchmark sweep and returns the combined rows list.
    Factored out from main() so tests (and anything else) can run a small, fast
    configuration without also writing CSV/PNG files to disk."""
    rows = []
    run_fibonacci_benchmarks(rows, fib_sizes, fib_cutoff, verbose=verbose)
    run_knapsack_benchmarks(rows, knap_sizes, knap_cutoff, knap_capacity, verbose=verbose)
    run_lcs_benchmarks(rows, lcs_sizes, lcs_cutoff, verbose=verbose)
    return rows


def export_csv(rows, path):
    fieldnames = ["algorithm", "variant", "input_size", "time_us", "peak_memory_kb", "calls", "max_depth"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = run_all_benchmarks()

    csv_path = os.path.join(RESULTS_DIR, "dp_vs_recursive_table.csv")
    export_csv(rows, csv_path)
    print(f"\nWrote {csv_path}")

    save_comparison_figure(
        rows, "fibonacci", os.path.join(RESULTS_DIR, "fibonacci_comparison.png"),
        "Fibonacci: recursive vs. memo vs. tabulation", "n",
    )
    print(f"Wrote {os.path.join(RESULTS_DIR, 'fibonacci_comparison.png')}")

    save_comparison_figure(
        rows, "knapsack", os.path.join(RESULTS_DIR, "knapsack_performance.png"),
        "0/1 Knapsack: recursive vs. memo vs. tabulation", "number of items",
    )
    print(f"Wrote {os.path.join(RESULTS_DIR, 'knapsack_performance.png')}")

    save_comparison_figure(
        rows, "lcs", os.path.join(RESULTS_DIR, "lcs_performance.png"),
        "LCS: recursive vs. memo vs. tabulation", "string length",
    )
    print(f"Wrote {os.path.join(RESULTS_DIR, 'lcs_performance.png')}")

    print(f"\nAll results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
