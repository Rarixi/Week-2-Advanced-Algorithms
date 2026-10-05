# Week 6: Advanced Dynamic Programming

## Description
Four classic "advanced DP" problems, each illustrating a different idea beyond basic
memoization/tabulation:

- **Space-optimized 0/1 Knapsack** -- collapsing a full O(n * capacity) 2D table down
  to a single O(capacity) 1D array.
- **Matrix Chain Multiplication (MCM)** -- recursive, memoized, and tabulated versions,
  plus reconstructing the actual optimal parenthesization.
- **Floyd-Warshall** -- all-pairs shortest paths via a DP that sweeps over every
  possible intermediate node.
- **Traveling Salesman (TSP)** -- brute force permutation search vs. bitmask
  (Held-Karp) dynamic programming.

## Structure
```
week6_project/
├── src/
│   ├── dp_advanced/            # the four algorithms
│   └── utils/                   # matrix_utils.py, bitmask_utils.py, visualization.py
├── tests/                        # per-algorithm correctness tests + benchmark harness tests
├── benchmarks/                    # week6_dp_advanced_benchmark.py and its results/ (plots + CSV)
├── analysis/                      # week6_report.md
└── examples/                      # week6_demo.py
```

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib
```

## Running Tests
From inside `week6_project/`:
```bash
pytest tests/
```

## Running the Benchmark
```bash
python3 benchmarks/week6_dp_advanced_benchmark.py
```
Writes performance plots and `comparison_table.csv` to `benchmarks/results/`. Takes a
couple of minutes -- Matrix Chain Multiplication's naive recursion, Floyd-Warshall at
graph sizes up to 500 nodes, and TSP's brute force search out to 12 cities are each
deliberately pushed close to where they become slow, to make the comparison plots show
something real. TSP's brute force at 12 cities is the single slowest part of the whole
run.

## Running a Single File Directly
`space_optimized_knapsack.py` can also be run on its own for a quick comparison,
without the full benchmark sweep:
```bash
python3 src/dp_advanced/space_optimized_knapsack.py
```
Prints a small comparison table (value, time, and peak memory for the 2D vs. 1D
versions at a few capacities) plus a one-line summary.

## Floyd-Warshall vs. Dijkstra
`floyd_warshall.py` includes `compare_with_dijkstra(matrix)`, which runs
Floyd-Warshall once and Dijkstra from every node on the same small graph, checks that
both agree on every pairwise distance, and times both approaches. This is meant for
small graphs (not the large benchmark sizes above), since repeating Dijkstra once per
node is its own extra multiplier on top of each run's cost. See
`tests/test_floyd_warshall.py`'s `TestDijkstraComparison` for example usage.

## Report
See `analysis/week6_report.md`.
