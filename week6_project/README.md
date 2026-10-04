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
Writes performance plots and `comparison_table.csv` to `benchmarks/results/`. Takes
roughly 30-40 seconds -- Matrix Chain Multiplication's naive recursion, Floyd-Warshall
at larger graph sizes, and TSP's bitmask DP at higher city counts are each
deliberately pushed close to where they become slow, to make the comparison plots
show something real.

## Report
See `analysis/week6_report.md`.
