# Week 3: Data Structures

## Description
Three data structures -- a binary heap (MinHeap/MaxHeap/PriorityQueue), a
self-balancing AVL tree, and two hash tables (separate chaining and open
addressing/linear probing) -- implemented, tested, and benchmarked against each
other and against Python's built-in `dict`.

## Structure
```
week3_project/
├── src/data_structures/       # heap.py, avl_tree.py, hash_table.py
├── tests/                      # per-structure correctness tests
├── benchmarks/                 # week3_structures_benchmark.py and its results/ (plots + CSV)
├── analysis/                   # week3_report.md
└── examples/                   # week3_demo.py
```

Note: `heap.py` also appears in `week4_project/src/data_structures/`, duplicated
there because Week 4's Dijkstra implementation is built on this same
`PriorityQueue`. It's the same file in both places, kept identical so each week's
project folder stays independently runnable without cross-folder imports.

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib
```

## Running Tests
From inside `week3_project/`:
```bash
pytest tests/
```

## Running the Benchmark
```bash
python3 benchmarks/week3_structures_benchmark.py
```
Writes performance plots and `comparison_table.csv` to `benchmarks/results/`.

## Report
See `analysis/week3_report.md`.
