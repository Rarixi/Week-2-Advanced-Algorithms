# Week 4: Graphs -- Representation, Traversal, and Shortest Paths

## Description
A Graph supporting both adjacency-list and adjacency-matrix representations, BFS
and DFS traversal (iterative and recursive), and Dijkstra's shortest-path
algorithm (heap-based and a slower list-based baseline for comparison) --
implemented, tested, and benchmarked across sparse, dense, random, and weighted
graphs.

## Structure
```
week4_project/
├── src/
│   ├── graphs/                 # graph.py, bfs.py, dfs.py, dijkstra.py
│   ├── data_structures/        # heap.py (PriorityQueue -- see note below)
│   └── utils/                  # graph_generator.py
├── tests/                       # per-algorithm correctness tests
├── benchmarks/                   # week4_representation_benchmark.py, week4_graph_benchmark.py,
│                                 #   and their shared results/ (plots + CSVs)
├── analysis/                     # week4_report.md
└── examples/                     # week4_demo.py
```

Note: `src/data_structures/heap.py` is duplicated from `week3_project/` --
Dijkstra's heap-based implementation is built directly on Week 3's
`PriorityQueue`, so a copy of `heap.py` lives here too, kept identical to
Week 3's, so this folder stays independently runnable without importing across
week folders.

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib, networkx
```

## Running Tests
From inside `week4_project/`:
```bash
pytest tests/
```

## Running the Benchmarks
```bash
python3 benchmarks/week4_representation_benchmark.py   # Part 1: list vs. matrix
python3 benchmarks/week4_graph_benchmark.py             # Parts 2-4: BFS/DFS/Dijkstra
```
Writes performance plots and CSVs to `benchmarks/results/`.

## Report
See `analysis/week4_report.md`.
