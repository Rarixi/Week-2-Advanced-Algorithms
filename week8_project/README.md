# Week 8: Computational Complexity and Approximation

## Description
This week moves from algorithms that always give an exact answer to the class of
problems where an exact answer, in the worst case, costs exponential time to find at
all. Two tools for dealing with that: reductions, which let you carry hardness from
one problem to another, and approximation algorithms, which trade the exact answer
away on purpose for a provable, bounded amount of error in exchange for running in
polynomial time.

- **Reduction Demo** -- the classic 3-SAT to Vertex Cover construction, built step by
  step with every step logged, plus an independent correctness check (does "a vertex
  cover of the target size exists" actually agree with "the formula is satisfiable",
  verified via a real SAT solver and a real exact vertex cover solver, not assumed).
- **SAT Solver** -- plain DFS/backtracking over variable assignments for small CNF
  instances, with basic dead-clause pruning. Expected complexity O(2^n).
- **Subset Sum, Backtracking vs Heuristic** -- an exact include/exclude backtracking
  solver (O(2^n)) next to a greedy, sort-then-fill heuristic (O(n log n)) that trades
  the exact guarantee away for speed.
- **Vertex Cover Approximation** -- the standard greedy edge-picking 2-approximation
  (O(E)) next to an exact brute force solver used only on small instances, to measure
  the real approximation ratio instead of just citing the 2x bound.
- **TSP Approximation** -- a nearest neighbor heuristic (O(n^2)) next to an exact
  brute force solver used only on small instances, same idea, measuring the real
  approximation ratio on actual tours.

## Structure
```
week8_project/
├── src/
│   ├── complexity/        # the five algorithms (reduction, sat, subset sum, vertex cover, tsp)
│   └── utils/              # timer.py, graph_utils.py, visualization.py
├── tests/                   # per-algorithm correctness tests
├── benchmarks/               # week8_complexity_benchmark.py and its results/ (plots + CSV)
├── analysis/                  # week8_report.md
└── examples/                   # week8_demo.py
```

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib
```

## Running Tests
From inside `week8_project/`:
```bash
pytest tests/
```

## Running the Demo
```bash
python3 examples/week8_demo.py
```
A quick, readable walkthrough of all five parts on small examples, including the
reduction's full step by step log.

## Running the Benchmark
```bash
python3 benchmarks/week8_complexity_benchmark.py
```
Writes `runtime_growth.png`, `approximation_quality.png`, and `comparison_table.csv`
to `benchmarks/results/`. Takes well under a minute. The exponential algorithms
(subset sum backtracking and the SAT solver) are deliberately benchmarked on their
genuine worst case instances (a target just out of reach for subset sum, and random
3-CNF at the empirically hardest clause-to-variable ratio for SAT), since an easy
random instance can get lucky and finish fast regardless of size, which would hide
the exponential growth the benchmark is supposed to be showing. The approximation
ratio sweeps (vertex cover and TSP against their exact solvers) stay on small
instances only, since both exact solvers are themselves exponential, that's the whole
reason approximation algorithms exist in the first place.

Like week 7, this week's five algorithms don't share one row schema, so
`comparison_table.csv` uses one wide, mostly-sparse table, every row tagged with an
`algorithm` column, filling in only the fields that apply to it.

## Report
See `analysis/week8_report.md`.
