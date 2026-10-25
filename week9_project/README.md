# Week 9: Approximation Algorithms

## Description
This week covers polynomial-time approximation algorithms for problems that don't
have known exact polynomial solutions, greedy approximations with provable ratio
bounds, LP relaxation and rounding, a metric TSP approximation, and a randomized
approximation with an expected-value guarantee.

- **Vertex Cover (2-approximation)** -- the greedy edge-picking algorithm, cost
  provably at most 2x the true optimal.
- **Facility Location (greedy heuristic)** -- opens facilities greedily for the
  Uncapacitated Facility Location problem, benchmarked against small-instance optimal
  solutions.
- **Set Cover (LP relaxation and rounding)** -- models Set Cover as an integer
  program, solves the LP relaxation, then recovers an integer solution via both
  randomized and deterministic rounding, targeting an O(log n) approximation.
- **Metric TSP approximation** -- MST doubling (2-approximation), the simpler
  student-level alternative to Christofides' algorithm.
- **Max-Cut (randomized approximation)** -- each vertex picks a side at random, with
  expected cut value at least half of OPT, compared against a local search heuristic.

Every algorithm here ships with its own exact or brute-force solver, used purely to
measure the real approximation ratio on small instances rather than just citing the
proof.

## Structure
```
week9_project/
├── src/
│   ├── approximation/      # the five algorithms (vertex cover, facility location,
│   │                         set cover, tsp, max-cut)
│   └── utils/                # lp_solver.py, graph_utils.py, visualization.py
├── tests/                      # per-algorithm correctness tests + benchmark harness tests
├── benchmarks/                   # week9_approx_benchmark.py and its results/ (plots + CSV)
├── analysis/                       # week9_report.md
└── examples/                        # week9_demo.py
```

## Setup
```bash
pip install scipy matplotlib
```
scipy's `linprog` solves the Set Cover LP relaxation, matplotlib renders the
benchmark figures.

## Running Tests
From inside `week9_project/`:
```bash
pytest tests/
```
42 tests across 5 files, all passing: one file per algorithm (vertex cover shares
its file with facility location, since they're both Part 1), plus a benchmark
harness test file that checks the benchmark script's output shape rather than
re-testing the algorithms themselves.

## Running the Demo
```bash
python3 examples/week9_demo.py
```
Builds one small instance of each problem and prints the approximation next to the
exact answer, a quick way to see every algorithm working without reading the test
suite.

## Running the Benchmark
```bash
python3 benchmarks/week9_approx_benchmark.py
```
Writes `approx_ratios.png`, `cost_vs_optimal.png`, `randomized_maxcut_results.png`, and
`comparison_table.csv` to `benchmarks/results/`. Takes a few seconds, small-instance
ratio sweeps run each algorithm's exact solver alongside the approximation, larger
runtime sweeps skip the exact solver entirely and just time the approximation
algorithm as n grows.

## Report
See `analysis/week9_report.md`.
