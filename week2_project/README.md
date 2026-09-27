# Week 2: Sorting Algorithms

## Description
Five sorting algorithms -- Bubble Sort, Selection Sort, and Insertion Sort (O(n^2)),
plus Merge Sort and Quick Sort (O(n log n)) -- implemented, tested, and benchmarked
against each other across random, sorted, reverse-sorted, nearly sorted, and
duplicate-heavy input.

## Structure
```
week2_project/
├── src/sorting/              # bubble/selection/insertion/merge/quick sort implementations
├── tests/                     # per-algorithm correctness tests + cross-algorithm agreement tests
├── benchmarks/                # week2_performance.py and its results/ (plots + CSV)
├── analysis/                  # week2_report.md
└── examples/                  # week2_demo.py
```

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib, pandas
```

## Running Tests
From inside `week2_project/`:
```bash
pytest tests/
```

## Running the Benchmark
```bash
python3 benchmarks/week2_performance.py
```
Writes performance plots and `comparison_table.csv` to `benchmarks/results/`.

## Report
See `analysis/week2_report.md`.
