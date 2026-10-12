# Week 7: Randomized Algorithms

## Description
Five algorithms built around randomness, split between two uses of it: avoiding
worst-case input (randomized QuickSort and QuickSelect) and probabilistic
approximation (Monte Carlo primality testing, Bloom filters, and MinHash similarity
estimation).

- **Randomized QuickSort** -- random pivot selection vs. a fixed-pivot deterministic
  baseline, comparing runtime, comparison counts, and worst-case behavior.
- **Randomized Selection (QuickSelect)** -- expected O(n) kth-smallest-element
  selection vs. a sort-based deterministic baseline, tracking recursion depth and
  pivot "success" rate.
- **Monte Carlo Primality Testing** -- Fermat's and Miller-Rabin's probabilistic
  primality tests vs. deterministic trial division, at k = 5, 10, 20 rounds.
- **Bloom Filter** -- a probabilistic set membership structure (k hash functions over
  a bit array), measuring false-positive rate vs. load factor.
- **MinHash & Similarity Estimation** -- estimating Jaccard similarity between sets
  via compact signatures, at signature sizes k = 16 to 512.

A note on the two "vs. deterministic" comparisons: Week 2's actual `quick_sort.py`
already uses a random pivot (plus an insertion-sort cutoff), so it isn't a true
deterministic baseline, and there's no existing deterministic selection algorithm
anywhere in this repo either. `randomized_quicksort.py` and `randomized_selection.py`
each implement their own deterministic baseline (a last-element-pivot QuickSort, and a
sort-then-index Select) specifically so the randomized-vs-deterministic comparisons
this week is about are actually apples to apples. See each file's module docstring for
the full explanation.

## Structure
```
week7_project/
├── src/
│   ├── randomized/              # the five algorithms
│   └── utils/                    # random_utils.py, probability_analysis.py, visualization.py
├── tests/                         # per-algorithm correctness tests + benchmark harness tests
├── benchmarks/                     # week7_randomized_benchmark.py and its results/ (plots + CSV)
├── analysis/                       # week7_report.md
└── examples/                       # week7_demo.py
```

## Setup
```bash
pip install -r ../requirements.txt   # matplotlib
```

## Running Tests
From inside `week7_project/`:
```bash
pytest tests/
```

## Running the Benchmark
```bash
python3 benchmarks/week7_randomized_benchmark.py
```
Writes performance plots and `summary_table.csv` to `benchmarks/results/`. Takes
well under a minute -- the slowest single piece is deterministic QuickSort's
worst-case demonstration on already-sorted input up to n=4000, which is the whole
point of that part of the comparison (its O(n^2) behavior should actually be visible,
not just claimed).

Unlike earlier weeks, this week's five algorithms don't share one row schema (a
QuickSort row and a Bloom Filter row don't have much in common), so
`summary_table.csv` uses one wide, mostly-sparse table instead -- every row is tagged
with an `algorithm` column, and each row only fills in the columns that apply to it.

## Report
See `analysis/week7_report.md`.
