# AdvancedDataStructures

## Description
`advds`: a library of advanced data structures written from scratch in
Python, with three applications, a benchmark suite and a written report
(Chapter 12). numpy is used for array storage and sorting; every data
structure and algorithm is implemented here.

- **Range-query trees** (`advds/trees/`)
  - `segment_tree.py`: O(n) build, range sum / min / max, **lazy
    propagation** for range add and range assign, ASCII and matplotlib
    visualization; plus a bottom-up iterative segment tree.
  - `fenwick_tree.py`: point update + prefix sums in n + 1 words, range
    sums by prefix differences, O(log n) k-th element search; plus a
    two-tree variant for range add + range sum.
  - `persistent_tree.py`: persistent segment tree (every update is a new
    version, old versions stay queryable, versions share nodes in a DAG,
    O(log n) new nodes per update, fast diffs between versions) and a
    persistent Merkle hash trie (`PersistentMap`).
- **Succinct structures** (`advds/succinct/`)
  - `bit_vector.py`: fixed-length packed bit vector.
  - `rank_select.py`: rank1/rank0 in O(1) with superblocks, blocks and a
    popcount lookup table; select1/select0 with sampled superblocks and a
    select-in-byte table.
  - `wavelet_tree.py`: access, rank, select, symbol frequency, range
    counting, range quantile, value-range counting, compressed text
    representation (balanced or Huffman-shaped), built on rank/select.
- **Cache-oblivious algorithms** (`advds/cache_oblivious/`)
  - `matrix_ops.py`: recursive traversal, divide-and-conquer
    multiplication, cache-oblivious transpose, and the naive versions to
    compare against (pure Python, numpy, and memory-address traces).
  - `layouts.py`: van Emde Boas layout for binary search trees vs BFS
    (array-based heap) and sorted-array layouts.
  - `cache_sim.py`: ideal-cache simulator (LRU, configurable line and
    cache size, multi-level) that measures misses.
- **Applications** (`advds/applications/`)
  - `range_queries.py`: Fenwick + segment tree + persistent tree:
    fast updates, sum/min/max queries, time-travel queries, diff, undo.
  - `version_control.py`: tiny Git: content-addressed blobs, commits as
    versions of a persistent tree, branches, checkout, Merkle-tree
    diffs, three-way merge with conflict detection, history graph.
  - `dna_index.py`: wavelet tree over ACGT (count of a base in [l, r],
    k-th smallest base in a window, GC content) and an FM-index (wavelet
    tree over the BWT + rank/select-sampled suffix array) for pattern
    frequency and positions; FASTA reader.
- **Benchmarks** (`advds/benchmarks/performance.py`): segment vs Fenwick,
  persistent overhead, rank/select speed, wavelet compression ratios,
  cache-oblivious vs naive (simulated misses and real timings); time
  plots, memory footprint, comparison tables.

## Structure
```
AdvancedDataStructures/
├── advds/
│   ├── trees/            segment_tree.py, fenwick_tree.py, persistent_tree.py
│   ├── succinct/         bit_vector.py, rank_select.py, wavelet_tree.py
│   ├── cache_oblivious/  matrix_ops.py, layouts.py, cache_sim.py
│   ├── applications/     range_queries.py, version_control.py, dna_index.py
│   └── benchmarks/       performance.py
├── tests/                205 tests
├── examples/             range_query_demo.py, vcs_demo.py, dna_demo.py, cache_demo.py
├── data/                 sample_fragment.fasta (synthetic, 50 kb)
├── benchmark_results/    plots, CSVs, tables.md, summary.txt from the full benchmark run
├── docs/                 chapter12_report.md
├── setup.py
└── README.md
```

## Setup
Needs numpy, matplotlib and pytest, which the course virtualenv already has:
```bash
source ~/algorithms_course/bin/activate
cd ~/algorithms_course/week13_project/AdvancedDataStructures
```

## Running Tests
```bash
pytest                       # all 205 tests, ~15 s (includes a quick benchmark run)
pytest -m "not slow"         # 204 tests, ~4 s
```
Every structure is checked against brute force on random inputs
(rank/select exhaustively at every position), plus the expected
complexity (nodes visited, nodes created per persistent update) and the
expected cache behaviour (recursive transpose hits the compulsory miss
count; vEB beats BFS and sorted layouts at every block size).

## Running the Demos
```bash
python3 examples/range_query_demo.py
python3 examples/vcs_demo.py
python3 examples/dna_demo.py                 # or: python3 examples/dna_demo.py genome.fasta
python3 examples/cache_demo.py
```
Figures go to `examples/output/`.

## Running the Benchmark
```bash
python3 -m advds.benchmarks.performance            # ~30 s, writes benchmark_results/
python3 -m advds.benchmarks.performance --quick    # ~15 s
```
`benchmark_results/summary.txt` has the headline numbers and
`benchmark_results/tables.md` the comparison tables. Timings depend on
the machine; miss counts and space figures don't.

## Notes on scope
- `rank1(i)` counts ones in positions [0, i) (the standard convention);
  ranges elsewhere in the library are inclusive [l, r].
- Rank is O(1). Select is O(1) when ones are spread out and O(log n) in
  the worst case (no extra directories for sparse regions).
- Python can't observe hardware cache misses, so cache behaviour is
  measured with the simulator; numpy-leaf timings show the real effect
  for transpose.
- Memory is reported both as compact-layout words/bits (what the theory
  counts) and as the Python heap actually used.
- The sample FASTA is synthetic (seeded, labelled in its header), not real
  genome data; any FASTA file can be passed to the DNA demo.
