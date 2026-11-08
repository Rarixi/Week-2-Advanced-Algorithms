# StringSearchEngine

## Description
A complete string matching toolkit, four core algorithms, three real
applications built on top of them, visualization, a benchmark suite, and
a written report. Fully implemented, tested, and benchmarked.

- **Knuth-Morris-Pratt (KMP)** -- deterministic linear time pattern
  matching via the failure function (LPS table). O(n + m).
- **Rabin-Karp** -- randomized fingerprint based search using a rolling
  hash, with single and multi pattern (batch) modes, collision detection,
  and a demo of how modulus choice affects the collision rate.
- **Suffix Array** -- O(n log n) prefix doubling construction, Kasai's
  algorithm for the LCP array, O(m log n) binary search substring query.
- **Suffix Tree (compressed trie)** -- substring/prefix queries, longest
  repeated substring, and distinct substring count, all O(n) once built.
  Edges store index pairs into the text, not copied substrings, to keep
  memory at O(n) even on texts with little internal repetition.
- **Unified Search Engine** -- `StringSearchEngine(text).find(pattern,
  method="kmp"|"rk"|"sa"|"tree")`, tracking preprocessing time, search
  time, and match counts per method, plus a `recommend()` built from its
  own recorded call history.
- **Plagiarism Detection** -- Rabin-Karp batch matching for overlapping
  phrases, a generalized suffix array for the longest common substring,
  a similarity score, and highlighted matched regions.
- **DNA Gene Search** -- FASTA loading, exact gene search via the suffix
  array or suffix tree, a gene panel scanner that reuses one index for
  many genes, and a brute force Hamming distance search for point
  mutations.
- **Autocomplete** -- a frequency weighted prefix trie over an
  independent vocabulary (not the suffix tree, see the module docstring
  for why), supporting prefix queries and top-k completions.

## Structure
```
StringSearchEngine/
├── core/              # kmp.py, rabin_karp.py, suffix_array.py, suffix_tree.py, search_engine.py
├── applications/      # plagiarism.py, dna_search.py, autocomplete.py
├── visualization/     # plot_suffix_array.py, draw_suffix_tree.py
├── utils/             # rolling_hash.py, text_loader.py, generators.py
├── tests/             # 112 tests, cross-validated against brute force references
├── benchmarks/        # benchmark_search.py, results/ (figures, CSVs, demo plot)
├── analysis/          # chapter10_report.md, the written technical report
└── README.md
```

## Setup
```bash
pip install matplotlib numpy
```

## Running Tests
From inside `StringSearchEngine/`:
```bash
pytest tests/
```
All 112 tests pass, covering every core algorithm (cross-validated
against brute force references and against each other, e.g. the suffix
array's and suffix tree's longest repeated substring answers always
agree), all three applications, the unified search engine, and the
rolling hash / FASTA / generator utilities.

## Running the Benchmark
```bash
python3 benchmarks/benchmark_search.py
```
Sweeps KMP, Rabin-Karp, and the suffix array from 10,000 to 1,000,000
characters, the suffix tree over its own smaller sweep (10,000 to
100,000, see the module docstring for why), and a Rabin-Karp collision
analysis across six different moduli. Writes `runtime_comparison.png`,
`collision_analysis.png`, `sa_lcp_stats.png`, `heatmap_dataset_sizes.png`,
a small suffix tree demo plot, and `comparison_table.csv` /
`collision_table.csv` to `benchmarks/results/`.

At 1,000,000 characters: KMP ran in about 0.08s, Rabin-Karp in about
0.57s, and the suffix array took 8.27s to build (search afterward took
about 0.00005s). The suffix tree at its largest tested size, 100,000
characters, took 1.67s to build. The collision analysis showed a 95%
collision rate at a modulus of 101, dropping to exactly 0% by a modulus
of 10,007 and staying there at the project's real default modulus.

## A note on scope
Two scope decisions were made deliberately and are explained in more
detail in the suffix tree's module docstring and in the report:
- The suffix tree is built as a compressed trie (the assignment's
  "easier" option), not Ukkonen's online algorithm. Its construction is
  O(n^2) in the worst case rather than Ukkonen's O(n), which is why it
  gets a smaller benchmark sweep than the other three algorithms.
- The benchmark skips the 100MB "large" dataset tier the assignment
  lists as optional, since the suffix array and suffix tree's
  construction costs would turn a short benchmark into a very long one
  without showing anything the smaller sizes don't already demonstrate.

## Report
`analysis/chapter10_report.md` covers string matching complexity, the
implementation (including a real bug the suffix array / suffix tree
cross-check caught), the benchmark results, the preprocessing-vs-runtime,
deterministic-vs-randomized, and memory-vs-speed trade-offs, how each
application maps to a real use case, and the engineering lessons from
the suffix tree's construction cost.
