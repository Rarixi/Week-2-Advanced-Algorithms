# Week 5 Analysis Report: Dynamic Programming — Fibonacci, Knapsack, and LCS

## Executive Summary

Dynamic programming turns exponential recursion into polynomial computation whenever a problem has overlapping subproblems: the same smaller calculation gets asked for again and again along different paths through the recursion tree, and caching the answer the first time (memoization) or building it up in a fixed order ahead of time (tabulation) means it's only ever computed once. The benchmarks make that concrete. For Fibonacci, going from naive recursion to tabulation at n=24 cut the time from 5,110.35 μs to 1.26 μs. A 4,049x speedup, while the call count fell from 150,050 to 2. For 0/1 knapsack at 18 items, memoization made 3,468 calls versus recursion's 110,577 for the identical problem, meaning the average subproblem was recomputed roughly 32 times over before caching. For LCS at length 12, the overlap was far more dramatic, memoization needed 201 calls, whereas recursion needed 180,802, about 899 redundant recomputations per distinct subproblem. None of these problems changed, only the strategy for reusing work did.

## Methodology

### Hardware and Environment

Benchmarks ran directly on rari, the same machine used for the Week 3 and Week 4 assignments: a KVM virtual machine with 7 vCPUs (host CPU an 11th Gen Intel Core i7-11700F @ 2.50GHz), 44 GiB RAM, Ubuntu on a 7.0.0-31-generic kernel, and Python 3.14.4 inside the project's own virtual environment. Nothing else was running during the benchmark.

### Problem Parameters and Input Sizes

Naive recursion is exponential for all three problems, so it only ran up to a size that still finished in reasonable time, while memoization and tabulation were pushed much further to show their own scaling too. Fibonacci's recursive version was tested at n = 5, 10, 15, 20, 24 (n=24 already takes 150,050 calls, n=40 or 45 would take on the order of a billion and wasn't attempted), with memo/tabulation extended to n=1,000. Knapsack instances used a fixed capacity of 200, weights drawn uniformly from 1–50, values from 1–100 (fixed random seed), recursion was tested at 5, 10, 15, and 18 items, with memo/tabulation extended to 1,000. LCS instances used random strings over a 4-letter alphabet (ACGT, the real alphabet of DNA, relevant to the case study below), recursion was tested at lengths 4, 6, 8, 10, 12, with memo/tabulation extended to 500.

### Tools and Libraries

Every timing figure is the mean of 3 runs from `time.perf_counter()`. Peak memory comes from `tracemalloc`, measured in its own separate call so its bookkeeping doesn't inflate timing. Function call counts and maximum recursion depth come from a custom `sys.settrace`-based instrument (`utils/timer.py`) rather than a simple wrapper, because knapsack's and LCS's actual recursion happens inside a private nested closure, from outside, only one call to `knapsack_recursive` is ever visible, so a wrapper around the public function alone would always report "1 call" and miss the recursion entirely. `matplotlib` produced all performance plots.

## Results

### Time and Space Complexity

| Problem | Recursive | Memoized | Tabulated |
|---|---|---|---|
| Fibonacci | O(2^n) time, O(n) stack | O(n) time, O(n) cache + stack | O(n) time, O(1) space |
| 0/1 Knapsack | O(2^n) time, O(n) stack | O(n·capacity) time, O(n·capacity) cache + O(n) stack | O(n·capacity) time, O(n·capacity) table |
| LCS | O(2^(n+m)) time, O(n+m) stack | O(n·m) time, O(n·m) cache + O(n+m) stack | O(n·m) time, O(n·m) table |

### Fibonacci

| n | recursive time | memo time | tabulation time | recursive calls | memo calls |
|---|---|---|---|---|---|
| 15 | 64.97 μs | 3.16 μs | 0.93 μs | 1,974 | 30 |
| 20 | 737.39 μs | 3.72 μs | 0.96 μs | 21,892 | 40 |
| 24 | 5,110.35 μs | 5.44 μs | 1.26 μs | 150,050 | 48 |

### 0/1 Knapsack (capacity=200)

| items | recursive time | memo time | tabulation time | memo memory | tabulation memory |
|---|---|---|---|---|---|
| 5 | 5.73 μs | 8.25 μs | 89.48 μs | 3.59 KB | 9.49 KB |
| 18 | 6,029.21 μs | 492.03 μs | 326.82 μs | 197.47 KB | 47.31 KB |
| 1,000 | — | 186,680.93 μs | 26,738.82 μs | 31,453.03 KB | 2,159.73 KB |

### LCS

| length | recursive time | memo time | tabulation time | memo memory | tabulation memory |
|---|---|---|---|---|---|
| 4 | 3.05 μs | 6.76 μs | 4.38 μs | 1.27 KB | 0.26 KB |
| 12 | 11,724.92 μs | 95.79 μs | 12.95 μs | 12.50 KB | 1.45 KB |
| 500 | — | 170,496.12 μs | 22,372.52 μs | 29,098.31 KB | 2,146.42 KB |

### Graphical Performance Comparisons

The graph `fibonacci_comparison.png` shows the computation time and number of calls for the recursive, memoized, and tabulated methods, along with the recursive versus dynamic programming (DP) speedup factor. The recursive method's line splits off and terminates at n=24, while both memoized and tabulated methods display linear trends up to n=1,000. The speedup factor increases from approximately 2x at n=5 to over 4,000x at n=24. In `knapsack_performance.png`, the tabulated method's timeline drops below that of the memoised method partway through the range and remains lower thereafter. In `lcs_performance.png`, the crossover between methods is more variable for small and medium lengths, with the two methods alternating positions several times between lengths 4 and 20, before tabulation consistently surpasses memoization beyond length 50.

### Observed vs. Theoretical Complexity

Fibonacci's recursion provides the closest theoretical match in this context. The call count increases by a factor of φ≈1.618 with each increment of n. For instance, from n=20 to n=24, the number of calls increased, which is consistent with the theoretical φ⁴ value of 6.854 to three decimal places. This is expected, as the call count is a fixed function of n (2·fib(n+1)−1), independent of machine or overhead. Knapsack tabulation time increased from n=100 to n=1,000, corresponding to the O(n·capacity) prediction for a tenfold increase in n at fixed capacity. LCS memoization time grew from length 300 to 500 (a 1.67-fold increase per dimension), compared to the O(n·m) bound prediction of 2.78 times. Both algorithms remain within their expected bounds, though constant factors rise as cache or table sizes reach megabytes and memory impacts become notable. For LCS at small lengths (4–20), memoization and tabulation times are similar, often fluctuating between which is faster. At these sizes, scheduler jitter and timer resolution have a greater impact than the underlying algorithmic differences, until input magnitudes increase.

## Discussion

### When and Why DP Outperforms Recursion

Dynamic programming is most effective when the recursion tree revisits the same state multiple times, with greater benefits as this redundancy increases. For the knapsack problem with 18 items, the recursive tree made 110,577 calls to solve 3,468 distinct states, with each state requested approximately 32 times on average before its answer was cached. In contrast, for the longest common subsequence (LCS) problem of length 12, 180,802 calls covered only 201 distinct states, resulting in about 899 requests per state. This difference in state overlap explains why LCS (122x) significantly outperformed knapsack (12.3x) at comparable problem sizes. The greater the redundancy in naive recursion, the more substantial the savings achieved through memoization.

### Memoization vs. Tabulation

Tabulation is consistently faster than memoization for Fibonacci, as it uses only two integers and avoids both dictionaries and the call stack. For the Knapsack problem, the results differ: with 5 items, tabulation was slower than memoization (89.48 μs vs. 8.25 μs) and used more memory (9.49 KB vs. 3.59 KB), since tabulation allocates the entire (items+1)×(capacity+1) table up front, while memoization only stores states reached during recursion. As input size increases, this trend reverses. With 1,000 items, tabulation became 6.98 times faster and used 14.6 times less memory, due to lower per-cell overhead and the absence of function call costs. The LCS problem shows a similar pattern, though results are less consistent at small sizes. In practice, memoization is preferable when the reachable state space is much smaller than the full table, while tabulation is more efficient when most of the table is used.

### Memory Trade-offs

Fibonacci's memo cache grows linearly with n purely from dictionary bookkeeping (130.05 KB at n=1,000) against tabulation's near-constant 0.36 KB — a 358x difference for a recurrence needing only the previous two values. Knapsack and LCS's memoized caches carry the overhead of storing every visited state as a full dictionary entry, which is why, despite sharing the same big-O space bound, memoization used 14.6x more memory than tabulation for knapsack and 13.6x more for LCS at the largest sizes tested.

### Optimal Substructure and Overlapping Subproblems

All three problems have the same shape: the best answer to the whole thing is built straight out of the best answers to smaller versions of it. Fibonacci's fib(n) is just fib(n−1) + fib(n−2). Knapsack's best value at (item i, capacity w) comes from picking the better of skipping item i or taking it and recursing on (i+1, w−weight[i]). LCS either extends the LCS of both strings' shorter prefixes when the last characters match, or takes the better of the two subproblems with one character dropped. That's optimal substructure, and it's what makes a DP formulation possible at all — without it, the smaller answers wouldn't combine into the bigger one. Overlapping subproblems is a separate thing, and it's what actually makes DP worth doing: it's why the same (i, w) or (i, j) state gets solved over and over under naive recursion, and why caching each one once saves so much work.

## Conclusion

Every result here traces back to one idea: an exponential recursion tree and a polynomial DP table compute the same answer, and the only thing separating them is whether identical subproblems get solved once or over and over — roughly 899 redundant recomputations per subproblem for LCS at length 12, a 4,049x gap in wall-clock time for Fibonacci at n=24. The memoization-versus-tabulation numbers are the more interesting finding, though: both are "DP" with the same asymptotic bound, yet at small knapsack sizes tabulation was slower and hungrier for memory than memoization, before flipping decisively in tabulation's favor at scale. Asymptotic complexity says two algorithms are in the same class; it doesn't say which to use at a given input size, and that gap only shows up once you build both versions and measure them — exactly what this assignment asked for.
