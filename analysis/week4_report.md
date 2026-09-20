# Week 4 Analysis Report: Graph Representations, Traversal, and Shortest Paths

## Executive Summary

This week ties together Graph representation, BFS/DFS traversal, and heap-based Dijkstra, benchmarked across sparse, dense, random, and weighted graphs at n = 100, 400, 1,000, and 10,000 for comparison. The representation numbers are the best result in the whole assignment: going from n=100 to n=10,000, the adjacency list's memory grew from 54.66 to 7664.66, while the matrix's grew from 116.86 to 836098.86, and `get_neighbors()` on the matrix scaled almost exactly linearly with n 3.5 μs to 34.3 μs to 297.9 μs, while the list stayed flat from 0.11 to 0.31 μs. That's O(n) vs. O(n²) and O(1) vs. O(n) showing up in real measurements.

BFS beat DFS on every row of the traversal benchmark, not just on dense graphs. The most interesting thing I saw was memory on dense graphs at n=1000: DFS peaked at 2,056.75 vs. BFS at 49.41. The difference between the two algorithms is within the same O(V+E) space complexity. That gap traces directly to an implementation detail, not the overall theory. Dijkstra's heap-based implementation held up well against density. Sparse and dense graphs of the same size n= 1000 differed in edge count by a lot, 2,000 vs. 499,500 edges, but only in Dijkstra's computing time. Edge count alone would predict a much larger runtime increase, so the heap is absorbing most of that density increase efficiently.

## Methodology

### Hardware and Environment

I ran everything on the same machine as the Week 3 benchmarks. A KVM virtual machine with 7 vCPUs (host CPU an 11th Gen Intel Core i7-11700F @ 2.50GHz), 44 GiB RAM, Ubuntu on a 7.0.0-31-generic kernel, and Python 3.14.4 inside the project's own virtual environment (`algorithms_course/`). Single threaded, nothing else running.

### Graph Generation Strategy

The file src/utils/graph_generator.py produces four types of graph shapes, each directed and weighted unless otherwise noted. The sparse graph contains approximately two edges per node, closely resembling a forest. The dense graph includes a fixed fraction of all possible directed edges, specifically n(n-1), with a default density of 50%. The random graph is generated using the Erdos-Renyi G(n,p) model with p set to 0.05, resulting in a variable number of edges. The weighted graph maintains a sparse structure with about three edges per node but assigns weights on edges from 1 to 1000, in contrast to the range of 1 to 20, to challenge Dijkstra’s algorithm. For the BFS, DFS, and Dijkstra performance tests, the parameter n was set to 100, 400, and 1000 for each graph type, and a constant random seed of 42 was used to maintain reproducibility. For the representation benchmark, which compared list and matrix implementations, a separate fixed sparse topology with approximately two edges per node was used at n values of 100, 1,000, and 10,000 to focus on storage and lookup costs rather than graph topology.

### Tools and Libraries

Every timing number in the report comes from `time.perf_counter()` which runs 3 times per graph, and `tracemalloc` for peak memory. `networkx` + `matplotlib` are Python libraries to make the image reports and performance plots. Dijkstra uses the custom binary heap from Week 3 (`src/data_structures/heap.py`), which has no decrease-key operation, so the algorithm pushes a fresh, better entry whenever it finds a shorter distance and skips stale entries when they surface later.

## Results

### Adjacency List vs. Matrix

| n | list memory | matrix memory | list lookup | matrix lookup |
|---|---|---|---|---|
| 100 | 54.66 KB | 116.86 KB | 0.110 μs | 3.549 μs |
| 1,000 | 599.66 KB | 8,693.95 KB | 0.200 μs | 34.291 μs |
| 10,000 | 7,664.66 KB | 836,098.86 KB | 0.309 μs | 297.857 μs |

Since the matrix allocates all n² cells no matter how sparse the actual graph is, the amount of memory it uses depends on n alone. When n is 10,000 it is using about 109 times more memory than the list for the same set of edges. The lookup time tells a similar story from the other side: with the matrix the `get_neighbors()` function has to scan an entire row (O(n)) whereas with the list it returns the result directly via a dictionary (O(1)), and the approximately 10 times increase for every 10 times increase in n behaviour on the matrix side matches the prediction very closely.

### BFS vs. DFS: Time and Memory

| type | n | edges | BFS time | DFS time | BFS mem | DFS mem |
|---|---|---|---|---|---|---|
| sparse | 100 | 200 | 35.14 μs | 42.68 μs | 12.10 KB | 11.31 KB |
| sparse | 1,000 | 2,000 | 304.84 μs | 463.04 μs | 43.68 KB | 44.72 KB |
| dense | 100 | 4,950 | 114.26 μs | 214.36 μs | 11.63 KB | 30.22 KB |
| dense | 1,000 | 499,500 | 45,501.09 μs | 62,097.81 μs | 49.41 KB | 2,056.75 KB |

BFS is faster everywhere, and the gap widens with density. DFS takes about 1.2-1.5x as long as BFS on sparse graphs but 1.4-2.3x as long on dense ones. Memory is close between the two on sparse/random/weighted graphs, but on dense graphs it diverges sharply - 41.6x at n=1000.

### Dijkstra vs. Graph Density (n=1000, heap-based)

| type | edges | time | memory |
|---|---|---|---|
| sparse | 2,000 | 2,944.71 μs | 129.55 KB |
| random | 50,314 | 16,936.72 μs | 209.51 KB |
| weighted | 3,000 | 4,662.46 μs | 153.05 KB |
| dense | 499,500 | 116,656.28 μs | 296.75 KB |

The runtime closely follows the edge count, as predicted by the O(E log V) complexity, although the relationship is not strictly linear. For example, the dense graph contains 250 times more edges than the sparse graph, yet its runtime increases by only approximately 40 times.

### Observed vs. Theoretical Complexity

The cleanest confirmation is the matrix lookup scaling above. Traversal time was noisier: regarding sparse graphs (where V+E scales exactly 10x from n=100 to n=1000), BFS's observed 8.7x growth undershoots theory slightly, but on dense graphs, where V+E only grows ~99x, BFS's observed growth was ~398x and DFS's ~290x - both far above what O(V+E) alone predicts. Dijkstra ran the other direction: theory predicts ~15x growth regarding sparse (E log V) and observed was ~11x; for dense, theory predicts ~151x and observed was ~107x, both a consistent ~70% of the prediction. None of this breaks the asymptotic bounds - it's what happens when Python-level costs (dict growth, cache locality, GC) don't stay perfectly constant per operation the way idealized complexity analysis assumes.

## Discussion

### Why BFS Outperformed DFS

The observed difference does not primarily concern the breadth-first versus depth-first approach. In `bfs.py`, a node is marked as visited immediately upon being enqueued, making sure that each node enters the queue exactly once and the queue size stays limited by O(V). In contrast, the iterative version of `dfs.py` tags a node as visited only when it is popped from the stack. Consequently, a node may be pushed onto the stack once for each incoming edge discovered before its first pop, with duplicate entries being skipped subsequently. In dense graphs with high average in-degree, this results in a significant number of additional pushes; the stack may contain on the order of E entries rather than V. This behavior accounts for the 41.6-fold increase in memory usage observed for DFS at n=1000 on dense graphs. Although both algorithms maintain O(V+E) asymptotic complexity, the constant factor for DFS in this context scales with the number of edges instead of only with the number of nodes.

### How Density Affects Performance

Graphs with few edges are the regime where V+E behaves almost like V .Dijkstra, BFS, and DFS all look close to linear in node count. As density rises, edges dominate, and the cost of an operation like DFS's stack pushes or the matrix's row scan starts scaling with degree rather than staying flat. Dijkstra handled that increase better than BFS/DFS did in relative terms (40x runtime for 250x edges vs. BFS's disproportionate scaling on dense graphs) - one plausible reason is that a node's outgoing edges only get relaxed once it's popped off the heap, and once a node is finalized, later attempts to reach it are skipped outright, capping some of the redundant work that a dense graph would otherwise generate.

### Practical Implications

The representation numbers alone make the case for adjacency lists at scale: extrapolating the matrix's O(n²) growth, a graph of a million nodes would need on the order of terabytes just to allocate the matrix, which is why every real routing engine, social graph, or AI planning graph uses list-like sparse representations. BFS's cost advantage matters most in unweighted "fewest hops" queries - social network degrees-of-separation, unweighted routing - while Dijkstra's relative resilience to density is reassuring pertaining to weighted shortest-path use cases, though its absolute cost on genuinely dense graphs (116 ms at just 1,000 nodes) is a real argument for A* or bidirectional search once graphs get both dense and large.

## Visualization Summary

`bfs_dfs_time_vs_n.png` and `bfs_dfs_memory_vs_n.png` plot sparse vs. dense, BFS vs. DFS, on log-scaled axes - both make DFS's dense-graph disadvantage visually obvious as the widening gap between the solid and dashed red/green lines. `dijkstra_density_comparison.png` is a grouped bar chart across all four graph shapes and three sizes, showing dense consistently costliest and sparse cheapest at every n. `bfs_traversal_order.png` and `dfs_traversal_order.png` visualize the same 12-node directed graph, colored by visit order - they make the structural difference concrete: BFS's coloring fans out in rings from the source, while DFS's follows one branch to its end before backtracking, most visible in nodes 5 and 7 getting visited in a different relative order between the two. `representation_comparison.png` puts the list-vs-matrix memory and lookup numbers on log-log axes, where the O(n) vs. O(n²) and O(1) vs. O(n) lines are visually unmistakable.

## Conclusion

Every result here comes back to the same idea: asymptotic complexity sets the ceiling, but the constant factors underneath it are decided by implementation choices, and those choices start mattering as soon as a graph gets big or dense. A matrix and a list both satisfy the Graph interface, but only one survives past a few thousand nodes. BFS and DFS share an O(V+E) bound, but marking nodes visited at enqueue-time versus pop-time is the difference between a queue bounded by V and a stack that can balloon toward E. Dijkstra's heap keeps shortest-path search efficient even as density climbs, though not for free. None of that is visible from the Big-O notation alone - it only shows up once you actually build the graphs and measure them, which is the entire point of doing the benchmarking rather than just trusting the complexity table in `graph.py`'s docstring.
