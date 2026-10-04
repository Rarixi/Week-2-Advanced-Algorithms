# Week 6 Report Advanced Dynamic Programming

## Executive Summary

Basic dynamic programming usually means breaking a problem into overlapping subproblems and storing each answer so it never gets solved twice. This project shows three ways that core idea can be pushed further once a problem's structure is understood more deeply. Space optimization, shown in the 0/1 knapsack problem, recognizes that a dynamic program does not always need its entire table, since many problems only depend on the previous row or a small slice of earlier results, so memory use can shrink dramatically while the answer stays exactly the same. State compression, shown in the bitmask traveling salesman solution, lets the state itself be an entire subset of elements packed into a single integer, turning an otherwise factorial brute force search into an exponential but far smaller one. Interval dynamic programming, shown in matrix chain multiplication, solves a problem over every possible subrange of a sequence rather than just a prefix, the same idea behind parsing expressions and splitting a larger computation into the cheapest smaller pieces. Floyd Warshall extends this philosophy to an entire network at once, computing every shortest path between every pair of nodes in one sweep instead of solving each pair separately. Together these four problems show that advanced dynamic programming is less about new math and more about noticing extra structure that basic memoization alone never uses.

## Methodology

All four algorithms were implemented in Python with at least two versions each, checked against one another and against known textbook or hand worked examples. Running time was measured with the time module's perf counter function, and peak memory was measured separately with the tracemalloc module, so tracemalloc's own overhead would not distort timing. Every benchmark ran on my own Ubuntu machine, rari, using plain Python and matplotlib, with no outside hardware involved. Knapsack used thirty items at capacities of 10, 50, 100, 500, 1000, 5000, and 10000. Matrix chain multiplication used chains of 3, 5, 8, 10, 12, 20, 40, 80, 120, and 200 matrices, with plain recursion only run through 12 matrices since beyond that it becomes too slow to measure. Floyd Warshall used graphs of 5, 10, 20, 40, 80, 120, and 160 nodes. Traveling salesman used city counts from 4 through 16, with brute force only run through 9 cities. The Dijkstra numbers used for comparison come from the Week 4 project, benchmarked at 100, 400, and 1000 nodes across sparse, dense, random, and weighted types, each run from a single starting node.

## Results

### Standard vs Optimized DP

A clear example is shown by the two knapsack problem implementations. The standard two-dimensional (2D) version maintains a complete table with one row per item, whereas the optimized one-dimensional (1D) version retains only a single row and reuses it, iterating the capacity loop in reverse to prevent double-counting items. Matrix chain multiplication shows a similar optimization principle. In this case, plain recursion serves as the unoptimized baseline, while both memoization and tabulation represent optimized approaches. For an input of 12 matrices, tabulation completes in approximately 36 microseconds, compared to plain recursion's 19,103 microseconds, representing a speedup of over 500 times. Memoization is about 125 times faster than plain recursion for the same input size.

### Space vs Time Trade Offs

The bitmask traveling salesman solution is the strongest example of trading space for time. At 16 cities it used about 28014 kilobytes of peak memory to finish in roughly 409 milliseconds, while brute force at a much smaller 9 cities used under 1 kilobyte but still took about 31 milliseconds. The bitmask version spends far more memory holding partial results for every visited subset in exchange for skipping brute force's repeated work. Knapsack shows the same idea in reverse on a smaller scale. The 1D version used as little as about one thirtieth the memory of the 2D version at small capacities while running only slightly faster, showing that cutting memory does not always come with a matching cut in time.

### MCM vs TSP Scalability

Even after optimization, these two problems scale completely differently. Tabulated matrix chain multiplication runs in roughly cubic time, so even at 200 matrices it only took about 143 milliseconds. The bitmask traveling salesman solution runs in time proportional to n squared times 2 to the n, and even though that beats factorial brute force badly, it still grows fast enough that just 16 cities took about 409 milliseconds, more than three times longer than matrix chain multiplication's time at an input over ten times larger. That gap shows how much harder an exponential problem remains even after optimization, compared to a polynomial problem that was always going to scale well.

### Floyd Warshall vs Dijkstra from Week 4

Floyd Warshall solves every pair of nodes in one pass, while Dijkstra only solves shortest paths from one starting node, so getting every pair out of Dijkstra means running it once per node. Using the real Week 4 numbers, Dijkstra on a dense 100 node graph with 4950 edges took about 1088 microseconds from a single source, so running it from all 100 nodes would take roughly 108837 microseconds total. My Floyd Warshall benchmark on an even larger 120 node graph finished the entire all pairs computation in about 62865 microseconds, well under half that total. On a dense or nearly complete graph, one Floyd Warshall pass beats repeating Dijkstra from every node, since Floyd Warshall pays no extra cost for having that many edges while Dijkstra's heap based approach does. On a sparse graph the expectation from class is the opposite, since Dijkstra barely touches a small edge count while Floyd Warshall still pays its full cubic cost regardless, though my own sparse and dense runs were not sized closely enough to show that crossover directly.

## Discussion

### When and Why to Prioritize Space Optimization

Space optimization is worth doing whenever memory, not time, is actually running out, which tends to happen with large capacities or large tables where every extra row adds up fast. If the table only ever needs the previous row, like in knapsack, the optimization is close to free and barely changes the running time. It becomes less worth doing when the full table is needed later, such as when the project also wants to reconstruct the exact choices made, since the space optimized version throws away that history.

### Trade Offs in Readability vs Efficiency

The tabulated versions here are almost always faster than their recursive or memoized counterparts, but harder to read at a glance, since the fill order has to be worked out carefully instead of trusting the recursion to call things in the right order on its own. Memoization sits in between, keeping the natural recursive structure easy to follow while still skipping repeated work, even if it ends up a bit slower and heavier on memory than tabulation. Bitmask state is the extreme version of this trade off, since packing a whole subset into one integer is efficient but makes the code much harder to read without comments explaining what each bit means.

### Insights from State Compression and Interval DP

The biggest lesson from bitmask state compression is that a dynamic program's state does not have to be a simple index, it can be any piece of information small enough to store efficiently, including an entire set. The biggest lesson from interval dynamic programming is that some problems are naturally about subranges rather than prefixes, and forcing a prefix based recursion onto one of those problems would miss the real structure entirely.

### Real World Applications

Interval dynamic programming similar to matrix chain multiplication shows up in compilers deciding the cheapest way to parse or evaluate an expression, which is really the same subrange problem in disguise. State compression similar to bitmask traveling salesman shows up in route optimization problems like delivery planning, where a small number of stops can still be planned exactly before the problem grows too large for the trick to work. All pairs shortest path algorithms like Floyd Warshall show up in genome alignment and network analysis, anywhere a full picture of how every element relates to every other matters more than just one path at a time.

## Conclusion

Looking back at how this course started with plain recursion and basic memoization, this project feels like the point where those basic ideas start paying off in more complicated ways. Instead of just remembering an answer so it does not get recomputed, Week 6 is really about looking closely at what a dynamic program's state needs to hold onto, and realizing that state can be smaller, shaped differently, or spread across every node in a network instead of just one. None of these four algorithms needed fundamentally new math, they just needed a closer look at the structure already hiding inside the problem. That feels like the real theme of advanced dynamic programming, not new tricks so much as noticing structure that was always there and reusing what is already known about a problem as fully as possible.
