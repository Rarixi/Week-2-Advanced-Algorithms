# Week 6 Report Advanced Dynamic Programming

## Executive Summary

Up until now, dynamic programming in this course has mostly meant one thing, break a problem into smaller overlapping pieces and save each answer so you never solve the same piece twice. This week showed me three ways that idea can be pushed further once you look closely at a problem's structure. The first is space optimization, used on the 0/1 knapsack problem. A dynamic program does not always need its whole table. A lot of the time it only needs the row right before the current one, so memory use can shrink a ton without changing the answer. The second is state compression, used in the bitmask traveling salesman problem. Instead of the state being a simple index, it can be an entire group of cities packed into one integer using bits, turning an impossible factorial search into something merely exponential, which sounds bad but is actually a massive improvement. The third is interval dynamic programming, used for matrix chain multiplication, solving a problem over every possible chunk in the middle instead of a prefix, the same idea behind how a compiler groups operations. On top of that, Floyd Warshall stretches the same idea across a whole network at once, finding every shortest path between every pair of nodes in one pass instead of one at a time. None of these four things needed brand new math. They just needed me to notice more of the structure already there.

## Methodology

I built at least two versions of every algorithm so I could check them against each other, plus a known textbook example or a case I worked out by hand. For timing I used perf counter, and for memory I used tracemalloc, run separately so checking memory would not mess with the timing numbers. Everything ran on my own Ubuntu machine, rari, with regular Python and matplotlib, nothing cloud based. For knapsack I used thirty items at capacities of 10, 50, 100, 500, 1000, 5000, and 10000. For matrix chain multiplication I tested chain sizes of 3, 5, 8, 10, 12, 20, 40, 80, 120, and 200, only letting plain recursion run through 12 since past that it gets too slow to wait for. For Floyd Warshall I tested graphs with 5, 10, 20, 40, 80, 120, and 160 nodes. For traveling salesman I tested city counts from 4 up to 16, only letting brute force run through 9. The Dijkstra numbers I used for comparison come from my Week 4 project, already benchmarked at 100, 400, and 1000 nodes across sparse, dense, random, and weighted types, each run from one starting node.

## Results

### Standard vs Optimized DP

This distinction is visible in the two implementations of the knapsack problem. The standard two-dimensional (2D) approach upholds a complete table with a row for each item. In contrast, the optimized one-dimensional (1D)version uses a single row, updating it in reverse order to prevent multiple uses of the same item. A similar principle applies to matrix chain multiplication. Plain recursion repeatedly solves identical subproblems and serves as the unoptimized baseline. Memoization and tabulation are optimized strategies. For instance, with 12 matrices, tabulation completed in approximately 36 microseconds, while plain recursion required about 19,103 microseconds, resulting in a speedup of over 500 times. Memoization improved performance by approximately 125 times compared to plain recursion for the same input size.

### Space vs Time Trade Offs

Bitmask traveling salesman shows this trade off best. At 16 cities it used about 28014 kilobytes of memory to finish in roughly 409 milliseconds, while brute force at a much smaller 9 cities used under 1 kilobyte but still took about 31 milliseconds just from checking so many orderings. The bitmask version spends more memory holding partial answers for every visited subset, and in return skips the repeated work brute force keeps doing. Knapsack shows a smaller version of the same idea flipped around. The 1D version used as little as about one thirtieth the memory of the 2D version at smaller capacities while only running a bit faster, telling me saving memory does not always mean saving much time too. Sometimes the memory savings are the whole point on their own.

### MCM vs TSP Scalability

Even once both are optimized, these two problems grow at totally different speeds. Tabulated matrix chain multiplication runs in roughly cubic time, so even at 200 matrices it only took about 143 milliseconds. The bitmask traveling salesman version runs in time proportional to n squared times 2 to the n, and even though that destroys brute force, it still grows fast enough that just 16 cities took about 409 milliseconds, over three times longer than matrix chain multiplication at a far bigger input. That gap shows how much harder an exponential problem stays even after optimization, compared to a polynomial problem that was always going to behave well.

### Floyd Warshall vs Dijkstra from Week 4

Floyd Warshall solves every pair of nodes in a single pass, while Dijkstra only solves shortest paths from one node, so getting every pair out of Dijkstra means running it once per node. Using my real Week 4 numbers, Dijkstra on a dense 100 node graph with 4950 edges took about 1088 microseconds from one source, so running it from all 100 nodes would add up to roughly 108837 microseconds total. My Floyd Warshall benchmark, run on an even bigger 120 node graph, finished the whole all pairs computation in about 62865 microseconds, under half of that. So on a dense graph, one Floyd Warshall pass beats running Dijkstra from every node, since it never pays extra for having that many edges while Dijkstra's heap based approach does. On a sparse graph I would expect the opposite based on class, though I did not test both at matching sparse sizes to show that crossover myself.

## Discussion

### When and Why to Prioritize Space Optimization

Space optimization makes the most sense when memory, not time, is the actual problem, which tends to happen with bigger capacities or tables where every extra row adds up fast. If a table only needs the row right before it, like in knapsack, the optimization is basically free and barely touches running time. It matters less, or can even hurt, when you need the full table later, like reconstructing the exact choices made, since the space optimized version throws that history away on purpose.

### Trade Offs in Readability vs Efficiency

The tabulated versions here are almost always fastest, but also harder to read quickly, since you have to work out the fill order carefully instead of letting the recursion figure out the order for you. Memoization sits in the middle, keeping the natural recursive shape that is easy to follow while still skipping repeated work, even if slower and heavier on memory than tabulation. Bitmask state takes this trade off the furthest, since packing a whole group of cities into one integer is efficient but makes the code rough to read without good comments explaining each bit.

### Insights from State Compression and Interval DP

The biggest thing I took from bitmask state compression is that a dynamic program's state does not have to be a plain index, it can be almost any piece of information small enough to store, including an entire set. The biggest thing from interval dynamic programming is that some problems are really about chunks in the middle of a sequence rather than a prefix, and forcing a prefix style recursion onto one of those would miss what is actually going on.

### Real World Applications

Interval dynamic programming like matrix chain multiplication shows up in compilers deciding the cheapest way to parse an expression, the same chunk splitting problem in a different outfit. State compression like bitmask traveling salesman shows up in route planning like delivery routing, where a small number of stops can still be solved exactly before the trick stops working. All pairs shortest path algorithms like Floyd Warshall show up in genome alignment and network analysis, anywhere the full picture of how everything relates to everything else matters more than one path at a time.

## Conclusion

Looking back at how this course started with plain recursion and basic memoization, this project feels like the point where those early ideas start paying off in bigger ways. It is not just about remembering an answer so you do not solve it twice anymore. It is about looking closely at what a dynamic program's state actually needs to hold onto, and realizing that state can be smaller, shaped differently, or spread across a whole network instead of staying tied to one starting point. None of these four algorithms needed anything mathematically new. They just needed a closer look at the structure already hiding inside the problem, which feels like the real lesson here, not learning brand new tricks so much as noticing structure that was there all along and reusing what I already know as fully as I can.
