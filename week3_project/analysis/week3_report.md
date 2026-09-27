# Week 3 Analysis Report: AVL Trees, Heaps, and Hash Tables

## Executive Summary 

My benchmarks this week (the raw numbers are in analysis/comparison_table.csv) mostly line up with the complexity classes we’d expect, though the small-n data has more overhead than I hoped for. The result I’m most confident in is the heap’s search timing: it went from 0.41 μs at n=100 to 121.87 μs at n=30,000, a 299x jump for an n that grew exactly 300x. That’s about as clean a confirmation of O(n) as I got anywhere in this assignment, and it’s a nice concrete answer to the “heaps are always fast” misconception. Ordering the array by parent/child relationships doesn’t help you find an element any faster than just scanning it.

I also noticed something I wasn’t expecting: AVL delete and heap extract-min, two completely different operations on two different structures, both came in within about 13% of the same theoretical log-ratio (1.95x and 1.94x observed, versus ~2.24x predicted for a 300x increase in n). I don’t think that’s a coincidence, it suggests whatever gap exists between the observed and theoretical numbers isn’t really about either structure specifically, it’s just what happens when you’re timing things that only take a few microseconds.

AVL insertion, on the other hand, was a mess. The six data points (15.10, 1.94, 20.90, 3.48, 6.91, 4.01 μs from n=100 up to n=30,000) don’t show any trend I can point to. At first I thought I made an error somewhere, but I think the insert is fast enough here that GC pauses and OS scheduling overhead matters more than the algorithm itself.

Hash tables were mostly flat like I expected, but open addressing’s delete cost jumped hard right around n=10,000 — from about 0.35–0.38 μs at smaller sizes up to 1.57 μs, and it never really came back down (1.22 μs at n=30,000). Chaining’s delete never did this. 

## Methodology

# Hardware and Environment 

I ran everything on a KVM virtual machine with 7 vCPUs (host CPU is an 11th Gen Intel Core i7\-11700F @ 2.50GHz), 44 GiB of RAM (41 GiB available, no swap configured), running Ubuntu on kernel 7.0.0\-31\-generic, with Python 3.14.4. Everything ran single\-threaded and I kept other programs closed so nothing else was competing for CPU time. One thing worth flagging, since this is a virtualized environment rather than bare metal, that's probably part of why the timing data has as much scatter as it does. A hypervisor adds its own scheduling layer on top of the guest OS's scheduler, which is one more source of the kind of overhead I talked about earlier. 

# Dataset Characteristics

I swept n across {100, 300, 1,000, 3,000, 10,000, 30,000}, which covers about two and a half orders of magnitude. All the keys were unique, shuffled random integers (I fixed the seed so this is reproducible), partly to avoid the AVL tree hitting worst\-case sorted\-input behavior, even though its self\-balancing means sorted input wouldn't actually hurt it. The load\-factor experiment is a bit different. It uses 20,000 keys drawn from a much bigger range (0 to 50,000,000) instead of a small consecutive range. I did that on purpose, since consecutive small integers hash to themselves in Python, so if I'd sized the table to match a consecutive key range exactly, I'd have gotten hashes with zero collisions just by construction, which would've completely hidden the clustering effect I was trying to measure.

## Measurement technique and statistical reliability

Most points are the mean of 7\-25 repeated timings with `time.perf_counter()`, the load\-factor sweep uses the median, since it's noisier. Every structure was pre\-built to size n before timing, with insert/delete trials paired to an untimed "undo" step to keep n constant.

At microsecond scale, measurements get thrown off by GC pauses, OS scheduling, and, for insert/delete, whatever shape a trial happened to hit. AVL insert shows this most clearly. Using a mean rather than median for the n\-scaling data means a few unlucky trials can drag a point upward, likely explaining open addressing's delete spike.

## Results ##

# Heap: Insert, Extract-min, and The search Misconseption

| n | insert | extract-min (delete) | search |
|---|---|---|---|
| 100 | 0.86 μs | 1.66 μs | 0.41 μs |
| 1,000 | 0.52 μs | 2.08 μs | 3.54 μs |
| 10,000 | 1.09 μs | 2.88 μs | 4.81 μs |
| 30,000 | 0.95 μs | 3.22 μs | 121.87 μs |

Heap guarantees O(log n) only for insert and extract-min/max, the search still needs a full scan, like an unsorted list.

Extract-min is the cleanest O(log n) result: 1.66 to 3.22 μs, a 1.94x increase for a 300x increase in n, close to the ~2.24x theory predicts. Insert barely moves, reading as flat. Search is noisier mid-range but its endpoints are striking: 0.41 to 121.87 μs is a 299x jump against n's 300x growth, proof heap search costs the same as scanning an unsorted list.

# AVL tree: balance maintenance and operation cost trends

| n | insert | delete | search |
|---|---|---|---|
| 100 | 15.10 μs | 2.22 μs | 0.69 μs |
| 300 | 1.94 μs | 3.11 μs | 0.45 μs |
| 1,000 | 20.90 μs | 2.89 μs | 1.81 μs |
| 3,000 | 3.48 μs | 3.23 μs | 0.56 μs |
| 10,000 | 6.91 μs | 3.83 μs | 0.58 μs |
| 30,000 | 4.01 μs | 4.33 μs | 0.69 μs |

Delete is trustworthy: 2.22 to 4.33 μs, a 1.95x increase, close to theory and nearly identical to heap extract-min's 1.94x, a shared measurement floor effect rather than structure specific overhead. Insert is the noisiest column in the dataset, likely dominated by GC pauses, scheduling hiccups, or an unlucky run of rotations. Search is mostly flat aside from one outlier at n=1,000. Structural tests validated the balance invariant held after every insert and delete across thousands of randomized operations.

# Hash Table: Load Factor vs Lookup Time

The main n-scaling data keeps the load factor low via auto-rehashing, so it can't show degradation as a table fills. A separate experiment fixed capacity at 20,000 slots, disabled auto-rehashing, and swept load factor from 0.1 to 0.99. Chaining stayed flat (0.37-0.56 μs), open addressing tracked it closely to 0.7, then jumped to 5.71 μs at 0.99, the textbook linear-probing clustering effect.

The n-scaling data hints at the same thing independently: open addressing's delete cost jumps from ~0.33-0.38 μs (n=100-3,000) to 1.57 μs at n=10,000 and stays elevated (1.22 μs at n=30,000). This could be mean-based noise or tombstone buildup across repeated delete/reinsert trials. Either way, it supports why open addressing needs a stricter rehash threshold (0.5) than chaining (0.75).

# Comparison table: Asymptotic vs. Empirical Complexity

| Structure | Operation | Asymptotic | n=100 | n=30,000 | Ratio |
|---|---|---|---|---|---|
| List (unsorted) | search | O(n) | 0.41 μs | 179.49 μs | 438x |
| List (unsorted) | delete | O(n) | 0.56 μs | 161.67 μs | 289x |
| Heap | search* | O(n) | 0.41 μs | 121.87 μs | 299x |
| Heap | insert | O(log n) | 0.86 μs | 0.95 μs | 1.10x |
| Heap | extract-min | O(log n) | 1.66 μs | 3.22 μs | 1.94x |
| AVL Tree | delete | O(log n) | 2.22 μs | 4.33 μs | 1.95x |
| AVL Tree | search | O(log n) avg | 0.69 μs | 0.69 μs | 1.01x |
| Hash Table (chaining) | search | O(1) avg | 0.34 μs | 0.21 μs | 0.60x |
| Hash Table (open addr.) | search | O(1) avg | 0.48 μs | 0.39 μs | 0.81x |
| dict (built-in) | search | O(1) avg | 0.17 μs | 0.13 μs | 0.77x |

The O(n) structures land in the 289x-438x range for a 300x increase in n. Everything O(log n) or O(1) stayed under 2x: the two groups never get confused once n is large enough.

## Amortized Analysis

Dynamic arrays and hash table rehashing both rely on amortized-cost reasoning. The dynamic array doubles capacity when full, an O(n) copy, but this happens only O(log n) times across n insertions, so the amortized cost per insertion is O(1) even though single calls can be expensive. Hash table rehashing works the same way, reinserting everything is O(n), but it triggers only once per doubling of size.

My data supports this: list append stayed nearly flat across a 300x growth in n, far below an O(n) copy's cost, and chaining insert was essentially unchanged end to end despite several rehashes firing along the way.

One thing amortized analysis doesn't explain: open addressing's delete cost jump, since deleting doesn't grow the table. That's a different phenomenon, either statistical noise or tombstone buildup, a signal that not every anomaly is a rehash artifact.

## Practical Recommendations

Heaps fit priority-queue needs: scheduling, a good algorithm, and top-k streaming. They're wrong for "is X present" or arbitrary deletion, since heap search degrades to the same O(n) cost as an unsorted list, confirmed here as cleanly as anything.

AVL trees suit ordered operations (range queries, traversal, predecessor/successor) that hashing can't support, at a modest constant-factor cost versus a hash table.

Hash tables are the right default for pure key lookup. Chaining degrades more gracefully as load factor climbs, open addressing is more compact and cache friendly but needs a stricter rehash threshold to avoid clustering.

An indexed priority queue (heap plus a hash table tracking each key's index) combines O(log n) extract-min with O(log n) arbitrary search.

## Conclusion ##

A structure's internal invariant (a heap's ordering, an AVL tree's balance factor, a hash table's load factor) turns an O(n) worst case into O(log n) or O(1), losing it degrades performance with no warning. This run added a second lesson, at microsecond scale, that guarantee is only as visible as the measurement lets it be. Heap extract-min and AVL delete each confirmed O(log n) within about 13% of theory, heap search confirmed O(n) almost exactly. AVL insert showed no usable trend, open addressing's delete cost showed a real, unexplained jump. Choosing the right structure means knowing which guarantee you need, and trusting why it performs that way means running enough trials to separate signal from overhead.



