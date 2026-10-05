"""
space_optimized_knapsack.py
-----------------------------
Week 6: the 0/1 knapsack problem revisited, this time focused on SPACE complexity
rather than just time. Week 5's knapsack_tabulation (src/dp/knapsack.py) builds a full
(n+1) x (capacity+1) table -- O(n * capacity) memory. This file shows that the table
only ever needs its two most recent rows (row i only reads from row i-1), so the whole
thing collapses to a single 1D array of size (capacity + 1) -- O(capacity) memory,
independent of n.

Provides:
    knapsack_2d(weights, values, capacity)  -- the Week 5-style full table, O(n * capacity)
                                                 time AND space. Kept here as the baseline
                                                 this file's space optimization is measured
                                                 against.
    knapsack_1d(weights, values, capacity)  -- same O(n * capacity) TIME, but O(capacity)
                                                 SPACE: one 1D array, updated in place.
    trace_solution(weights, values, capacity) -- (max_value, selected_indices), reconstructed
                                                 from the full 2D table.

The space/time tradeoff this file demonstrates: knapsack_1d uses dramatically less
memory than knapsack_2d for the same answer, but it can no longer tell you WHICH items
were selected -- once row i-1 is overwritten by row i, that history is gone. Getting the
selected items back (trace_solution) requires the full 2D table, or a more elaborate
scheme (e.g. periodically checkpointing rows) that this file doesn't implement.
"""


def knapsack_2d(weights, values, capacity):
    """The Week 5-style full (n+1) x (capacity+1) table. dp[i][w] = best value using
    the first i items with capacity w. O(n * capacity) time AND space.

    Kept here as the baseline knapsack_1d's memory savings are measured against --
    see benchmarks/week6_dp_advanced_benchmark.py's knapsack_space_comparison.png.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]


def knapsack_1d(weights, values, capacity):
    """Space-optimized 0/1 knapsack: ONE array of size (capacity + 1) instead of a full
    (n+1) x (capacity+1) table. O(n * capacity) time (unchanged from knapsack_2d),
    O(capacity) space.

    The key trick is the direction of the inner loop: w must go from capacity DOWN to
    weights[i-1], not up. Going downward guarantees that by the time dp[w] is updated
    for item i, every dp[w'] for w' < w it might read from (dp[w - weights[i-1]]) still
    holds item (i-1)'s value, not item i's -- i.e. each item is still only ever
    considered once per capacity level, exactly like the 2D version's dp[i-1][...]
    reads. Looping upward instead would let an item be "reused" in the same pass
    (reading a cell this same item already updated), turning this into UNBOUNDED
    knapsack by accident.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


def trace_solution(weights, values, capacity):
    """Return (max_value, selected_indices) by walking the full 2D DP table backward:
    dp[i][w] != dp[i-1][w] means item i-1 was taken to reach this cell's value.

    This needs the full table from knapsack_2d -- it's the concrete illustration of
    the space/time tradeoff this file is about: knapsack_1d's memory savings come
    specifically from throwing away the history this function depends on.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    max_value = dp[n][capacity]
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()
    return max_value, selected


if __name__ == "__main__":
    # A small runnable demo: prints a comparison table and runtime summary showing
    # knapsack_2d vs. knapsack_1d on a fixed instance, so this file's own space
    # optimization claim can be checked at a glance without needing the separate
    # benchmark script in benchmarks/.
    import random
    import time
    import tracemalloc

    rng = random.Random(0)
    n_items = 25
    weights = [rng.randint(1, 50) for _ in range(n_items)]
    values = [rng.randint(1, 100) for _ in range(n_items)]

    print("Space-optimized 0/1 knapsack: 2D table vs. 1D array")
    print(f"({n_items} items, weights/values chosen at random with seed 0)\n")

    header = f"{'capacity':>10} | {'2D value':>8} | {'1D value':>8} | {'2D time (us)':>13} | {'1D time (us)':>13} | {'2D mem (KB)':>12} | {'1D mem (KB)':>12}"
    print(header)
    print("-" * len(header))

    for capacity in (50, 200, 1000, 5000):
        start = time.perf_counter()
        value_2d = knapsack_2d(weights, values, capacity)
        time_2d_us = (time.perf_counter() - start) * 1e6

        start = time.perf_counter()
        value_1d = knapsack_1d(weights, values, capacity)
        time_1d_us = (time.perf_counter() - start) * 1e6

        tracemalloc.start()
        knapsack_2d(weights, values, capacity)
        _, mem_2d_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        knapsack_1d(weights, values, capacity)
        _, mem_1d_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert value_2d == value_1d, "2D and 1D versions disagreed -- correctness bug!"

        print(
            f"{capacity:>10} | {value_2d:>8} | {value_1d:>8} | "
            f"{time_2d_us:>13.2f} | {time_1d_us:>13.2f} | "
            f"{mem_2d_peak / 1024:>12.2f} | {mem_1d_peak / 1024:>12.2f}"
        )

    print(
        "\nSummary: both versions agree on every value above (correctness preserved), "
        "while the 1D version uses dramatically less peak memory, scaling with capacity "
        "alone (O(capacity)) instead of with capacity times item count (O(n * capacity))."
    )
