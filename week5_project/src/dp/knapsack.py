"""
knapsack.py
-----------
Part 2 deliverable: the 0/1 knapsack problem -- given item weights, item values, and
a capacity, find the maximum total value achievable without exceeding capacity, using
each item at most once (hence "0/1": each item is either fully taken or left out,
never split or duplicated).

Provides:
    knapsack_recursive(weights, values, capacity)   -- naive recursion, O(2^n)
    knapsack_memo(weights, values, capacity)        -- top-down DP, O(n * capacity)
    knapsack_tabulation(weights, values, capacity)  -- bottom-up DP, O(n * capacity)
    trace_solution(weights, values, capacity)       -- (max_value, selected_indices),
                                                        reconstructed from the
                                                        tabulation DP table

All four assume weights/values are non-negative integers and capacity is a
non-negative integer.
"""


def knapsack_recursive(weights, values, capacity):
    """Naive recursion: at each item, branch on "take it" vs. "skip it".

    Time: O(2^n) -- n independent take/skip decisions, no memoization.
    """
    def solve(i, remaining_capacity):
        if i == len(weights):
            return 0
        best = solve(i + 1, remaining_capacity)
        if weights[i] <= remaining_capacity:
            best = max(best, values[i] + solve(i + 1, remaining_capacity - weights[i]))
        return best
    return solve(0, capacity)


def knapsack_memo(weights, values, capacity):
    """Top-down DP: same recursion, cached on (item_index, remaining_capacity).

    Time: O(n * capacity).
    """
    cache = {}

    def solve(i, remaining_capacity):
        if i == len(weights):
            return 0
        key = (i, remaining_capacity)
        if key in cache:
            return cache[key]
        best = solve(i + 1, remaining_capacity)
        if weights[i] <= remaining_capacity:
            best = max(best, values[i] + solve(i + 1, remaining_capacity - weights[i]))
        cache[key] = best
        return best

    return solve(0, capacity)


def knapsack_tabulation(weights, values, capacity):
    """Bottom-up DP: (n+1) x (capacity+1) table, dp[i][w] = best value using
    first i items with capacity w.

    Time: O(n * capacity).
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]


def trace_solution(weights, values, capacity):
    """Return (max_value, selected_indices) by walking the DP table backward:
    dp[i][w] != dp[i-1][w] means item i-1 was taken to reach this cell's value.
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
