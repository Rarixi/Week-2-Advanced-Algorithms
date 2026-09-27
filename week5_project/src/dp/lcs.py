"""
lcs.py
------
Part 3 deliverable: Longest Common Subsequence -- the length (and one example) of the
longest sequence of characters that appears, in order but not necessarily
contiguously, in both of two input strings.

Provides:
    lcs_recursive(x, y)   -- naive recursion, O(2^(n+m))
    lcs_memo(x, y)        -- top-down DP, O(n * m)
    lcs_tabulation(x, y)  -- bottom-up DP, O(n * m)
    lcs_reconstruct(x, y) -- one actual longest common subsequence (a string),
                             reconstructed from the tabulation DP table

n = len(x), m = len(y) throughout. When multiple longest common subsequences exist
(ties), lcs_reconstruct returns whichever one its backward walk happens to find
first -- any of them is equally valid.
"""


def lcs_recursive(x, y):
    """Naive recursion on (i, j): if last chars match, recurse on both shortened;
    otherwise take the best of dropping the last char of either string.

    Time: O(2^(n+m)).
    """
    def solve(i, j):
        if i == 0 or j == 0:
            return 0
        if x[i - 1] == y[j - 1]:
            return 1 + solve(i - 1, j - 1)
        return max(solve(i - 1, j), solve(i, j - 1))
    return solve(len(x), len(y))


def lcs_memo(x, y):
    """Top-down DP: same recursion, cached on (i, j).

    Time: O(n * m).
    """
    cache = {}

    def solve(i, j):
        if i == 0 or j == 0:
            return 0
        if (i, j) in cache:
            return cache[(i, j)]
        if x[i - 1] == y[j - 1]:
            result = 1 + solve(i - 1, j - 1)
        else:
            result = max(solve(i - 1, j), solve(i, j - 1))
        cache[(i, j)] = result
        return result

    return solve(len(x), len(y))


def lcs_tabulation(x, y):
    """Bottom-up DP: (n+1) x (m+1) table, dp[i][j] = LCS length of x[:i], y[:j].

    Time: O(n * m).
    """
    n, m = len(x), len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def lcs_reconstruct(x, y):
    """Walk the DP table backward from dp[n][m]: when characters match, that
    char is part of the LCS and both indices step back together; otherwise
    step toward whichever neighbor matches the current cell's value.
    """
    n, m = len(x), len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    chars = []
    i, j = n, m
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            chars.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    chars.reverse()
    return "".join(chars)
