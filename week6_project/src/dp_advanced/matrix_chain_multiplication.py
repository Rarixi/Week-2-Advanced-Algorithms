"""
matrix_chain_multiplication.py
--------------------------------
Matrix Chain Multiplication (MCM): given a chain of matrices A1, A2, ..., An with
compatible dimensions, find the cheapest order in which to parenthesize their
multiplication -- matrix multiplication is associative, so every parenthesization
produces the same final matrix, but the NUMBER of scalar multiplications needed to get
there can differ enormously depending on grouping.

Matrices are described by a single `dims` list of length n+1: matrix i (1-indexed,
i = 1..n) has shape dims[i-1] x dims[i]. This is the standard compact encoding -- it
only needs to store each shared dimension once, since matrix i's column count must
equal matrix i+1's row count for the chain to be valid at all.

Provides:
    mcm_recursive(dims)             -- naive recursion over every possible split point,
                                        exponential (specifically Catalan-number growth)
    mcm_memo(dims)                  -- top-down DP, O(n^3) time, O(n^2) space
    mcm_tabulation(dims)            -- bottom-up DP, O(n^3) time, O(n^2) space
    optimal_parenthesization(dims)  -- (min_cost, string), the string showing exactly
                                        where to place parentheses to achieve min_cost

All four take `dims`, a list of n+1 positive integers for a chain of n matrices, and
assume n >= 1 (a chain of at least one matrix).
"""


def mcm_recursive(dims):
    """Naive recursion: for the chain of matrices i..j, try every possible split
    point k and recursively solve both halves, taking the cheapest split.

    Time: exponential. Specifically, the number of ways to parenthesize a chain of n
    matrices is the (n-1)th Catalan number, and naive recursion re-explores the same
    sub-chains over and over without caching -- the "overlapping subproblems" this
    file's memo/tabulation versions exploit.
    """
    n = len(dims) - 1

    def solve(i, j):
        if i == j:
            return 0
        best = float("inf")
        for k in range(i, j):
            cost = solve(i, k) + solve(k + 1, j) + dims[i - 1] * dims[k] * dims[j]
            best = min(best, cost)
        return best

    return solve(1, n)


def mcm_memo(dims):
    """Top-down DP: same recursion as mcm_recursive, cached on (i, j) -- the matrix
    sub-chain being solved. O(n^3) time (O(n^2) distinct (i, j) states, each doing an
    O(n) loop over split points k), O(n^2) space for the cache.
    """
    n = len(dims) - 1
    cache = {}

    def solve(i, j):
        if i == j:
            return 0
        if (i, j) in cache:
            return cache[(i, j)]
        best = float("inf")
        for k in range(i, j):
            cost = solve(i, k) + solve(k + 1, j) + dims[i - 1] * dims[k] * dims[j]
            best = min(best, cost)
        cache[(i, j)] = best
        return best

    return solve(1, n)


def mcm_tabulation(dims):
    """Bottom-up DP: dp[i][j] = minimum cost to multiply matrices i..j, filled by
    increasing CHAIN LENGTH (j - i) rather than by row or column -- dp[i][j] depends on
    dp[i][k] and dp[k+1][j] for every split k in between, both of which are shorter
    sub-chains, so every shorter chain length must already be filled in before a
    longer one can be computed. O(n^3) time, O(n^2) space.
    """
    n = len(dims) - 1
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for chain_len in range(2, n + 1):
        for i in range(1, n - chain_len + 2):
            j = i + chain_len - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost

    return dp[1][n]


def optimal_parenthesization(dims):
    """Return (min_cost, parenthesization_string) -- e.g. "((A1 A2) (A3 A4))" for a
    4-matrix chain. Builds the same dp[][] table as mcm_tabulation, but also tracks
    split[i][j] = the k that achieved dp[i][j]'s minimum, then reconstructs the
    optimal grouping by recursively splitting at those recorded points.
    """
    n = len(dims) - 1
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    split = [[0] * (n + 1) for _ in range(n + 1)]

    for chain_len in range(2, n + 1):
        for i in range(1, n - chain_len + 2):
            j = i + chain_len - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    def build(i, j):
        if i == j:
            return f"A{i}"
        k = split[i][j]
        return f"({build(i, k)} {build(k + 1, j)})"

    return dp[1][n], build(1, n)
