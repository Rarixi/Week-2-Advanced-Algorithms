"""
fibonacci.py
------------
Part 1 deliverable: three ways to compute the nth Fibonacci number (fib(0)=0,
fib(1)=1, fib(n)=fib(n-1)+fib(n-2)), demonstrating the jump from exponential to
polynomial time that dynamic programming buys.

Provides:
    fibonacci_recursive(n)   -- naive recursion, O(2^n) time, O(n) call-stack depth
    fibonacci_memo(n)        -- top-down DP (memoization), O(n) time, O(n) space
    fibonacci_tabulation(n)  -- bottom-up DP (tabulation), O(n) time, O(1) space

All three raise ValueError for negative n (Fibonacci isn't defined there) and return
the same int for the same n -- see benchmarks/week5_dp_benchmark.py for the
execution-time / call-count comparison that demonstrates the practical difference
between these three approaches. fibonacci_recursive is only benchmarked up to n=24
there (2^24 is already ~150K calls; n=40 or 45 would take far too long to be worth
running) -- fibonacci_memo and fibonacci_tabulation are benchmarked up to n=1000 to
also show their own scalability well past where naive recursion becomes impractical.
"""


def fibonacci_recursive(n):
    """Naive recursion: fib(n) = fib(n-1) + fib(n-2), with no memory of past results.

    Time: O(2^n) -- every call except the two base cases spawns two more calls, and
    most of those calls recompute values already computed higher up the call tree
    (e.g. computing fib(5) recomputes fib(3) twice and fib(2) three times). Space:
    O(n) for the call stack's depth, since only one branch of the recursion tree is
    "open" on the stack at any given moment.
    """
    if n < 0:
        raise ValueError(f"fibonacci is not defined for negative n (got {n})")
    if n in (0, 1):
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_memo(n, _cache=None):
    """Top-down DP: same recursive structure as fibonacci_recursive, but every
    result is cached the first time it's computed, so each fib(k) for k in 0..n is
    computed exactly once instead of exponentially many times.

    Time: O(n) -- n distinct subproblems, each solved once. Space: O(n) for the
    cache plus O(n) call-stack depth.

    `_cache` is a private, internal parameter used only so the recursive calls can
    share one dict across the whole call tree -- callers should never pass it
    themselves. It defaults to None rather than {} because a mutable default
    argument would be shared across EVERY call to this function for the life of the
    program; using None and creating a fresh dict per top-level call keeps separate
    calls to fibonacci_memo() fully independent of each other.
    """
    if n < 0:
        raise ValueError(f"fibonacci is not defined for negative n (got {n})")
    if _cache is None:
        _cache = {}
    if n in (0, 1):
        return n
    if n in _cache:
        return _cache[n]
    result = fibonacci_memo(n - 1, _cache) + fibonacci_memo(n - 2, _cache)
    _cache[n] = result
    return result


def fibonacci_tabulation(n):
    """Bottom-up DP: build the sequence forward from fib(0)/fib(1) up to fib(n) in a
    simple loop, keeping only the last two values instead of a whole table.

    Time: O(n) -- one pass, constant work per step. Space: O(1) -- unlike
    fibonacci_memo, nothing here grows with n. This is also why tabulation has no
    recursion-depth limit to worry about, unlike the other two versions.
    """
    if n < 0:
        raise ValueError(f"fibonacci is not defined for negative n (got {n})")
    if n in (0, 1):
        return n
    previous, current = 0, 1
    for _ in range(2, n + 1):
        previous, current = current, previous + current
    return current
