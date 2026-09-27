"""
timer.py
--------
Shared measurement helpers for benchmarks/week5_dp_benchmark.py: wall-clock timing,
peak memory, and call-count / recursion-depth instrumentation -- kept here instead of
inside fibonacci.py/knapsack.py/lcs.py so the DP implementations' public interfaces
stay free of anything benchmarking-related (same principle as Week 4's bfs_full/
dfs_full/dijkstra, which don't self-instrument either).

Provides:
    time_execution(fn, repeats=3)     -> mean wall-clock time in seconds
    measure_peak_memory(fn)           -> peak traced memory in bytes
    count_calls_and_depth(fn)         -> (result, call_count, max_depth)
"""

import sys
import time
import tracemalloc


def time_execution(fn, repeats=3):
    """Mean wall-clock time (seconds) of `fn()` over `repeats` calls. `fn` takes no
    arguments -- callers wrap whatever they're timing in a small lambda/closure."""
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        total += time.perf_counter() - start
    return total / repeats


def measure_peak_memory(fn):
    """Peak traced memory (bytes) of ONE call to fn(), measured in its own tracemalloc
    session, kept separate from time_execution so tracemalloc's own bookkeeping doesn't
    inflate the timing numbers (same discipline used in Week 4's benchmark)."""
    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def count_calls_and_depth(fn):
    """Runs fn() under a Python-level trace hook and returns (result, call_count,
    max_depth): the total number of function calls made anywhere during fn(), and the
    deepest the call stack got (counting from fn()'s own call as depth 1).

    This uses sys.settrace rather than wrapping the target algorithm function directly
    because knapsack_recursive/knapsack_memo and lcs_recursive/lcs_memo do their actual
    recursion inside a private nested `solve` closure -- from outside, only ONE call to
    e.g. knapsack_recursive is ever visible, so counting calls to the public function
    itself would always just say "1" and completely miss the exponential blowup the
    recursion actually does internally. settrace sees every frame Python creates,
    including ones hidden inside a closure, without requiring any change to
    fibonacci.py/knapsack.py/lcs.py's public interfaces.

    Note: this adds real overhead (a Python-level hook fires on every call/return), so
    it's noticeably slower than an untraced run -- that's why week5_dp_benchmark.py
    calls this separately from time_execution instead of trying to get both readings
    from the same run.
    """
    call_count = 0
    depth = 0
    max_depth = 0

    def tracer(frame, event, arg):
        nonlocal call_count, depth, max_depth
        if event == "call":
            call_count += 1
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif event == "return":
            depth -= 1
        return tracer

    sys.settrace(tracer)
    try:
        result = fn()
    finally:
        sys.settrace(None)
    return result, call_count, max_depth
