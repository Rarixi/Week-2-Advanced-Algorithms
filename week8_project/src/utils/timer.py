"""
timer.py
--------
Small timing helper shared across week 8's benchmarks and tests. Mirrors the
time_execution pattern used in week 7 so the rest of this project stays
consistent with the rest of the course repo.
"""

import time


def time_execution(fn, repeats=5):
    """
    Run fn() `repeats` times with no arguments, returning (mean_seconds,
    variance_seconds_squared). Uses perf_counter since it's the highest
    resolution monotonic clock available and isn't affected by system clock
    adjustments.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples.append(end - start)

    mean = sum(samples) / len(samples)
    variance = sum((s - mean) ** 2 for s in samples) / len(samples)
    return mean, variance


def time_once(fn):
    """Run fn() exactly once, returning (result, elapsed_seconds). Useful when
    a function is slow enough (or its result is needed) that repeating it
    five times per time_execution isn't practical, which comes up a lot this
    week since some of these algorithms are exponential."""
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return result, end - start
