"""
probability_analysis.py
-------------------------
Shared math for the "how well does the theory match the measurements" side of this
week's report -- expected comparison counts, error probability bounds, and the true
Jaccard calculation MinHash's estimate gets checked against. Kept separate from the
algorithm files themselves so an algorithm file only has to implement the algorithm,
not also know how to grade itself.

Provides:
    expected_quicksort_comparisons(n)        -- theoretical ~2n ln(n) for randomized
                                                 quicksort's expected comparison count
    variance(values)                          -- plain sample variance, no numpy
    miller_rabin_error_bound(k)              -- worst-case false-positive probability
                                                 upper bound, (1/4)^k
    fermat_error_bound(k)                     -- a much looser bound (Fermat has no
                                                 clean closed form like Miller-Rabin's;
                                                 this is only a rough reference point)
    true_jaccard(set_a, set_b)                -- |intersection| / |union|
    estimation_error(true_value, estimate)    -- absolute error, |true - estimate|
"""

import math


def expected_quicksort_comparisons(n):
    """Theoretical expected number of comparisons for randomized quicksort on n
    elements: 2n ln(n) + O(n), from the standard indicator-variable argument (every
    pair of elements is compared at most once, and the probability that a given pair
    IS compared works out to 2 / (rank_difference + 1), summed over all pairs). Used
    as the reference line the measured comparison counts get checked against."""
    if n <= 1:
        return 0.0
    return 2 * n * math.log(n)


def variance(values):
    """Plain sample variance (no numpy dependency needed for this one calculation).
    Returns 0.0 for fewer than 2 values, since variance is undefined for a single
    sample."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def miller_rabin_error_bound(k):
    """Worst-case upper bound on the probability that Miller-Rabin's k rounds all
    falsely call a composite number "probably prime": at most (1/4)^k for ANY
    composite, a much tighter guarantee than Fermat's test has (some composites,
    Carmichael numbers, fool Fermat for almost every base)."""
    return 0.25 ** k


def fermat_error_bound(k):
    """A rough reference bound for Fermat's test, (1/2)^k, which is optimistic for
    most composites but does NOT hold for Carmichael numbers, where Fermat's test can
    have a much higher, sometimes near-100%, failure rate regardless of k. This bound
    is only meant as a loose point of comparison against Miller-Rabin's bound, not a
    guarantee -- the benchmark's empirical measurements are what actually matter."""
    return 0.5 ** k


def true_jaccard(set_a, set_b):
    """Exact Jaccard similarity |A intersect B| / |A union B|. Two empty sets are
    defined as perfectly similar (similarity 1.0) rather than dividing by zero, since
    they share "nothing to disagree on."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def estimation_error(true_value, estimate):
    """Plain absolute error between a true value and an estimate of it."""
    return abs(true_value - estimate)
