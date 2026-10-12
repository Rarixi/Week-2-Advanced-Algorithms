"""
random_utils.py
-----------------
Small shared helpers for generating randomized test/benchmark input across all five
Week 7 algorithms. Centralized here so every algorithm and the benchmark script pull
from the same seeded RNG conventions instead of each rolling its own.

Provides:
    make_rng(seed)                              -- a seeded random.Random instance
    random_int_array(n, low, high, rng)         -- list[int] of length n
    random_permutation_array(n, rng)            -- list[int], a permutation of 0..n-1
    worst_case_sorted_array(n)                  -- already-sorted list (quicksort's
                                                    worst case if a naive pivot rule,
                                                    like "always first element", is used)
    random_token_set(vocab, size, rng)          -- a set[str] of distinct tokens drawn
                                                    from a shared vocabulary, for MinHash
    related_token_set(base_set, vocab, overlap_fraction, size, rng)
                                                    -- a second set sharing roughly
                                                    overlap_fraction of base_set's
                                                    elements, for testing MinHash/Jaccard
    large_random_token_set(size, rng)           -- a set[str] of `size` distinct
                                                    tokens drawn from a huge integer
                                                    domain instead of the small fixed
                                                    vocabulary below, for speed
                                                    benchmarks that need sets far
                                                    bigger than that vocabulary could
                                                    ever supply
                                                    at a known, controlled similarity
"""

import random
import string


def make_rng(seed=None):
    """Returns a seeded random.Random instance. seed=None lets Python pick its own
    (non-reproducible) seed, which benchmarks avoid but exploratory use might want."""
    return random.Random(seed)


def random_int_array(n, low, high, rng):
    """n random integers in [low, high], inclusive, as a plain list."""
    return [rng.randint(low, high) for _ in range(n)]


def random_permutation_array(n, rng):
    """A random permutation of 0..n-1 -- distinct elements, so rank-k queries (used by
    randomized_selection's tests) always have an unambiguous answer."""
    arr = list(range(n))
    rng.shuffle(arr)
    return arr


def worst_case_sorted_array(n):
    """A plain ascending list of length n. Deterministic quicksort variants that always
    pick a fixed position (first or last element) as the pivot degrade to O(n^2) on
    input like this, which is exactly the scenario randomized pivoting is meant to
    avoid."""
    return list(range(n))


_VOCAB_DEFAULT = [
    f"{letter}{num}" for letter in string.ascii_lowercase for num in range(40)
]
# ^ 1040 short synthetic "tokens" (a0, a1, ..., z39) standing in for words/shingles --
#   enough to draw varied random subsets from without needing a real text corpus.


def random_token_set(size, rng, vocab=None):
    """A set of `size` distinct tokens drawn from `vocab` (or the default synthetic
    vocabulary), simulating one document's token/shingle set for MinHash."""
    vocab = vocab if vocab is not None else _VOCAB_DEFAULT
    size = min(size, len(vocab))
    return set(rng.sample(vocab, size))


def related_token_set(base_set, target_size, overlap_fraction, rng, vocab=None):
    """Builds a second token set that shares roughly `overlap_fraction` of its elements
    with `base_set`, so true Jaccard similarity can be controlled and known ahead of
    time rather than left to chance -- useful for checking that MinHash's *estimated*
    similarity actually tracks a *known* true similarity as k varies."""
    vocab = vocab if vocab is not None else _VOCAB_DEFAULT
    # sorted(), not list(): plain `list(a_set)` iterates in an order driven by each
    # element's hash, which for strings is randomized differently every interpreter
    # process (PYTHONHASHSEED) unless pinned -- that would make rng.sample() draw a
    # DIFFERENT subset on every run even with the same rng seed, silently breaking
    # reproducibility. Sorting first makes the order (and therefore the sample)
    # depend only on rng's seed, exactly as intended.
    base_list = sorted(base_set)
    n_shared = min(int(round(target_size * overlap_fraction)), len(base_list))
    shared = set(rng.sample(base_list, n_shared)) if n_shared > 0 else set()

    remaining_needed = target_size - len(shared)
    outside_vocab = [t for t in vocab if t not in shared]
    extra = set(rng.sample(outside_vocab, min(remaining_needed, len(outside_vocab))))
    return shared | extra


def large_random_token_set(size, rng):
    """A set of `size` distinct tokens drawn from a huge integer domain (0 to 10
    billion), rather than from the small ~1000-token _VOCAB_DEFAULT random_token_set
    above draws from. Needed for benchmarks (like comparing MinHash's speed against
    computing exact Jaccard directly) that need sets with tens or hundreds of
    thousands of distinct elements, where exact token identity doesn't matter, only
    having `size` genuinely distinct elements. Collisions are vanishingly unlikely at
    this domain size even for the largest sets this project benchmarks, so plain
    rejection sampling (keep drawing until the set is full) finishes in essentially
    `size` draws."""
    tokens = set()
    while len(tokens) < size:
        tokens.add(str(rng.randint(0, 10_000_000_000)))
    return tokens
