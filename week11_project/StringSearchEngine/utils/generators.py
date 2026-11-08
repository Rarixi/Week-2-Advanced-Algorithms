"""
generators.py
-----------------
Part of the string search project. Synthetic string generation, used by
the test suite for edge cases every pattern matcher needs to survive
(periodic strings, long runs of a single character) and by the benchmark
suite for datasets of a controlled size with a known ground truth answer
(a pattern planted at a known set of positions, so a benchmark run can
assert correctness, not just speed).
"""

import random
import string

DNA_ALPHABET = "ACGT"


def random_string(length, alphabet=string.ascii_lowercase, rng=None):
    """A uniformly random string over `alphabet`. Low repetition, the
    "easy" case for every algorithm here since there's rarely a long
    common prefix between unrelated windows."""
    rng = rng or random.Random()
    return "".join(rng.choice(alphabet) for _ in range(length))


def repeated_char_string(length, char="a"):
    """`char` repeated `length` times, the degenerate case that makes a
    naive matcher re-compare the same prefix over and over and is exactly
    what KMP's failure function is built to avoid redoing work on."""
    return char * length


def periodic_string(length, period="ab"):
    """`period` tiled out to `length` characters (truncating the last
    repetition if it doesn't divide evenly). Periodic strings are the
    other classic stress case alongside repeated characters, they don't
    just repeat one character, they repeat a whole block, which is the
    structure KMP's LPS table and Rabin-Karp's hash collisions both have
    to handle correctly."""
    if not period:
        raise ValueError("period must be a non-empty string")
    repeats = length // len(period) + 1
    return (period * repeats)[:length]


def random_dna_sequence(length, rng=None):
    """A random sequence over the 4 letter DNA alphabet, used by
    dna_search.py's tests and by the benchmark's genome-scale text."""
    rng = rng or random.Random()
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def text_with_planted_pattern(length, pattern, occurrences, alphabet=string.ascii_lowercase, rng=None):
    """
    Builds a length `length` random string with `pattern` planted at
    `occurrences` positions chosen up front, so a test or benchmark knows
    the exact ground truth match count and can assert every algorithm
    found exactly those positions (and nothing else, which matters just
    as much, a matcher that reports extra or missing positions on a
    planted-pattern text has a real bug, not a performance quirk).

    Returns (text, sorted_list_of_planted_start_positions). Occurrences
    that would run past the end of the text, or that would overlap an
    already-planted occurrence, are skipped, so the returned position
    list may be shorter than `occurrences` asked for on a crowded request.
    """
    rng = rng or random.Random()
    if len(pattern) > length:
        raise ValueError("pattern cannot be longer than the text")

    chars = list(random_string(length, alphabet=alphabet, rng=rng))
    planted = []
    candidates = list(range(length - len(pattern) + 1))
    rng.shuffle(candidates)

    for start in candidates:
        if len(planted) >= occurrences:
            break
        end = start + len(pattern)
        if any(not (end <= p_start or start >= p_start + len(pattern)) for p_start in planted):
            continue
        chars[start:end] = list(pattern)
        planted.append(start)

    return "".join(chars), sorted(planted)


def random_fasta_records(num_sequences, sequence_length, rng=None, header_prefix="seq"):
    """A list of (header, sequence) tuples over the DNA alphabet, the
    in-memory shape utils.text_loader.write_fasta and dna_search.py both
    expect, used to build synthetic genome files without needing a real
    one on disk."""
    rng = rng or random.Random()
    return [
        (f"{header_prefix}{i}", random_dna_sequence(sequence_length, rng=rng))
        for i in range(num_sequences)
    ]
