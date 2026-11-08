"""
rabin_karp.py
----------------
Part 1.2 of the string search project. Rabin-Karp, the randomized
fingerprint based alternative to KMP. Instead of comparing characters
directly, it hashes every window of the text the same length as the
pattern (in O(1) per window using the rolling hash in
utils/rolling_hash.py) and only falls back to an actual character by
character comparison when a window's hash matches the pattern's hash.
On random text that comparison almost never triggers, which is what
makes the average case O(n + m), but it's still necessary, two different
strings hashing to the same value, a collision, is always possible with
a finite modulus no matter how good the hash function is, and a matcher
that reported a hit on hash equality alone would occasionally be wrong.
Every match this module reports has been verified character by
character, so correctness never depends on the hash function at all,
only speed does.

That hash-then-verify design is also exactly why this module can measure
and demonstrate collisions directly rather than just asserting they're
rare, every window where the hash matched but the string comparison
failed is a real, countable collision, and collision_rate_demo below
uses that to show how badly a weak modulus degrades Rabin-Karp toward
its O(n*m) worst case, where every window collides and every single one
needs a full verification.
"""

from utils.rolling_hash import RollingHash

# A second, independently chosen (base, modulus) pair for double hashing.
# 2^31 - 1 is also a Mersenne prime, small enough to collide more often
# than the default 2^61 - 1 modulus on its own, which is exactly why
# pairing it with the default as a second, independent check makes a
# false positive on *both* hashes at once far rarer than relying on
# either one alone.
_SECONDARY_BASE = 131
_SECONDARY_MODULUS = (1 << 31) - 1


def rabin_karp_search(text, pattern, hasher=None, double_hash=False, secondary_hasher=None):
    """
    Returns a dict:
      matches           verified start indices of `pattern` in `text`
      hash_matches      how many windows had a matching primary hash
                         (and, if double_hash, a matching secondary hash
                         too) before verification
      collisions        hash_matches that did NOT verify, real hash
                         collisions between unrelated substrings
      verifications     how many full character by character comparisons
                         were actually run (every hash match gets one)

    `double_hash=True` checks a second, independent hash in addition to
    the primary one before paying for a verification, cutting down how
    often an expensive full comparison runs without changing the final
    `matches` list at all, verification still happens and is still what
    makes the result correct.
    """
    hasher = hasher or RollingHash()
    n, m = len(text), len(pattern)
    result = {"matches": [], "hash_matches": 0, "collisions": 0, "verifications": 0}
    if m == 0:
        raise ValueError("pattern must be non-empty")
    if m > n:
        return result

    pattern_hash = hasher.hash_of(pattern)

    if double_hash:
        secondary_hasher = secondary_hasher or RollingHash(base=_SECONDARY_BASE, modulus=_SECONDARY_MODULUS)
        secondary_pattern_hash = secondary_hasher.hash_of(pattern)
        primary_windows = hasher.window_hashes(text, m)
        secondary_windows = secondary_hasher.window_hashes(text, m)
        for (start, h), (_, sh) in zip(primary_windows, secondary_windows):
            if h == pattern_hash and sh == secondary_pattern_hash:
                result["hash_matches"] += 1
                result["verifications"] += 1
                if text[start:start + m] == pattern:
                    result["matches"].append(start)
                else:
                    result["collisions"] += 1
    else:
        for start, h in hasher.window_hashes(text, m):
            if h == pattern_hash:
                result["hash_matches"] += 1
                result["verifications"] += 1
                if text[start:start + m] == pattern:
                    result["matches"].append(start)
                else:
                    result["collisions"] += 1

    return result


def rabin_karp_search_multi(text, patterns, hasher=None):
    """
    Batch matching, Rabin-Karp's strongest case, many patterns against
    one text in a single pass per distinct pattern length rather than
    one full KMP style pass per pattern. Patterns are grouped by length,
    and for each length, every window of the text that length is hashed
    once and checked against a lookup table of {hash: [patterns with
    that hash]}, so the text is scanned len(distinct_lengths) times
    total instead of len(patterns) times.

    Returns {pattern: sorted_list_of_start_indices}, every pattern from
    the input appears as a key even if it has no matches (empty list).
    """
    hasher = hasher or RollingHash()
    matches_by_pattern = {p: [] for p in patterns}

    # Dedupe before grouping, a repeated pattern string in the input list
    # would otherwise land in its length group twice, making every one
    # of its matches get appended twice over during the scan below.
    # matches_by_pattern above already collapses duplicate keys to one
    # list on its own, this just keeps the scan itself from redoing (and
    # re-counting) the same pattern's work more than once.
    by_length = {}
    for p in dict.fromkeys(patterns):
        if not p:
            raise ValueError("patterns must be non-empty")
        by_length.setdefault(len(p), []).append(p)

    for length, group in by_length.items():
        if length > len(text):
            continue
        hash_to_patterns = {}
        for p in group:
            hash_to_patterns.setdefault(hasher.hash_of(p), []).append(p)

        for start, h in hasher.window_hashes(text, length):
            candidates = hash_to_patterns.get(h)
            if not candidates:
                continue
            window = text[start:start + length]
            for p in candidates:
                if p == window:
                    matches_by_pattern[p].append(start)

    return matches_by_pattern


def collision_rate_demo(text, pattern, moduli=None):
    """
    Runs rabin_karp_search with the same base but a series of different,
    deliberately varied moduli, to demonstrate directly how much the
    modulus choice affects the collision rate. A small modulus (say,
    101) gives the hash function only 101 possible values, so on any
    text longer than a couple hundred characters, by the pigeonhole
    principle alone, most windows are guaranteed to collide with some
    other window, and a lot of those collisions will also collide with
    the pattern's hash. A large prime modulus close to 2^61 spreads
    hashes over an enormous range, making a collision with any one
    specific pattern hash astronomically unlikely on realistic text
    sizes. Returns {modulus: {"hash_matches":, "collisions":,
    "collision_rate":, "verifications":}}, collision_rate is
    collisions / hash_matches (0.0 when there were no hash matches at
    all to collide on).
    """
    moduli = moduli or [101, 1009, 104729, (1 << 31) - 1, (1 << 61) - 1]
    results = {}
    for modulus in moduli:
        hasher = RollingHash(base=257, modulus=modulus)
        outcome = rabin_karp_search(text, pattern, hasher=hasher)
        hash_matches = outcome["hash_matches"]
        collisions = outcome["collisions"]
        results[modulus] = {
            "hash_matches": hash_matches,
            "collisions": collisions,
            "collision_rate": (collisions / hash_matches) if hash_matches else 0.0,
            "verifications": outcome["verifications"],
        }
    return results
