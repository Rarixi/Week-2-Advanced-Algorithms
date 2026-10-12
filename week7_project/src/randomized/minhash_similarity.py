"""
minhash_similarity.py
-----------------------
Part 5: MinHash & Similarity Estimation -- estimates Jaccard similarity between two
sets (standing in for documents, represented here as sets of token/shingle strings)
without ever materializing the full intersection or union, by comparing compact
"signatures" instead.

The idea: pick k random hash functions. For each one, hash every element of a set and
keep only the MINIMUM resulting value -- that single number is "this hash function's
summary" of the whole set. It's a classical fact that, across a uniformly random hash
function, the probability two sets' minimum-hash values happen to agree is EXACTLY
their true Jaccard similarity. So running k independent hash functions and counting
what fraction of them agree gives an unbiased estimate of Jaccard similarity, and
averaging over more of them (larger k) tightens that estimate, at the cost of k times
the hashing work.

Provides:
    generate_hash_functions(num_hashes, rng)        -- k random (a, b) coefficient
                                                         pairs for k independent
                                                         universal hash functions over
                                                         a shared large prime modulus
    minhash_signature(token_set, hash_functions)     -- list of k minimum hash values,
                                                         one per hash function
    estimate_jaccard_from_signatures(sig_a, sig_b)   -- fraction of positions where
                                                         the two signatures agree
"""

import hashlib

# A prime comfortably larger than any hash value _stable_hash can produce (sha256's
# first 8 bytes, so < 2^64), making the universal hash family's modular arithmetic
# well-defined and roughly uniform.
_PRIME = 2 ** 61 - 1


def _stable_hash(token):
    """Maps a token (string or anything str()-able) to a stable, process-independent
    integer via sha256 -- needed because Python's built-in hash() for strings is
    randomly salted per interpreter run unless PYTHONHASHSEED is fixed, which would
    make signatures impossible to compare across separate runs/processes."""
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def generate_hash_functions(num_hashes, rng):
    """Returns a list of `num_hashes` (a, b) coefficient pairs, each defining one
    universal hash function h(x) = (a * x + b) mod _PRIME. a is drawn from [1,
    _PRIME - 1] (never 0, which would collapse the hash to a constant) and b from
    [0, _PRIME - 1]."""
    return [(rng.randint(1, _PRIME - 1), rng.randint(0, _PRIME - 1)) for _ in range(num_hashes)]


def minhash_signature(token_set, hash_functions):
    """Computes the MinHash signature of token_set: for each (a, b) hash function,
    hashes every token in the set and keeps the minimum result. Returns a list with
    one minimum value per hash function, in the same order as hash_functions.

    An empty set's signature is all None (there's no minimum of nothing); callers
    comparing signatures should treat a None/None match as similarity-neutral,
    mirroring true_jaccard's convention that two empty sets are perfectly similar."""
    if not token_set:
        return [None] * len(hash_functions)

    hashed_tokens = [_stable_hash(t) for t in token_set]
    signature = []
    for a, b in hash_functions:
        min_value = min((a * h + b) % _PRIME for h in hashed_tokens)
        signature.append(min_value)
    return signature


def estimate_jaccard_from_signatures(sig_a, sig_b):
    """Estimated Jaccard similarity: the fraction of positions where the two
    signatures agree. With k = len(sig_a) hash functions, this estimate's standard
    error shrinks roughly as 1 / sqrt(k) -- doubling k roughly cuts expected error by
    a factor of sqrt(2), not in half, which is why going from k=16 to k=512 helps a
    lot early on but keeps paying off at a slower and slower rate."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must come from the same hash functions (equal length)")
    if len(sig_a) == 0:
        return 1.0
    agreements = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agreements / len(sig_a)
