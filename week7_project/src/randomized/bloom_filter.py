"""
bloom_filter.py
-----------------
Part 4: Bloom Filter -- a space-efficient probabilistic set membership structure. A
Bloom filter can have false positives ("probably in the set" when it actually isn't)
but never false negatives (if it says "definitely not in the set", that's always
correct) -- the randomness here lives in the k hash functions spreading items across
the bit array, not in any random choice the filter makes at query time.

Uses the classic "double hashing" trick (Kirsch-Mitzenmacher) to simulate k
independent hash functions from only two real hash computations per item:
    h_i(item) = (h1(item) + i * h2(item)) mod m
which is both faster than computing k separate hashes and still spreads items well
enough in practice for a filter of this kind.

Provides:
    optimal_parameters(expected_items, target_fp_rate)  -- (m, k), the standard
                                                             formulas for bit array
                                                             size and hash count that
                                                             MINIMIZE the false
                                                             positive rate for a given
                                                             expected item count
    class BloomFilter
        add(item)                   -- insert item (sets k bits)
        check(item)                 -- True = "probably present", False = "definitely
                                        absent"
        current_load_factor         -- property: items_added / bit_array_size
        estimated_false_positive_rate  -- property: theoretical (1 - e^(-kn/m))^k,
                                           using the CURRENT item count n
"""

import hashlib
import math


def optimal_parameters(expected_items, target_fp_rate):
    """Standard formulas for the bit array size m and hash function count k that
    minimize the false positive rate for `expected_items` insertions at a target
    false positive rate:
        m = ceil(-n * ln(p) / (ln 2)^2)
        k = round((m / n) * ln 2)
    Returns (m, k), each clamped to be at least 1."""
    n = max(expected_items, 1)
    p = min(max(target_fp_rate, 1e-9), 0.5)
    m = math.ceil(-n * math.log(p) / (math.log(2) ** 2))
    k = max(1, round((m / n) * math.log(2)))
    return max(m, 1), k


class BloomFilter:
    """A Bloom filter backed by a plain Python list of 0/1 ints as the bit array
    (simple and transparent for this assignment's purposes -- a real-world filter
    would pack bits into a bytearray for an 8x memory improvement, which is a
    natural follow-up optimization but not the point being tested here)."""

    def __init__(self, expected_items, target_fp_rate=0.01):
        self.m, self.k = optimal_parameters(expected_items, target_fp_rate)
        self.bits = [0] * self.m
        self.items_added = 0

    def _hash_pair(self, item):
        """Two independent-enough base hashes, derived from sha256 (stable across
        runs and interpreter restarts, unlike Python's built-in hash() for strings,
        which is salted differently per-process unless PYTHONHASHSEED is fixed)."""
        key = str(item).encode("utf-8")
        h1 = int(hashlib.sha256(key + b"|1").hexdigest(), 16)
        h2 = int(hashlib.sha256(key + b"|2").hexdigest(), 16)
        return h1, h2

    def _bit_indices(self, item):
        """The k bit positions this item maps to, via double hashing:
        h_i(item) = (h1 + i * h2) mod m for i = 0..k-1."""
        h1, h2 = self._hash_pair(item)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item):
        """Inserts item by setting all k of its bit positions to 1."""
        for index in self._bit_indices(item):
            self.bits[index] = 1
        self.items_added += 1

    def check(self, item):
        """Returns True if ALL k of item's bit positions are set (meaning "probably
        in the set" -- could be a false positive caused by other items' bits lining
        up), or False if ANY of them are unset (meaning "definitely not in the set",
        always correct, no false negatives possible)."""
        return all(self.bits[index] == 1 for index in self._bit_indices(item))

    @property
    def current_load_factor(self):
        """items_added / m -- how full the filter is relative to its bit array size,
        independent of what expected_items it was originally sized for."""
        return self.items_added / self.m if self.m else 0.0

    @property
    def estimated_false_positive_rate(self):
        """Theoretical false positive rate (1 - e^(-k*n/m))^k, using the CURRENT
        items_added as n rather than the expected_items the filter was sized for --
        this is what the rate is predicted to be RIGHT NOW, which can differ a lot
        from the target rate if more or fewer items than expected have been added."""
        if self.m == 0:
            return 1.0
        exponent = -self.k * self.items_added / self.m
        return (1 - math.exp(exponent)) ** self.k
