"""
bitmask_utils.py
-----------------
Small helpers for treating a plain Python int as a set of city/item indices -- a
"bitmask" -- used by src/dp_advanced/bitmask_traveling_salesman.py's Held-Karp DP.

Bit i of the mask is set exactly when item/city i is "in" the represented set. This
is the standard trick behind any "subset DP": instead of using an actual set object
as a DP key (slow to hash, awkward to enumerate), a subset of up to ~20-25 elements
fits in a single machine integer, so checking membership, adding/removing an element,
or using the whole subset as a dict/array index are all O(1) bitwise operations.

Provides:
    full_mask(n)          -- the mask representing "every one of n elements is in
                              the set", i.e. n bits all set to 1
    is_bit_set(mask, i)    -- True if element i is in the set represented by mask
    set_bit(mask, i)       -- the mask with element i added to the set
    clear_bit(mask, i)     -- the mask with element i removed from the set
    popcount(mask)         -- how many elements are currently in the set (how many
                              bits are set)
"""


def full_mask(n):
    """The mask representing all n elements (0..n-1) being in the set: n ones in
    binary, e.g. full_mask(4) == 0b1111 == 15."""
    return (1 << n) - 1


def is_bit_set(mask, i):
    """True if element i is a member of the set `mask` represents."""
    return (mask >> i) & 1 == 1


def set_bit(mask, i):
    """Return a new mask with element i added to the set (no-op if already present)."""
    return mask | (1 << i)


def clear_bit(mask, i):
    """Return a new mask with element i removed from the set (no-op if already absent)."""
    return mask & ~(1 << i)


def popcount(mask):
    """How many elements are currently in the set -- how many 1 bits `mask` has.
    Used, e.g., to group bitmask DP states by "how many cities visited so far"."""
    return bin(mask).count("1")
