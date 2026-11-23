"""
bit_vector.py
----------------
Part 2.1 of the AdvancedDataStructures project: a fixed-length bit
vector, the storage layer under rank/select and the wavelet tree.

n bits are packed 64 to a word (bit i lives in word i // 64 at bit
position i % 64, least-significant first), so the payload is exactly
ceil(n / 64) * 64 bits: the information-theoretic minimum for an
arbitrary n-bit string, rounded up to a whole word. A Python list of
ints holds the words; a Python int is not a 64-bit machine word in
memory, so `size_in_bits()` reports the size the structure would have
in a compact (C-style) layout, which is what the space analysis in the
report is about. `python_bytes()` reports what this implementation
actually occupies, for honesty in the benchmarks.

Bit vectors here are "fixed-length": the length is set at construction
and individual bits can be set or cleared, but not inserted or deleted.
Once a RankSelect index is built over one, it should be treated as
read-only (the index would go stale).
"""

import sys

import numpy as np

WORD = 64
_WORD_MASK = (1 << WORD) - 1


class BitVector:
    """
        bv = BitVector.from_bits([1, 0, 1, 1, 0])
        bv[2]          # 1
        len(bv)        # 5
        bv.count()     # 3
    """

    __slots__ = ("n", "words")

    def __init__(self, n):
        if n < 0:
            raise ValueError("length must be non-negative")
        self.n = n
        self.words = [0] * ((n + WORD - 1) // WORD)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_bits(cls, bits):
        """From any iterable of 0/1 (or bools). Uses numpy packing when it can."""
        arr = np.asarray(bits if not isinstance(bits, str) else [c == "1" for c in bits])
        return cls.from_numpy(arr.astype(bool))

    @classmethod
    def from_numpy(cls, mask):
        """From a numpy bool array, packed 8 bits per byte, then 8 bytes per word."""
        mask = np.asarray(mask, dtype=bool).ravel()
        bv = cls(mask.size)
        if mask.size:
            packed = np.packbits(mask, bitorder="little")
            pad = (-packed.size) % 8
            if pad:
                packed = np.concatenate([packed, np.zeros(pad, dtype=np.uint8)])
            bv.words = packed.view("<u8").tolist()
        return bv

    @classmethod
    def from_string(cls, s):
        """'10110' -> BitVector (bit 0 is the leftmost character)."""
        return cls.from_bits([c == "1" for c in s])

    # -- access ------------------------------------------------------------

    def __len__(self):
        return self.n

    def _check(self, i):
        if not 0 <= i < self.n:
            raise IndexError(f"bit {i} outside [0, {self.n - 1}]")

    def __getitem__(self, i):
        self._check(i)
        return (self.words[i >> 6] >> (i & 63)) & 1

    def set(self, i, value=1):
        self._check(i)
        if value:
            self.words[i >> 6] |= 1 << (i & 63)
        else:
            self.words[i >> 6] &= _WORD_MASK ^ (1 << (i & 63))

    def clear(self, i):
        self.set(i, 0)

    def flip(self, i):
        self._check(i)
        self.words[i >> 6] ^= 1 << (i & 63)

    def __iter__(self):
        for i in range(self.n):
            yield (self.words[i >> 6] >> (i & 63)) & 1

    def to_list(self):
        return list(self)

    def to_numpy(self):
        if not self.n:
            return np.zeros(0, dtype=bool)
        raw = np.array(self.words, dtype="<u8").view(np.uint8)
        return np.unpackbits(raw, bitorder="little")[: self.n].astype(bool)

    def __str__(self):
        return "".join(str(b) for b in self)

    def __eq__(self, other):
        return isinstance(other, BitVector) and self.n == other.n and self.words == other.words

    def count(self):
        """Total number of 1 bits (popcount of every word)."""
        return sum(w.bit_count() for w in self.words)

    # -- size ----------------------------------------------------------------

    def size_in_bits(self):
        """Payload size in a compact layout: whole 64-bit words."""
        return len(self.words) * WORD

    def python_bytes(self):
        """What this Python implementation really occupies (list + int objects)."""
        return sys.getsizeof(self.words) + sum(sys.getsizeof(w) for w in self.words)
