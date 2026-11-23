"""
wavelet_tree.py
------------------
Part 2.2 of the AdvancedDataStructures project: the wavelet tree.

A wavelet tree stores a sequence S of n symbols from an alphabet of size
sigma as a binary tree of bit vectors. The root splits the alphabet in
two: its bit vector has a 0 for every position whose symbol belongs to
the left half, a 1 for the right half. The left child stores, in order,
only the symbols that went left, split again; and so on down to one
symbol per leaf. Every level together holds n bits, there are about
log2(sigma) levels, so the whole thing is n*log2(sigma) bits plus the
rank/select directories: no larger than the plain sequence packed into
log2(sigma)-bit codes, yet it answers far more than access.

With a rank/select index (rank_select.py) on each node's bit vector:

  access(i)            walk down: the bit says which child, rank maps
                       the position into that child.            O(log sigma)
  rank(c, i)           occurrences of c in S[0, i): walk down c's path,
                       mapping i through rank at each level.     O(log sigma)
  select(c, k)         position of the k-th c: walk down to c's leaf,
                       then back up with select.                O(log sigma)
  range_count(l, r, c) occurrences of c in S[l..r] = two ranks.  O(log sigma)
  quantile(l, r, k)    k-th smallest value in S[l..r]: at each node,
                       count how many of the range went left; if more
                       than k, go left, else go right.          O(log sigma)
  range_count_values(l, r, a, b)
                       how many values in S[l..r] lie in [a, b]. O(log sigma)

Shapes. The default "balanced" shape splits the alphabet by value, so
left = smaller symbols, which is what quantile and value-range counting
need. The "huffman" shape builds the tree from a Huffman code over the
symbol frequencies instead: frequent symbols get short paths, the total
bits drop from n*ceil(log2 sigma) to about n*H0 (the zero-order
entropy), and the tree becomes a *compressed* representation of the
text. Access, rank, select and range_count still work; quantile and
value ranges don't, because a Huffman code is not ordered by value.

Symbols are integers 0..sigma-1 internally; `from_text` maps characters
to codes in sorted order and keeps the mapping.
"""

import heapq
import math
from collections import Counter

import numpy as np

from .bit_vector import BitVector
from .rank_select import RankSelect


class _Node:
    __slots__ = ("rs", "left", "right", "symbol", "lo", "hi", "n")

    def __init__(self):
        self.rs = None          # RankSelect over this node's bit vector
        self.left = None
        self.right = None
        self.symbol = None      # set on leaves
        self.lo = None          # smallest / largest symbol under this node
        self.hi = None
        self.n = 0              # sequence length at this node


def _balanced_codes(sigma):
    """Order-preserving prefix code: split [lo, hi] at the midpoint, recursively."""
    codes = [""] * sigma

    def split(lo, hi, prefix):
        if lo == hi:
            codes[lo] = prefix
            return
        mid = (lo + hi) // 2
        split(lo, mid, prefix + "0")
        split(mid + 1, hi, prefix + "1")

    if sigma == 1:
        return ["0"]
    split(0, sigma - 1, "")
    return codes


def _huffman_codes(freq):
    """Huffman code over symbol frequencies (symbols with frequency 0 get none)."""
    present = [s for s, f in enumerate(freq) if f > 0]
    codes = [None] * len(freq)
    if len(present) == 1:
        codes[present[0]] = "0"
        return codes
    heap = [(freq[s], s, (s,)) for s in present]   # (weight, tiebreak, symbols)
    heapq.heapify(heap)
    parts = {}
    counter = len(freq)
    while len(heap) > 1:
        fa, _, a = heapq.heappop(heap)
        fb, _, b = heapq.heappop(heap)
        for s in a:
            parts[s] = "0" + parts.get(s, "")
        for s in b:
            parts[s] = "1" + parts.get(s, "")
        heapq.heappush(heap, (fa + fb, counter, a + b))
        counter += 1
    for s in present:
        codes[s] = parts[s]
    return codes


class WaveletTree:
    """
        wt = WaveletTree([3, 1, 4, 1, 5, 2, 6, 5], sigma=7)
        wt.access(2)               # 4
        wt.rank(1, 4)              # 2   (two 1s in S[0..3])
        wt.select(5, 2)            # 7   (second 5 is at position 7)
        wt.range_count(0, 7, 5)    # 2
        wt.quantile(2, 6, 0)       # 1   (smallest of 4,1,5,2,6)
        wt.range_count_values(0, 7, 2, 4)   # 3  (values 3, 4, 2)
    """

    def __init__(self, sequence, sigma=None, shape="balanced"):
        seq = np.asarray(sequence, dtype=np.int64).ravel()
        if seq.size and seq.min() < 0:
            raise ValueError("symbols must be non-negative integers")
        self.n = int(seq.size)
        self.sigma = int(sigma if sigma is not None else (seq.max() + 1 if seq.size else 1))
        if seq.size and seq.max() >= self.sigma:
            raise ValueError("symbol outside alphabet")
        self.shape = shape
        freq = np.bincount(seq, minlength=self.sigma) if seq.size else np.zeros(self.sigma, dtype=np.int64)
        self._freq = freq.tolist()
        if shape == "balanced":
            self.codes = _balanced_codes(self.sigma)
            self.ordered = True
        elif shape == "huffman":
            self.codes = _huffman_codes(self._freq)
            self.ordered = False
        else:
            raise ValueError("shape must be 'balanced' or 'huffman'")
        self.alphabet_map = None       # char -> code, set by from_text
        self.root = self._build(seq, list(range(self.sigma)) if shape == "balanced"
                                else [s for s in range(self.sigma) if self.codes[s] is not None], 0)

    @classmethod
    def from_text(cls, text, shape="balanced"):
        """Wavelet tree over the characters of a string, alphabet = sorted distinct chars."""
        alphabet = sorted(set(text)) or ["\0"]
        mapping = {c: i for i, c in enumerate(alphabet)}
        seq = np.fromiter((mapping[c] for c in text), dtype=np.int64, count=len(text))
        wt = cls(seq, sigma=len(alphabet), shape=shape)
        wt.alphabet_map = mapping
        wt.alphabet = alphabet
        return wt

    # -- construction --------------------------------------------------------

    def _build(self, seq, symbols, depth):
        node = _Node()
        node.n = int(seq.size)
        node.lo, node.hi = min(symbols), max(symbols)
        if len(symbols) == 1:
            node.symbol = symbols[0]
            return node
        go_right = np.zeros(self.sigma, dtype=bool)
        left_syms, right_syms = [], []
        for s in symbols:
            if self.codes[s][depth] == "1":
                go_right[s] = True
                right_syms.append(s)
            else:
                left_syms.append(s)
        bits = go_right[seq] if seq.size else np.zeros(0, dtype=bool)
        node.rs = RankSelect(BitVector.from_numpy(bits))
        node.left = self._build(seq[~bits], left_syms, depth + 1)
        node.right = self._build(seq[bits], right_syms, depth + 1)
        return node

    # -- helpers ---------------------------------------------------------------

    def _code(self, c):
        if self.alphabet_map is not None and isinstance(c, str):
            c = self.alphabet_map.get(c, -1)
        if not 0 <= c < self.sigma or self.codes[c] is None:
            return None
        return self.codes[c]

    def _symbol_out(self, s):
        return self.alphabet[s] if self.alphabet_map is not None else s

    def __len__(self):
        return self.n

    # -- core queries ------------------------------------------------------------

    def access(self, i):
        """S[i]."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        node = self.root
        while node.symbol is None:
            if node.rs.access(i):
                i = node.rs.rank1(i)
                node = node.right
            else:
                i = node.rs.rank0(i)
                node = node.left
        return self._symbol_out(node.symbol)

    def rank(self, c, i):
        """Occurrences of symbol c in S[0, i)."""
        if not 0 <= i <= self.n:
            raise IndexError(i)
        code = self._code(c)
        if code is None:
            return 0
        node = self.root
        depth = 0
        while node.symbol is None:
            if code[depth] == "1":
                i = node.rs.rank1(i)
                node = node.right
            else:
                i = node.rs.rank0(i)
                node = node.left
            depth += 1
        return i

    def select(self, c, k):
        """Position of the k-th occurrence (k >= 1) of c."""
        code = self._code(c)
        if code is None or not 1 <= k <= self.count(c):
            raise IndexError(f"no occurrence {k} of {c!r}")
        path = []
        node = self.root
        depth = 0
        while node.symbol is None:
            bit = code[depth] == "1"
            path.append((node, bit))
            node = node.right if bit else node.left
            depth += 1
        pos = k - 1                       # 0-based position inside the leaf's sequence
        for parent, bit in reversed(path):
            pos = parent.rs.select1(pos + 1) if bit else parent.rs.select0(pos + 1)
        return pos

    def count(self, c):
        """Total occurrences of c (symbol frequency)."""
        return self.rank(c, self.n)

    def frequencies(self):
        """{symbol: count} for every symbol that occurs."""
        return {self._symbol_out(s): f for s, f in enumerate(self._freq) if f}

    def range_count(self, l, r, c):
        """Occurrences of c in S[l..r] (inclusive)."""
        self._check_range(l, r)
        return self.rank(c, r + 1) - self.rank(c, l)

    def _check_range(self, l, r):
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")

    def _need_order(self, what):
        if not self.ordered:
            raise ValueError(f"{what} needs the value-ordered 'balanced' shape")

    def quantile(self, l, r, k):
        """k-th smallest value (k = 0 is the minimum) in S[l..r]."""
        self._need_order("quantile")
        self._check_range(l, r)
        if not 0 <= k <= r - l:
            raise IndexError(f"k={k} outside [0, {r - l}]")
        node = self.root
        lo, hi = l, r + 1               # half-open range inside the current node
        while node.symbol is None:
            z_lo, z_hi = node.rs.rank0(lo), node.rs.rank0(hi)
            zeros = z_hi - z_lo
            if k < zeros:
                lo, hi = z_lo, z_hi
                node = node.left
            else:
                k -= zeros
                lo, hi = lo - z_lo, hi - z_hi     # rank1 = position - rank0
                node = node.right
        return self._symbol_out(node.symbol)

    def range_min(self, l, r):
        return self.quantile(l, r, 0)

    def range_max(self, l, r):
        return self.quantile(l, r, r - l)

    def range_median(self, l, r):
        return self.quantile(l, r, (r - l) // 2)

    def range_count_values(self, l, r, a, b):
        """How many of S[l..r] have value in [a, b]."""
        self._need_order("range_count_values")
        self._check_range(l, r)
        if self.alphabet_map is not None:
            a, b = self.alphabet_map.get(a, a), self.alphabet_map.get(b, b)

        def walk(node, lo, hi):
            if lo >= hi or node.hi < a or node.lo > b:
                return 0
            if a <= node.lo and node.hi <= b:
                return hi - lo
            z_lo, z_hi = node.rs.rank0(lo), node.rs.rank0(hi)
            return walk(node.left, z_lo, z_hi) + walk(node.right, lo - z_lo, hi - z_hi)

        return walk(self.root, l, r + 1)

    def range_frequencies(self, l, r):
        """{symbol: count} over S[l..r], visiting only subtrees that occur in the range."""
        self._check_range(l, r)
        out = {}

        def walk(node, lo, hi):
            if lo >= hi:
                return
            if node.symbol is not None:
                out[self._symbol_out(node.symbol)] = hi - lo
                return
            z_lo, z_hi = node.rs.rank0(lo), node.rs.rank0(hi)
            walk(node.left, z_lo, z_hi)
            walk(node.right, lo - z_lo, hi - z_hi)

        walk(self.root, l, r + 1)
        return out

    def to_list(self):
        return [self.access(i) for i in range(self.n)]

    def to_text(self):
        return "".join(self.to_list())

    # -- space accounting ------------------------------------------------------

    def _nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            yield node
            stack.append(node.left)
            stack.append(node.right)

    def payload_bits(self):
        """Bits in all node bit vectors (exactly sum of code lengths over S)."""
        return sum(node.rs.bv.n for node in self._nodes() if node.rs)

    def size_in_bits(self, include_index=True):
        """Compact-layout size: word-padded bit vectors, plus rank/select directories."""
        total = 0
        for node in self._nodes():
            if node.rs:
                total += node.rs.size_in_bits() if include_index else node.rs.bv.size_in_bits()
        return total

    def entropy_h0(self):
        """Zero-order empirical entropy of S, bits per symbol."""
        if not self.n:
            return 0.0
        h = 0.0
        for f in self._freq:
            if f:
                p = f / self.n
                h -= p * math.log2(p)
        return h

    def compression_stats(self, bits_per_symbol_original=8):
        """
        Size of the original (8-bit chars by default) vs the wavelet tree's
        payload and full size, and the entropy bound.
        """
        original = self.n * bits_per_symbol_original
        payload = self.payload_bits()
        full = self.size_in_bits()
        return {
            "n": self.n,
            "sigma": self.sigma,
            "shape": self.shape,
            "original_bits": original,
            "payload_bits": payload,
            "payload_bits_per_symbol": payload / max(self.n, 1),
            "total_bits_with_index": full,
            "total_bits_per_symbol": full / max(self.n, 1),
            "h0_bits_per_symbol": self.entropy_h0(),
            "fixed_width_bits_per_symbol": max(1, math.ceil(math.log2(max(self.sigma, 2)))),
            "compression_ratio_payload": original / max(payload, 1),
            "compression_ratio_total": original / max(full, 1),
        }


def symbol_counts(seq):
    """Plain Counter, used as a reference in tests."""
    return Counter(seq)
