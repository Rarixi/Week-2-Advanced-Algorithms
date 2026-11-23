"""
rank_select.py
-----------------
Part 2.1 of the AdvancedDataStructures project: rank and select on a
bit vector, the two primitives every succinct structure is built from.

    rank1(i)   = number of 1 bits in positions [0, i)      (rank1(n) = total)
    select1(k) = position of the k-th 1 bit, k = 1, 2, ...  (rank1(select1(k)) = k - 1)

and rank0 / select0 for zeros. "Up to i" in the assignment is read as
the half-open prefix [0, i), the standard convention (Jacobson; sdsl);
the inclusive count of [0, i] is rank1(i + 1).

Constant-time RANK uses three layers, Jacobson's two-level directory
plus a lookup table:

    superblocks (512 bits = 8 words): absolute count of 1s before each
                superblock. n/512 entries of 64 bits.
    blocks      (64 bits = 1 word):   count of 1s from the start of the
                enclosing superblock to the start of this block. At most
                448, so 9 bits would do; stored as 16-bit in the space
                accounting.
    lookup table: popcount of every 16-bit value (65,536 entries). The
                bits of the final, partial word are masked off and
                counted with 4 table lookups.

    rank1(i) = SB[i // 512] + B[i // 64] + popcount(word & low (i % 64) bits)

Three array reads and four table reads, regardless of n: O(1).

SELECT uses the same directories plus samples: the superblock holding
every 512th one (and zero) is recorded, so a select starts from a known
narrow range of superblocks, binary-searches the superblock counts
inside it, scans at most 8 block counts, then finds the byte by byte
popcounts and the bit by a 256 x 8 select-in-byte table. That is O(1)
whenever the 1s are spread out, and O(log(n/512)) in the worst case
(a long run of zeros between two samples). Truly worst-case O(1) select
(Clark / Munro) needs a further level of directories for sparse
regions, which isn't built here; the benchmark measures the practical
behaviour.

Space overhead over the n-bit payload: 64/512 + 16/64 = 0.375 n bits
for rank, plus 64 bits per 512 ones and per 512 zeros of select samples
(0.125 n bits whatever the density), 0.5 n bits in total. That is the
price of plain directories in Python-friendly sizes; packing block counts
into 9 bits and sampling more sparsely brings it to ~0.2 n.
`overhead_bits()` reports the exact figure.
"""

from .bit_vector import BitVector, WORD

SUPER = 512                      # bits per superblock
WORDS_PER_SUPER = SUPER // WORD  # 8
SAMPLE = 512                     # select sample rate (every 512th one / zero)

# lookup tables, built once at import
POP16 = bytes(bin(x).count("1") for x in range(1 << 16))
POP8 = bytes(bin(x).count("1") for x in range(256))
# SELECT8[b * 8 + r] = position (0..7) of the (r+1)-th set bit of byte b, 8 if none
SELECT8 = bytearray(256 * 8)
for _b in range(256):
    _r = 0
    for _pos in range(8):
        if (_b >> _pos) & 1:
            SELECT8[_b * 8 + _r] = _pos
            _r += 1
    for _rest in range(_r, 8):
        SELECT8[_b * 8 + _rest] = 8


def popcount64(w):
    """Popcount of a 64-bit word with four 16-bit table lookups."""
    return POP16[w & 0xFFFF] + POP16[(w >> 16) & 0xFFFF] + POP16[(w >> 32) & 0xFFFF] + POP16[w >> 48]


def select_in_word(w, r):
    """Position (0..63) of the (r+1)-th set bit of 64-bit word w (r is 0-based)."""
    for byte_index in range(8):
        byte = (w >> (8 * byte_index)) & 0xFF
        c = POP8[byte]
        if r < c:
            return 8 * byte_index + SELECT8[byte * 8 + r]
        r -= c
    raise ValueError("word has fewer set bits than requested")


class RankSelect:
    """
        rs = RankSelect(BitVector.from_string("0110100111"))
        rs.rank1(5)     # 3   (bits 0..4 = 0,1,1,0,1)
        rs.select1(3)   # 4   (the 3rd one is at position 4)
        rs.rank0(5)     # 2
        rs.select0(2)   # 3
    """

    def __init__(self, bv):
        if not isinstance(bv, BitVector):
            bv = BitVector.from_bits(bv)
        self.bv = bv
        self.n = bv.n
        words = bv.words
        self._words = words
        nw = len(words)

        # rank directories
        sb = []
        blk = [0] * nw
        total = 0
        for w_index in range(nw):
            if w_index % WORDS_PER_SUPER == 0:
                sb.append(total)
                in_super = 0
            blk[w_index] = in_super
            c = popcount64(words[w_index])
            in_super += c
            total += c
        sb.append(total)              # sentinel: SB[last + 1] = total ones
        self._sb = sb
        self._blk = blk
        self.ones = total
        self.zeros = self.n - total

        # select samples: superblock that contains the (j*SAMPLE + 1)-th one / zero
        self._sample1 = self._make_samples(one=True)
        self._sample0 = self._make_samples(one=False)

    # -- helpers ---------------------------------------------------------------

    def _ones_before_super(self, j):
        return self._sb[j]

    def _zeros_before_super(self, j):
        return min(j * SUPER, self.n) - self._sb[j]

    def _make_samples(self, one):
        count_before = self._ones_before_super if one else self._zeros_before_super
        total = self.ones if one else self.zeros
        samples = []
        j = 0
        nsb = len(self._sb) - 1
        for k in range(1, total + 1, SAMPLE):
            # first superblock j with count_before(j + 1) >= k
            while j < nsb and count_before(j + 1) < k:
                j += 1
            samples.append(j)
        samples.append(max(nsb - 1, 0))
        return samples

    # -- rank ------------------------------------------------------------------

    def rank1(self, i):
        """Number of 1s in positions [0, i). O(1)."""
        if not 0 <= i <= self.n:
            raise IndexError(f"rank position {i} outside [0, {self.n}]")
        w = i >> 6
        if w >= len(self._words):
            return self.ones          # i == n on a word boundary
        r = self._sb[i >> 9]
        r += self._blk[w]
        rem = i & 63
        if rem:
            word = self._words[w] & ((1 << rem) - 1)
            r += POP16[word & 0xFFFF] + POP16[(word >> 16) & 0xFFFF] \
                + POP16[(word >> 32) & 0xFFFF] + POP16[word >> 48]
        return r

    def rank0(self, i):
        """Number of 0s in positions [0, i). O(1)."""
        return i - self.rank1(i)

    def rank(self, bit, i):
        return self.rank1(i) if bit else self.rank0(i)

    def access(self, i):
        return self.bv[i]

    # -- select ----------------------------------------------------------------

    def _select(self, k, one):
        total = self.ones if one else self.zeros
        if not 1 <= k <= total:
            raise IndexError(f"select{int(one)}({k}): only {total} {'ones' if one else 'zeros'}")
        samples = self._sample1 if one else self._sample0
        count_before = self._ones_before_super if one else self._zeros_before_super
        s = (k - 1) // SAMPLE
        lo, hi = samples[s], samples[s + 1]
        # binary search: last superblock j in [lo, hi] with count_before(j) < k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_before(mid) < k:
                lo = mid
            else:
                hi = mid - 1
        j = lo
        remaining = k - count_before(j)                 # 1-based within superblock
        # scan the (at most 8) blocks of superblock j
        first = j * WORDS_PER_SUPER
        last = min(first + WORDS_PER_SUPER, len(self._words)) - 1
        w = first
        while w < last:
            nxt = self._blk[w + 1] if one else (w + 1 - first) * WORD - self._blk[w + 1]
            if nxt >= remaining:
                break
            w += 1
        before = self._blk[w] if one else (w - first) * WORD - self._blk[w]
        word = self._words[w]
        if not one:
            word = ~word & ((1 << WORD) - 1)
        return w * WORD + select_in_word(word, remaining - before - 1)

    def select1(self, k):
        """Position of the k-th 1 (k >= 1)."""
        return self._select(k, True)

    def select0(self, k):
        """Position of the k-th 0 (k >= 1)."""
        return self._select(k, False)

    def select(self, bit, k):
        return self._select(k, bool(bit))

    # -- size ------------------------------------------------------------------

    def overhead_bits(self):
        """Index size in a compact layout: 64-bit superblock counts, 16-bit
        block counts, 64-bit select samples. (The two lookup tables are
        shared by every instance and not counted.)"""
        return 64 * len(self._sb) + 16 * len(self._blk) + 64 * (len(self._sample1) + len(self._sample0))

    def size_in_bits(self):
        return self.bv.size_in_bits() + self.overhead_bits()

    def overhead_ratio(self):
        payload = max(self.bv.size_in_bits(), 1)
        return self.overhead_bits() / payload


class NaiveRankSelect:
    """
    Reference implementation for the benchmark and tests: rank by
    popcounting every word up to i (O(n / 64)), select by scanning words
    (O(n / 64)). No extra space at all.
    """

    def __init__(self, bv):
        if not isinstance(bv, BitVector):
            bv = BitVector.from_bits(bv)
        self.bv = bv
        self.n = bv.n
        self.ones = bv.count()
        self.zeros = self.n - self.ones

    def rank1(self, i):
        words = self.bv.words
        full = i >> 6
        r = sum(w.bit_count() for w in words[:full])
        rem = i & 63
        if rem:
            r += (words[full] & ((1 << rem) - 1)).bit_count()
        return r

    def rank0(self, i):
        return i - self.rank1(i)

    def select1(self, k):
        if not 1 <= k <= self.ones:
            raise IndexError(k)
        for w_index, w in enumerate(self.bv.words):
            c = w.bit_count()
            if k <= c:
                return w_index * WORD + select_in_word(w, k - 1)
            k -= c
        raise IndexError(k)

    def select0(self, k):
        if not 1 <= k <= self.zeros:
            raise IndexError(k)
        for w_index, w in enumerate(self.bv.words):
            valid = min(WORD, self.n - w_index * WORD)
            inv = ~w & ((1 << valid) - 1)
            c = inv.bit_count()
            if k <= c:
                return w_index * WORD + select_in_word(inv, k - 1)
            k -= c
        raise IndexError(k)
