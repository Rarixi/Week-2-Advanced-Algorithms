"""
dna_index.py
---------------
Part 4.3 of the AdvancedDataStructures project: indexing DNA with a
wavelet tree and bit vectors.

Two indexes over one sequence:

1. A wavelet tree over the sequence itself (alphabet A < C < G < T, plus
   N for unknown bases if the input has any). It answers, in
   O(log sigma) = O(1) for DNA:
     * count(base, l, r)          how many A's (etc.) in [l, r]
     * composition(l, r), gc_content(l, r)
     * kth_smallest(l, r, k)      k-th smallest base in the window, in the
                                  order A < C < G < T (so the median base,
                                  or "is there any G or T in this window")
   in 2 bits per base (vs 8 for a text file) plus rank directories.

2. An FM-index for pattern queries: the Burrows-Wheeler transform of the
   sequence, stored in a second wavelet tree. Backward search counts the
   occurrences of a pattern P with 2|P| rank queries, independent of the
   genome's length:

       for c in reversed(P):  [lo, hi) = C[c] + rank(c, lo), C[c] + rank(c, hi)

   and the number of occurrences is hi - lo. Positions are recovered by
   walking the LF mapping to the nearest sampled suffix-array entry. The
   sampled rows are marked in a BitVector with a RankSelect index, so
   finding a sample's slot is one rank1 call: the bit vector doing
   exactly the job succinct structures exist for.

The suffix array is built by prefix doubling (Manber & Myers) with numpy
sorting, O(n log^2 n), fine for genome fragments up to a few million
bases. FASTA input is supported (`read_fasta`); `random_genome` makes a
synthetic sequence with known planted repeats for the demos and tests.
"""

import numpy as np

from ..succinct.bit_vector import BitVector
from ..succinct.rank_select import RankSelect
from ..succinct.wavelet_tree import WaveletTree

BASES = "ACGT"


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def parse_fasta(text):
    """{record name: sequence} from FASTA text (multi-record, any line width)."""
    records = {}
    name = None
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].split()[0] if len(line) > 1 else f"record{len(records) + 1}"
            chunks = []
        else:
            chunks.append(line.upper())
    if name is not None:
        records[name] = "".join(chunks)
    elif chunks:
        records["sequence"] = "".join(chunks)
    return records


def read_fasta(path):
    with open(path) as fh:
        return parse_fasta(fh.read())


def clean_sequence(seq):
    """Uppercase; any IUPAC ambiguity code (R, Y, K, ...) becomes N."""
    seq = seq.upper()
    return "".join(c if c in "ACGTN" else "N" for c in seq if not c.isspace())


def random_genome(n, gc=0.41, seed=0, repeats=(("GATTACA", 25), ("TTAGGG", 40))):
    """
    Synthetic genome of n bases with GC content `gc` (human ~0.41), plus
    planted copies of a few motifs at random positions so pattern counts
    are known: returns (sequence, {motif: [positions]}).
    """
    rng = np.random.default_rng(seed)
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    arr = rng.choice(np.array(list(BASES)), size=n, p=p)
    planted = {}
    taken = np.zeros(n, dtype=bool)          # planted copies never overlap each other
    for motif, copies in repeats:
        m = len(motif)
        pos = []
        while len(pos) < copies:
            q = int(rng.integers(0, n - m))
            if not taken[q:q + m].any():
                taken[q:q + m] = True
                pos.append(q)
                arr[q:q + m] = list(motif)
        planted[motif] = sorted(pos)
    return "".join(arr.tolist()), planted


def write_fasta(path, records, width=60):
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


# ---------------------------------------------------------------------------
# suffix array
# ---------------------------------------------------------------------------

def suffix_array(codes):
    """
    Suffix array of an integer sequence that ends with a unique smallest
    sentinel, by prefix doubling: sort suffixes by their first 1, 2, 4, ...
    symbols, using the previous round's ranks as keys.
    """
    s = np.asarray(codes, dtype=np.int64)
    n = s.size
    rank = s.copy()
    k = 1
    sa = np.argsort(rank, kind="stable")
    while True:
        second = np.full(n, -1, dtype=np.int64)
        second[:n - k] = rank[k:] if k < n else second[:0]
        sa = np.lexsort((second, rank))
        r1, r2 = rank[sa], second[sa]
        new_group = np.empty(n, dtype=bool)
        new_group[0] = True
        new_group[1:] = (r1[1:] != r1[:-1]) | (r2[1:] != r2[:-1])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.cumsum(new_group) - 1
        rank = new_rank
        if rank.max() == n - 1:
            return sa
        k *= 2


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------

class DNAIndex:
    """
        idx = DNAIndex("ACGTACGTTAGGG...")
        idx.count("A", 0, 99)          # A's in the first 100 bases
        idx.gc_content(0, 999)
        idx.kth_smallest(10, 20, 0)    # smallest base in [10, 20]
        idx.pattern_count("GATTACA")
        idx.find("GATTACA")            # sorted positions
    """

    def __init__(self, sequence, sa_sample=32):
        seq = clean_sequence(sequence)
        if not seq:
            raise ValueError("empty sequence")
        self.sequence_length = len(seq)
        self.alphabet = sorted(set(seq) | set(BASES))       # A C G (N) T
        self.code = {c: i + 1 for i, c in enumerate(self.alphabet)}   # 0 is the sentinel '$'
        codes = np.fromiter((self.code[c] for c in seq), dtype=np.int64, count=len(seq))
        self.sigma = len(self.alphabet) + 1

        # 1) wavelet tree over the sequence (codes shifted down to 0..)
        self.wt = WaveletTree(codes - 1, sigma=len(self.alphabet))

        # 2) FM-index: suffix array, BWT, C array, sampled SA
        text = np.concatenate([codes, [0]])
        sa = suffix_array(text)
        bwt = text[(sa - 1) % text.size]
        self.n = text.size                            # includes the sentinel
        self.bwt_wt = WaveletTree(bwt, sigma=self.sigma)
        counts = np.bincount(text, minlength=self.sigma)
        self.C = np.concatenate([[0], np.cumsum(counts)[:-1]]).tolist()   # C[c] = # symbols < c
        self.sa_sample = sa_sample
        marks = (sa % sa_sample) == 0
        self.sampled = RankSelect(BitVector.from_numpy(marks))
        self.samples = sa[marks].tolist()             # SA values of marked rows, in row order
        self._sentinel_row = int(np.flatnonzero(bwt == 0)[0])

    # -- wavelet-tree range queries -------------------------------------------------

    def _check(self, l, r):
        if not (0 <= l <= r < self.sequence_length):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.sequence_length - 1}]")

    def _sym(self, base):
        base = base.upper()
        if base not in self.code:
            raise ValueError(f"unknown base {base!r}")
        return self.code[base] - 1

    def base_at(self, i):
        return self.alphabet[self.wt.access(i)]

    def count(self, base, l, r):
        """How many times `base` occurs in positions [l, r]."""
        self._check(l, r)
        return self.wt.range_count(l, r, self._sym(base))

    def composition(self, l, r):
        """{base: count} over [l, r]."""
        self._check(l, r)
        return {self.alphabet[s]: c for s, c in sorted(self.wt.range_frequencies(l, r).items())}

    def gc_content(self, l, r):
        """Fraction of G + C among the known bases (N excluded) in [l, r]."""
        comp = self.composition(l, r)
        known = sum(v for b, v in comp.items() if b != "N")
        return (comp.get("G", 0) + comp.get("C", 0)) / known if known else 0.0

    def kth_smallest(self, l, r, k):
        """k-th smallest base (k = 0 is the smallest) in [l, r], order A < C < G < (N <) T."""
        self._check(l, r)
        return self.alphabet[self.wt.quantile(l, r, k)]

    def median_base(self, l, r):
        return self.kth_smallest(l, r, (r - l) // 2)

    def window_profile(self, base, window, step=None):
        """Count of `base` in each window of `window` bases (sliding by `step`)."""
        step = step or window
        out = []
        for start in range(0, self.sequence_length - window + 1, step):
            out.append((start, self.count(base, start, start + window - 1)))
        return out

    # -- FM-index pattern queries ---------------------------------------------------------

    def _backward_search(self, pattern):
        lo, hi = 0, self.n
        for ch in reversed(pattern.upper()):
            c = self.code.get(ch)
            if c is None:
                return 0, 0
            lo = self.C[c] + self.bwt_wt.rank(c, lo)
            hi = self.C[c] + self.bwt_wt.rank(c, hi)
            if lo >= hi:
                return 0, 0
        return lo, hi

    def pattern_count(self, pattern):
        """Occurrences of `pattern` (overlapping ones included). O(|pattern|) ranks."""
        if not pattern:
            raise ValueError("empty pattern")
        lo, hi = self._backward_search(pattern)
        return hi - lo

    def _locate_row(self, row):
        """SA[row] by LF-walking back to a sampled row (at most sa_sample - 1 steps)."""
        steps = 0
        while not self.sampled.access(row):
            if row == self._sentinel_row:          # BWT char is '$': this suffix is the whole text
                return steps
            c = self.bwt_wt.access(row)
            row = self.C[c] + self.bwt_wt.rank(c, row)
            steps += 1
        return self.samples[self.sampled.rank1(row)] + steps

    def find(self, pattern, limit=None):
        """Sorted start positions of `pattern`."""
        lo, hi = self._backward_search(pattern)
        rows = range(lo, hi if limit is None else min(hi, lo + limit))
        return sorted(self._locate_row(r) for r in rows)

    def pattern_frequencies(self, patterns):
        return {p: self.pattern_count(p) for p in patterns}

    # -- space ------------------------------------------------------------------------------

    def size_report(self):
        """Bits used by each component (compact layout) vs plain 8-bit text."""
        n = self.sequence_length
        wt_bits = self.wt.size_in_bits()
        fm_bits = self.bwt_wt.size_in_bits() + self.sampled.size_in_bits() + 64 * len(self.samples)
        return {
            "bases": n,
            "plain_text_bits": 8 * n,
            "wavelet_tree_bits": wt_bits,
            "wavelet_tree_bits_per_base": wt_bits / n,
            "wavelet_payload_bits_per_base": self.wt.payload_bits() / n,
            "fm_index_bits": fm_bits,
            "fm_index_bits_per_base": fm_bits / n,
            "naive_suffix_array_bits": 64 * (n + 1),
        }


def naive_count(seq, pattern):
    """Overlapping occurrence count by scanning: the reference for the tests."""
    count = 0
    start = seq.find(pattern)
    while start != -1:
        count += 1
        start = seq.find(pattern, start + 1)
    return count
