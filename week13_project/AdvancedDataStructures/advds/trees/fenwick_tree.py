"""
fenwick_tree.py
------------------
Part 1.2 of the AdvancedDataStructures project: the Fenwick tree, or
Binary Indexed Tree (BIT).

A Fenwick tree stores n numbers in ONE array of n + 1 slots (index 0 is
unused) and supports point updates and prefix sums in O(log n). The
trick is that slot i (1-based) holds the sum of the block of
lowbit(i) = i & -i elements ending at i:

    i = 12 = 0b1100  -> lowbit 4 -> tree[12] = a[9] + a[10] + a[11] + a[12]

  * prefix_sum(i): add tree[i], strip the lowest set bit, repeat. Each
    step removes one 1-bit, so at most log2(n) steps.
  * add(i, delta): add delta to tree[i], ADD the lowest set bit, repeat.
    That walks exactly the slots whose blocks cover position i.
  * range_sum(l, r) = prefix_sum(r) - prefix_sum(l - 1).

Compared with a segment tree it does strictly less: no min/max (they
have no inverse, so "prefix difference" doesn't work) and no lazy range
assignment. In exchange it uses n + 1 words instead of ~4n per field, has
no recursion, and each operation is a tight loop of about log2(n)
additions. That makes it the go-to structure for streaming counts:
frequency tables, cumulative histograms, inversion counting.

Public API uses 0-based, inclusive indices like the rest of the library.
"""


class FenwickTree:
    """
        ft = FenwickTree([3, 1, 4, 1, 5])
        ft.range_sum(1, 3)   # 6
        ft.add(2, 10)        # a[2] += 10
        ft.prefix_sum(2)     # 18
    """

    def __init__(self, values_or_n):
        if isinstance(values_or_n, int):
            self.n = values_or_n
            self.tree = [0] * (self.n + 1)
            return
        values = list(values_or_n)
        self.n = len(values)
        # O(n) build: put each value in its slot, then push each slot's
        # total once into the next slot that covers it (its "parent").
        tree = [0] + values
        for i in range(1, self.n + 1):
            parent = i + (i & -i)
            if parent <= self.n:
                tree[parent] += tree[i]
        self.tree = tree

    def __len__(self):
        return self.n

    def add(self, i, delta):
        """a[i] += delta. O(log n)."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        i += 1
        tree, n = self.tree, self.n
        while i <= n:
            tree[i] += delta
            i += i & -i

    def prefix_sum(self, i):
        """a[0] + ... + a[i]; prefix_sum(-1) == 0. O(log n)."""
        if i >= self.n:
            raise IndexError(i)
        i += 1
        total = 0
        tree = self.tree
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    def range_sum(self, l, r):
        """a[l] + ... + a[r], as a difference of two prefix sums."""
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")
        return self.prefix_sum(r) - (self.prefix_sum(l - 1) if l else 0)

    def get(self, i):
        return self.range_sum(i, i)

    def set(self, i, value):
        """Point assignment, as an add of the difference."""
        self.add(i, value - self.get(i))

    def lower_bound(self, target):
        """
        Smallest index i with prefix_sum(i) >= target, for non-negative
        values (e.g. counts). Binary lifting over the implicit tree:
        O(log n), no separate binary search. Returns n if no prefix
        reaches target. This is how a Fenwick tree answers "which
        element is the k-th" in a frequency table.
        """
        pos = 0
        remaining = target
        step = 1 << self.n.bit_length()
        while step:
            nxt = pos + step
            if nxt <= self.n and self.tree[nxt] < remaining:
                pos = nxt
                remaining -= self.tree[nxt]
            step >>= 1
        return pos  # 0-based index of the answer

    def to_list(self):
        return [self.get(i) for i in range(self.n)]

    def memory_words(self):
        return self.n + 1


class RangeFenwickTree:
    """
    Range add AND range sum with two Fenwick trees (B1, B2).

    Adding v to a[l..r] changes prefix sums in a piecewise-linear way:
    prefix(i) = sum(B1, i) * i - sum(B2, i). Updating B1 at l and r+1
    with +v / -v and B2 with v*(l-1) / -v*r keeps that identity exact.
    Both operations stay O(log n) and memory is 2(n+1) words, still well
    under a lazy segment tree's.
    """

    def __init__(self, values_or_n):
        if isinstance(values_or_n, int):
            n = values_or_n
            values = None
        else:
            values = list(values_or_n)
            n = len(values)
        self.n = n
        self.b1 = FenwickTree(n)
        self.b2 = FenwickTree(n)
        if values:
            # point values are range adds of length 1; do it in O(n) by
            # building B2 from the base array (prefix(i) = -sum B2 when B1 = 0)
            self.b2 = FenwickTree([-v for v in values])

    def range_add(self, l, r, v):
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")
        # 1-based formulas mapped to 0-based storage
        self.b1.add(l, v)
        self.b2.add(l, v * l)
        if r + 1 < self.n:
            self.b1.add(r + 1, -v)
            self.b2.add(r + 1, -v * (r + 1))

    def prefix_sum(self, i):
        if i < 0:
            return 0
        return self.b1.prefix_sum(i) * (i + 1) - self.b2.prefix_sum(i)

    def range_sum(self, l, r):
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def add(self, i, v):
        self.range_add(i, i, v)

    def get(self, i):
        return self.range_sum(i, i)

    def memory_words(self):
        return 2 * (self.n + 1)
