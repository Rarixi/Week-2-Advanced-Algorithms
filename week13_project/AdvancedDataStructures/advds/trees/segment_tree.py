"""
segment_tree.py
------------------
Part 1.1 of the AdvancedDataStructures project: segment trees.

A segment tree stores an array of n values as a complete binary tree.
The root covers the whole array [0, n-1]; every internal node covering
[lo, hi] has two children covering [lo, mid] and [mid+1, hi]; the
leaves are the individual elements. Each node caches an aggregate of
its range (here: sum, min and max at once), so a query over any range
[l, r] only has to combine O(log n) nodes: the ones whose ranges tile
[l, r] exactly. At most two nodes per level are "partially" covered,
which is where the O(log n) bound comes from.

Two classes:

  * SegmentTree        - recursive, with LAZY PROPAGATION. Supports
                          point assignment, range add, range assign, and
                          range sum / min / max, all O(log n).
  * IterativeSegmentTree - the bottom-up, array-of-2n variant for a single
                          associative operation with point updates and
                          range queries, no lazy tags. Faster constant
                          factor; used in the benchmark to measure what
                          lazy propagation costs.

Lazy propagation: a range update that fully covers a node does not walk
down into its subtree. It updates the node's cached aggregates and leaves
a "tag" (pending add, or pending assign) on the node. The tag is pushed
one level down only when a later operation needs to look inside that
node. So a range update touches the same O(log n) nodes a range query
does, instead of every element in the range.

All ranges in this library are INCLUSIVE on both ends: [l, r].
"""

import math


class SegmentTree:
    """
    Segment tree over a list of numbers with lazy propagation.

        st = SegmentTree([5, 2, 8, 1, 9])
        st.query_sum(1, 3)     # 11
        st.range_add(0, 2, 10) # [15, 12, 18, 1, 9]
        st.query_min(0, 4)     # 1

    Storage: four arrays of size 4n (sum, min, max, tags). Build is O(n):
    every node is visited once, and a complete binary tree with n leaves
    has fewer than 2n nodes.
    """

    def __init__(self, values):
        values = list(values)
        if not values:
            raise ValueError("SegmentTree needs at least one value")
        self.n = len(values)
        size = 4 * self.n
        self._sum = [0] * size
        self._min = [0] * size
        self._max = [0] * size
        self._add = [0] * size          # pending "add v to every element"
        self._set = [None] * size       # pending "set every element to v"
        self.nodes_visited = 0          # instrumentation for tests / benchmarks
        self._build(1, 0, self.n - 1, values)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build(self, node, lo, hi, values):
        if lo == hi:
            v = values[lo]
            self._sum[node] = self._min[node] = self._max[node] = v
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self._pull(node)

    def _pull(self, node):
        a, b = 2 * node, 2 * node + 1
        self._sum[node] = self._sum[a] + self._sum[b]
        self._min[node] = min(self._min[a], self._min[b])
        self._max[node] = max(self._max[a], self._max[b])

    def _apply_set(self, node, length, v):
        self._sum[node] = v * length
        self._min[node] = self._max[node] = v
        self._set[node] = v
        self._add[node] = 0          # an assign wipes out any pending add

    def _apply_add(self, node, length, v):
        self._sum[node] += v * length
        self._min[node] += v
        self._max[node] += v
        if self._set[node] is not None:
            self._set[node] += v     # add on top of a pending assign = a different assign
        else:
            self._add[node] += v

    def _push(self, node, lo, hi):
        """Move this node's pending tag one level down to its children."""
        mid = (lo + hi) // 2
        left_len, right_len = mid - lo + 1, hi - mid
        if self._set[node] is not None:
            v = self._set[node]
            self._apply_set(2 * node, left_len, v)
            self._apply_set(2 * node + 1, right_len, v)
            self._set[node] = None
        if self._add[node]:
            v = self._add[node]
            self._apply_add(2 * node, left_len, v)
            self._apply_add(2 * node + 1, right_len, v)
            self._add[node] = 0

    def _check(self, l, r):
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")

    def _update(self, node, lo, hi, l, r, v, kind):
        self.nodes_visited += 1
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            if kind == "set":
                self._apply_set(node, hi - lo + 1, v)
            else:
                self._apply_add(node, hi - lo + 1, v)
            return
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        self._update(2 * node, lo, mid, l, r, v, kind)
        self._update(2 * node + 1, mid + 1, hi, l, r, v, kind)
        self._pull(node)

    def _query(self, node, lo, hi, l, r):
        """Returns (sum, min, max) of [l, r] ∩ [lo, hi], or None if disjoint."""
        self.nodes_visited += 1
        if r < lo or hi < l:
            return None
        if l <= lo and hi <= r:
            return self._sum[node], self._min[node], self._max[node]
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        a = self._query(2 * node, lo, mid, l, r)
        b = self._query(2 * node + 1, mid + 1, hi, l, r)
        if a is None:
            return b
        if b is None:
            return a
        return a[0] + b[0], min(a[1], b[1]), max(a[2], b[2])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __len__(self):
        return self.n

    def update(self, i, value):
        """Point assignment a[i] = value. O(log n)."""
        self._check(i, i)
        self._update(1, 0, self.n - 1, i, i, value, "set")

    def range_add(self, l, r, value):
        """a[i] += value for every i in [l, r]. O(log n) via lazy tags."""
        self._check(l, r)
        self._update(1, 0, self.n - 1, l, r, value, "add")

    def range_assign(self, l, r, value):
        """a[i] = value for every i in [l, r]. O(log n) via lazy tags."""
        self._check(l, r)
        self._update(1, 0, self.n - 1, l, r, value, "set")

    def query(self, l, r):
        """(sum, min, max) of a[l..r] in one O(log n) pass."""
        self._check(l, r)
        return self._query(1, 0, self.n - 1, l, r)

    def query_sum(self, l, r):
        return self.query(l, r)[0]

    def query_min(self, l, r):
        return self.query(l, r)[1]

    def query_max(self, l, r):
        return self.query(l, r)[2]

    def get(self, i):
        return self.query(i, i)[0]

    def to_list(self):
        """Current array contents (pushes every pending tag; O(n))."""
        out = [0] * self.n

        def walk(node, lo, hi):
            if lo == hi:
                out[lo] = self._sum[node]
                return
            self._push(node, lo, hi)
            mid = (lo + hi) // 2
            walk(2 * node, lo, mid)
            walk(2 * node + 1, mid + 1, hi)

        walk(1, 0, self.n - 1)
        return out

    def height(self):
        return math.ceil(math.log2(self.n)) + 1 if self.n > 1 else 1

    def memory_words(self):
        """Machine words the arrays hold (5 arrays x 4n)."""
        return 5 * 4 * self.n

    # ------------------------------------------------------------------
    # visualization (optional deliverable)
    # ------------------------------------------------------------------

    def render(self, field="sum", max_leaves=16):
        """
        ASCII drawing of the tree, one line per level, each node shown as
        [lo..hi]=value, with a '*' after nodes holding a pending lazy tag.
        Only meant for small arrays.
        """
        if self.n > max_leaves:
            raise ValueError(f"render is for n <= {max_leaves}")
        arr = {"sum": self._sum, "min": self._min, "max": self._max}[field]
        levels = []

        def walk(node, lo, hi, depth):
            if len(levels) <= depth:
                levels.append([])
            tag = "*" if (self._add[node] or self._set[node] is not None) and lo != hi else ""
            label = f"[{lo}]={arr[node]}" if lo == hi else f"[{lo}..{hi}]={arr[node]}{tag}"
            levels[depth].append(label)
            if lo != hi:
                mid = (lo + hi) // 2
                walk(2 * node, lo, mid, depth + 1)
                walk(2 * node + 1, mid + 1, hi, depth + 1)

        walk(1, 0, self.n - 1, 0)
        width = max(len("   ".join(lv)) for lv in levels)
        return "\n".join("   ".join(lv).center(width) for lv in levels)

    def plot(self, save_path=None, field="sum"):
        """Draw the tree with matplotlib (small n only)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = {"sum": self._sum, "min": self._min, "max": self._max}[field]
        fig, ax = plt.subplots(figsize=(max(6, self.n * 0.9), 1.3 * self.height() + 1))
        ax.axis("off")

        def walk(node, lo, hi, depth):
            x = (lo + hi + 1) / 2.0
            y = -depth
            pending = (self._add[node] or self._set[node] is not None) and lo != hi
            ax.text(x, y, f"{arr[node]}\n[{lo},{hi}]" if lo != hi else f"{arr[node]}\n[{lo}]",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#fde7d9" if pending else "#e3edfb", ec="#7a7a7a"))
            if lo != hi:
                mid = (lo + hi) // 2
                for clo, chi in ((lo, mid), (mid + 1, hi)):
                    ax.plot([x, (clo + chi + 1) / 2.0], [y - 0.25, -depth - 1 + 0.25], color="#9a9a9a", lw=0.8)
                walk(2 * node, lo, mid, depth + 1)
                walk(2 * node + 1, mid + 1, hi, depth + 1)

        walk(1, 0, self.n - 1, 0)
        ax.set_xlim(-0.2, self.n + 0.2)
        ax.set_ylim(-self.height() + 0.4, 0.6)
        ax.set_title(f"Segment tree ({field}); orange = pending lazy tag", loc="left", fontsize=10)
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        return fig


class IterativeSegmentTree:
    """
    Bottom-up segment tree for one associative operation, no lazy tags.
    Leaves live at positions n..2n-1 of a flat array, node i's children
    are 2i and 2i+1. Point update walks leaf -> root; a range query walks
    two pointers up from both ends. O(log n) each, about 3-4x faster in
    Python than the recursive lazy tree because there are no function
    calls per node.
    """

    def __init__(self, values, op=lambda a, b: a + b, identity=0):
        values = list(values)
        self.n = len(values)
        self.op = op
        self.identity = identity
        self.t = [identity] * self.n + values
        for i in range(self.n - 1, 0, -1):        # O(n) build
            self.t[i] = op(self.t[2 * i], self.t[2 * i + 1])

    def update(self, i, value):
        i += self.n
        self.t[i] = value
        i //= 2
        while i:
            self.t[i] = self.op(self.t[2 * i], self.t[2 * i + 1])
            i //= 2

    def query(self, l, r):
        """op over a[l..r] (inclusive)."""
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")
        left_acc, right_acc = self.identity, self.identity
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                left_acc = self.op(left_acc, self.t[l])
                l += 1
            if r & 1:
                r -= 1
                right_acc = self.op(self.t[r], right_acc)
            l //= 2
            r //= 2
        return self.op(left_acc, right_acc)
