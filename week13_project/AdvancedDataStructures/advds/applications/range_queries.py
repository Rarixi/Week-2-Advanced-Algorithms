"""
range_queries.py
-------------------
Part 4.1 of the AdvancedDataStructures project: a range query engine
that combines the three tree structures, each doing what it's best at.

    Fenwick tree (range-add variant)  -> range SUM, cheapest updates
    Segment tree with lazy tags       -> range MIN / MAX (sum as a cross-check)
    Persistent segment tree           -> every past version, "time travel"

Every update is applied to all three, O(log n) each, so the engine's
update cost is 3 x O(log n) and every query is O(log n). The persistent
tree records each update as a version, labelled with a timestamp, so
any query can be asked "as of" an earlier version or time, and two
moments can be diffed.

    eng = RangeQueryEngine([4, 8, 15, 16, 23, 42])
    eng.update(2, 100, t=10)          # a[2] = 100 at time 10
    eng.range_add(0, 5, 1, t=20)      # +1 everywhere at time 20
    eng.sum(0, 5)                     # 199
    eng.sum_at(0, 5, time=15)         # 193  (state as of time 15)
    eng.changes_between(time_a=5, time_b=25)
"""

import bisect

from ..trees.fenwick_tree import RangeFenwickTree
from ..trees.persistent_tree import PersistentSegmentTree
from ..trees.segment_tree import SegmentTree


class RangeQueryEngine:

    def __init__(self, values, t0=0):
        values = list(values)
        if not values:
            raise ValueError("need at least one value")
        self.n = len(values)
        self.fenwick = RangeFenwickTree(values)
        self.segment = SegmentTree(values)
        self.history = PersistentSegmentTree(values)
        self.version = 0                    # current version
        self.version_times = [t0]           # timestamp of each version
        self._clock = t0
        self._timeline = [(t0, 0)]          # (time, version) along the current line of history

    # -- updates -----------------------------------------------------------

    def _stamp(self, t):
        if t is None:
            t = self._clock + 1
        if t < self._clock:
            raise ValueError(f"time {t} is before the latest update ({self._clock})")
        self._clock = t
        return t

    def _record(self, new_version, t):
        self.version = new_version
        self.version_times.append(t)
        self._timeline.append((t, new_version))
        return new_version

    def update(self, i, value, t=None):
        """a[i] = value. Returns the new version number."""
        t = self._stamp(t)
        old = self.segment.get(i)
        self.fenwick.add(i, value - old)
        self.segment.update(i, value)
        return self._record(self.history.update(self.version, i, value, label=f"t={t}: a[{i}] = {value}"), t)

    def add(self, i, delta, t=None):
        """a[i] += delta."""
        return self.range_add(i, i, delta, t)

    def range_add(self, l, r, delta, t=None):
        """a[l..r] += delta, O(log n) in all three structures."""
        t = self._stamp(t)
        self.fenwick.range_add(l, r, delta)
        self.segment.range_add(l, r, delta)
        return self._record(self.history.range_add(self.version, l, r, delta,
                                                   label=f"t={t}: a[{l}..{r}] += {delta}"), t)

    # -- current-state queries ---------------------------------------------------

    def sum(self, l, r):
        return self.fenwick.range_sum(l, r)

    def min(self, l, r):
        return self.segment.query_min(l, r)

    def max(self, l, r):
        return self.segment.query_max(l, r)

    def mean(self, l, r):
        return self.sum(l, r) / (r - l + 1)

    def stats(self, l, r):
        s, mn, mx = self.segment.query(l, r)
        return {"sum": s, "min": mn, "max": mx, "mean": s / (r - l + 1), "count": r - l + 1}

    def get(self, i):
        return self.fenwick.get(i)

    def to_list(self):
        return self.segment.to_list()

    # -- time travel ---------------------------------------------------------------

    def version_at(self, time):
        """Latest version on the current history line with timestamp <= time."""
        times = [t for t, _ in self._timeline]
        k = bisect.bisect_right(times, time) - 1
        if k < 0:
            raise ValueError(f"no version exists at time {time}")
        return self._timeline[k][1]

    def _resolve(self, version, time):
        if version is None and time is None:
            return self.version
        if version is not None:
            return version
        return self.version_at(time)

    def sum_at(self, l, r, version=None, time=None):
        return self.history.query_sum(self._resolve(version, time), l, r)

    def min_at(self, l, r, version=None, time=None):
        return self.history.query_min(self._resolve(version, time), l, r)

    def max_at(self, l, r, version=None, time=None):
        return self.history.query_max(self._resolve(version, time), l, r)

    def get_at(self, i, version=None, time=None):
        return self.history.get(self._resolve(version, time), i)

    def snapshot(self, version=None, time=None):
        return self.history.to_list(self._resolve(version, time))

    def changes_between(self, version_a=None, version_b=None, time_a=None, time_b=None):
        """[(index, value_then, value_later), ...] between two versions or times."""
        a = self._resolve(version_a, time_a)
        b = self._resolve(version_b, time_b)
        return self.history.diff(a, b)

    def rollback(self, version=None, time=None):
        """
        Make an old version current again. Later updates branch off from
        it; the versions after it stay in the persistent tree (they are
        never destroyed) but leave the current timeline. The live Fenwick
        and segment trees are rebuilt from the snapshot, O(n).
        """
        v = self._resolve(version, time)
        values = self.history.to_list(v)
        self.fenwick = RangeFenwickTree(values)
        self.segment = SegmentTree(values)
        self.version = v
        chain = set(self.history.version_history(v))
        self._timeline = [(t, ver) for t, ver in self._timeline if ver in chain]
        return v

    def undo(self):
        """Roll back one update (to the current version's parent)."""
        parent = self.history.parents[self.version]
        if parent is None:
            raise ValueError("nothing to undo")
        return self.rollback(version=parent)

    def verify(self):
        """The three structures agree on the current state (used by the tests)."""
        live = self.segment.to_list()
        return live == self.fenwick_list() == self.history.to_list(self.version)

    def fenwick_list(self):
        return [self.fenwick.get(i) for i in range(self.n)]

    @property
    def num_versions(self):
        return self.history.num_versions
