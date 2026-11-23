"""
range_query_demo.py: segment tree, Fenwick tree, persistent tree and the
range query engine on a small "daily sales" array.

    python3 examples/range_query_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advds.applications.range_queries import RangeQueryEngine
from advds.trees.fenwick_tree import FenwickTree
from advds.trees.persistent_tree import PersistentSegmentTree
from advds.trees.segment_tree import SegmentTree

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    sales = [12, 7, 30, 4, 18, 25, 9, 14]

    print("== Segment tree with lazy propagation ==")
    st = SegmentTree(sales)
    print("sum/min/max of days 2..6:", st.query(2, 6))
    st.range_add(0, 3, 5)                 # +5 on days 0..3, one O(log n) update
    print("after +5 on days 0..3:", st.to_list())
    st.range_add(4, 7, 1)                 # leaves a pending tag on the [4..7] node
    print(st.render())                    # '*' marks a node holding a pending lazy tag
    st.plot(save_path=os.path.join(OUT, "segment_tree.png"))

    print("\n== Fenwick tree ==")
    ft = FenwickTree(sales)
    print("prefix sums:", [ft.prefix_sum(i) for i in range(len(sales))])
    ft.add(3, 10)
    print("days 2..5 after a[3] += 10:", ft.range_sum(2, 5))
    print("first day where cumulative sales reach 60:", ft.lower_bound(60))

    print("\n== Persistent segment tree ==")
    p = PersistentSegmentTree(sales)
    v1 = p.update(0, 2, 0)                # version 1: day 2 corrected to 0
    v2 = p.range_add(v1, 0, 7, 3)         # version 2: +3 everywhere
    v3 = p.update(v1, 7, 100)             # version 3 branches from version 1
    for v in range(p.num_versions):
        print(f"version {v} ({p.labels[v]:>14}, parent {p.parents[v]}):", p.to_list(v))
    print("diff v0 -> v2:", p.diff(0, v2))
    print("sharing:", p.sharing_stats())

    print("\n== Range query engine (time travel) ==")
    eng = RangeQueryEngine(sales, t0=0)
    eng.update(2, 0, t=10)
    eng.range_add(0, 7, 3, t=20)
    eng.add(5, -10, t=30)
    print("now:          ", eng.stats(0, 7))
    print("as of t=15:   ", eng.sum_at(0, 7, time=15), "(sum)", eng.max_at(0, 7, time=15), "(max)")
    print("changes t=0 -> t=30:", eng.changes_between(time_a=0, time_b=30))
    eng.undo()
    print("after undo:   ", eng.to_list(), "| structures agree:", eng.verify())
    print("figure in", OUT)


if __name__ == "__main__":
    main()
