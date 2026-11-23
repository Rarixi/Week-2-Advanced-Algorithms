"""Segment tree (lazy) and iterative segment tree vs brute force."""
import random

import pytest

from advds.trees.segment_tree import IterativeSegmentTree, SegmentTree


def brute(a, l, r):
    s = a[l:r + 1]
    return sum(s), min(s), max(s)


@pytest.mark.parametrize("seed", range(20))
def test_random_operations_match_brute_force(seed):
    r = random.Random(seed)
    n = r.randint(1, 60)
    a = [r.randint(-100, 100) for _ in range(n)]
    st = SegmentTree(a)
    for _ in range(300):
        l = r.randrange(n)
        rr = r.randrange(l, n)
        v = r.randint(-30, 30)
        op = r.random()
        if op < 0.2:
            a[l] = v
            st.update(l, v)
        elif op < 0.45:
            for i in range(l, rr + 1):
                a[i] += v
            st.range_add(l, rr, v)
        elif op < 0.6:
            for i in range(l, rr + 1):
                a[i] = v
            st.range_assign(l, rr, v)
        else:
            assert st.query(l, rr) == brute(a, l, rr)
    assert st.to_list() == a


def test_assign_then_add_and_add_then_assign():
    st = SegmentTree([1] * 8)
    st.range_assign(0, 7, 5)
    st.range_add(2, 5, 3)          # add on top of a pending assign
    assert st.to_list() == [5, 5, 8, 8, 8, 8, 5, 5]
    st.range_add(0, 7, 1)
    st.range_assign(3, 4, 0)       # assign overrides a pending add
    assert st.to_list() == [6, 6, 9, 0, 0, 9, 6, 6]
    assert st.query(0, 7) == (42, 0, 9)


def test_single_element_and_bounds():
    st = SegmentTree([7])
    assert st.query(0, 0) == (7, 7, 7)
    st.range_add(0, 0, 3)
    assert st.get(0) == 10
    with pytest.raises(IndexError):
        st.query(0, 1)
    with pytest.raises(IndexError):
        st.range_add(2, 1, 1)
    with pytest.raises(ValueError):
        SegmentTree([])


def test_query_and_update_are_logarithmic():
    n = 1 << 14
    st = SegmentTree(range(n))
    st.nodes_visited = 0
    st.query_sum(123, n - 77)
    q_nodes = st.nodes_visited
    st.nodes_visited = 0
    st.range_add(5, n - 5, 2)
    u_nodes = st.nodes_visited
    # O(log n): at most ~4 nodes per level
    assert q_nodes <= 4 * 15 + 1 and u_nodes <= 4 * 15 + 1


def test_lazy_tags_defer_work():
    st = SegmentTree(range(16))
    st.range_add(0, 15, 1)          # covered by the root alone
    assert st._add[1] == 1 and st._add[2] == 0
    assert st.query_sum(0, 0) == 1  # pushes the tag down on the way
    assert st._add[1] == 0


def test_floats():
    st = SegmentTree([0.5, 1.5, 2.25])
    st.range_add(0, 2, 0.25)
    assert st.query(0, 2) == pytest.approx((5.0, 0.75, 2.5))


def test_render_and_plot(tmp_path):
    st = SegmentTree([5, 2, 8, 1])
    st.range_add(0, 3, 1)
    text = st.render()
    assert "[0..3]=20" in text and "*" in text
    st.plot(save_path=str(tmp_path / "t.png"))
    assert (tmp_path / "t.png").exists()
    with pytest.raises(ValueError):
        SegmentTree(range(100)).render()


@pytest.mark.parametrize("op,identity,ref", [
    (lambda a, b: a + b, 0, sum), (min, float("inf"), min), (max, float("-inf"), max)])
def test_iterative_segment_tree(op, identity, ref):
    r = random.Random(3)
    a = [r.randint(-50, 50) for _ in range(37)]
    it = IterativeSegmentTree(a, op, identity)
    for _ in range(300):
        i = r.randrange(37)
        if r.random() < 0.4:
            a[i] = r.randint(-50, 50)
            it.update(i, a[i])
        else:
            j = r.randrange(i, 37)
            assert it.query(i, j) == ref(a[i:j + 1])
