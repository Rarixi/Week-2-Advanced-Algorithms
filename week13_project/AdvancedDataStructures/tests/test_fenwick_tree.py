"""Fenwick trees vs brute force."""
import random

import pytest

from advds.trees.fenwick_tree import FenwickTree, RangeFenwickTree


@pytest.mark.parametrize("seed", range(10))
def test_point_update_range_sum(seed):
    r = random.Random(seed)
    n = r.randint(1, 80)
    a = [r.randint(-20, 20) for _ in range(n)]
    ft = FenwickTree(a)
    for _ in range(400):
        i = r.randrange(n)
        if r.random() < 0.4:
            d = r.randint(-9, 9)
            a[i] += d
            ft.add(i, d)
        elif r.random() < 0.2:
            a[i] = r.randint(-9, 9)
            ft.set(i, a[i])
        else:
            j = r.randrange(i, n)
            assert ft.range_sum(i, j) == sum(a[i:j + 1])
            assert ft.prefix_sum(j) == sum(a[:j + 1])
    assert ft.to_list() == a


def test_linear_build_equals_incremental_build():
    a = list(range(1, 101))
    built = FenwickTree(a)
    inc = FenwickTree(len(a))
    for i, v in enumerate(a):
        inc.add(i, v)
    assert built.tree == inc.tree


def test_space_is_n_plus_one():
    assert len(FenwickTree(1000).tree) == 1001
    assert FenwickTree(1000).memory_words() == 1001


def test_lower_bound_finds_kth_element():
    counts = [3, 0, 2, 5, 1]           # prefix sums 3 3 5 10 11
    ft = FenwickTree(counts)
    assert [ft.lower_bound(k) for k in (1, 3, 4, 6, 10, 11, 12)] == [0, 0, 2, 3, 3, 4, 5]


@pytest.mark.parametrize("seed", range(10))
def test_range_add_range_sum(seed):
    r = random.Random(100 + seed)
    n = r.randint(1, 60)
    a = [r.randint(-20, 20) for _ in range(n)]
    rf = RangeFenwickTree(a)
    for _ in range(300):
        l = r.randrange(n)
        rr = r.randrange(l, n)
        if r.random() < 0.5:
            v = r.randint(-9, 9)
            for i in range(l, rr + 1):
                a[i] += v
            rf.range_add(l, rr, v)
        else:
            assert rf.range_sum(l, rr) == sum(a[l:rr + 1])
    assert [rf.get(i) for i in range(n)] == a


def test_bounds():
    ft = FenwickTree([1, 2, 3])
    with pytest.raises(IndexError):
        ft.add(3, 1)
    with pytest.raises(IndexError):
        ft.range_sum(2, 1)
    with pytest.raises(IndexError):
        RangeFenwickTree(3).range_add(0, 3, 1)
