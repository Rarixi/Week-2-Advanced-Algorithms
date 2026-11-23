"""Persistent segment tree and persistent map: every version stays correct."""
import random

import pytest

from advds.trees.persistent_tree import PersistentMap, PersistentSegmentTree


@pytest.mark.parametrize("seed", range(15))
def test_all_versions_remain_correct(seed):
    r = random.Random(seed)
    n = r.randint(1, 40)
    base = [r.randint(-9, 9) for _ in range(n)]
    p = PersistentSegmentTree(base)
    hist = [list(base)]
    for _ in range(60):
        v = r.randrange(len(hist))              # branch from ANY earlier version
        a = list(hist[v])
        l = r.randrange(n)
        rr = r.randrange(l, n)
        x = r.randint(-5, 5)
        if r.random() < 0.5:
            a[l] = x
            nv = p.update(v, l, x)
        else:
            for i in range(l, rr + 1):
                a[i] += x
            nv = p.range_add(v, l, rr, x)
        assert nv == len(hist)
        hist.append(a)
    for v, a in enumerate(hist):
        assert p.to_list(v) == a
        l = r.randrange(n)
        rr = r.randrange(l, n)
        assert p.query(v, l, rr) == (sum(a[l:rr + 1]), min(a[l:rr + 1]), max(a[l:rr + 1]))


def test_old_version_untouched():
    p = PersistentSegmentTree([1, 2, 3, 4])
    v1 = p.update(0, 2, 10)
    v2 = p.range_add(v1, 0, 3, 1)
    assert p.to_list(0) == [1, 2, 3, 4]
    assert p.to_list(v1) == [1, 2, 10, 4]
    assert p.to_list(v2) == [2, 3, 11, 5]
    assert p.version_history(v2) == [0, v1, v2]


def test_update_creates_only_log_n_nodes():
    n = 1 << 12
    p = PersistentSegmentTree(range(n))
    before = p.node_count
    p.update(0, 1234, -1)
    assert p.node_count - before == 13          # one node per level: log2(n) + 1
    before = p.node_count
    p.range_add(1, 100, 3000, 5)
    assert p.node_count - before <= 4 * 13


def test_versions_share_nodes():
    p = PersistentSegmentTree(list(range(1024)))
    v = 0
    for k in range(100):
        v = p.update(v, (k * 7) % 1024, -1)
    st = p.sharing_stats()
    assert st["versions"] == 101 and st["saving_factor"] > 50


@pytest.mark.parametrize("seed", range(10))
def test_diff_matches_brute_force(seed):
    r = random.Random(50 + seed)
    n = r.randint(1, 50)
    p = PersistentSegmentTree([0] * n)
    hist = [[0] * n]
    for _ in range(30):
        v = r.randrange(len(hist))
        a = list(hist[v])
        l = r.randrange(n)
        rr = r.randrange(l, n)
        x = r.randint(-3, 3)
        if r.random() < 0.5:
            a[l] = x
            p.update(v, l, x)
        else:
            for i in range(l, rr + 1):
                a[i] += x
            p.range_add(v, l, rr, x)
        hist.append(a)
    for _ in range(40):
        i, j = r.randrange(len(hist)), r.randrange(len(hist))
        expect = [(k, hist[i][k], hist[j][k]) for k in range(n) if hist[i][k] != hist[j][k]]
        assert p.diff(i, j) == expect


def test_diff_skips_shared_subtrees():
    n = 1 << 14
    p = PersistentSegmentTree([0] * n)
    v1 = p.update(0, 9999, 5)
    assert p.diff(0, v1) == [(9999, 0, 5)]
    assert p.last_diff_nodes <= 2 * 15 + 1       # one path, not n nodes


def test_bad_version():
    p = PersistentSegmentTree([1])
    with pytest.raises(IndexError):
        p.query(3, 0, 0)


def test_map_basic_and_immutability():
    m0 = PersistentMap()
    m1 = m0.set("a", "1").set("b", "2")
    m2 = m1.set("b", "3").delete("a")
    assert len(m0) == 0 and len(m1) == 2 and len(m2) == 1
    assert m1.get("b") == "2" and m2.get("b") == "3" and m2.get("a") is None
    assert "a" in m1 and "a" not in m2
    assert m1.items() == [("a", "1"), ("b", "2")]
    assert m1.set("a", "1") is m1                # no-op update returns the same map
    assert m1.delete("zzz") is m1


def test_map_canonical_form_independent_of_history():
    r = random.Random(1)
    keys = [f"src/file{i}.py" for i in range(400)]
    a = PersistentMap()
    for k in keys:
        a = a.set(k, "x")
    shuffled = keys[:]
    r.shuffle(shuffled)
    b = PersistentMap()
    for k in shuffled + ["extra1", "extra2"]:
        b = b.set(k, "x")
    b = b.delete("extra1").delete("extra2")
    assert a.root_hash == b.root_hash and a == b


def test_map_random_against_dict():
    r = random.Random(7)
    ref = {}
    m = PersistentMap()
    snapshots = []
    for step in range(1500):
        k = f"k{r.randrange(300)}"
        if r.random() < 0.3 and ref:
            k = r.choice(list(ref))
            del ref[k]
            m = m.delete(k)
        else:
            v = str(r.randrange(5))
            ref[k] = v
            m = m.set(k, v)
        if step % 100 == 0:
            snapshots.append((dict(ref), m))
    for ref_snap, m_snap in snapshots:
        assert dict(m_snap.items()) == ref_snap and len(m_snap) == len(ref_snap)


def test_map_diff_and_sharing():
    m = PersistentMap()
    for i in range(1000):
        m = m.set(f"f{i}", "v0")
    m2 = m.set("f5", "v1").delete("f7").set("new", "x")
    d = m.diff(m2)
    assert (d["added"], d["removed"], d["modified"]) == (["new"], ["f7"], ["f5"])
    assert d["nodes_visited"] < 120                    # vs ~1070 nodes in the trie
    shared = len(m.node_ids() & m2.node_ids())
    assert shared > 0.95 * len(m.node_ids())


def test_map_type_checks():
    with pytest.raises(TypeError):
        PersistentMap().set("a", 1)
