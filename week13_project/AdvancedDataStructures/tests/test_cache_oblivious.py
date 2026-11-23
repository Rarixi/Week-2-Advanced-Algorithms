"""Cache-oblivious matrix ops and layouts: correctness, and the miss counts behave as theory says."""
import numpy as np
import pytest

from advds.cache_oblivious import matrix_ops as M
from advds.cache_oblivious.cache_sim import CacheHierarchy, IdealCache, simulate
from advds.cache_oblivious.layouts import (LAYOUTS, StaticSearchTree, compare_layouts,
                                           veb_order)


def test_ideal_cache_lru():
    c = IdealCache(cache_bytes=128, block_bytes=64)      # 2 lines
    for a in (0, 64, 0, 128, 64):                         # 128 evicts 64 (LRU), so 64 misses again
        c.access(a)
    assert (c.misses, c.hits) == (4, 1)
    c.flush()
    assert not c.access(0)
    with pytest.raises(ValueError):
        IdealCache(32, 64)


def test_hierarchy_l2_sees_only_l1_misses():
    h = CacheHierarchy((("L1", 128, 64), ("L2", 1024, 64)))
    rep = h.run([0, 64, 128, 0, 64, 128])
    assert rep["L1"] == 6 and rep["L2"] == 3


@pytest.mark.parametrize("n", [1, 2, 5, 16, 21])
def test_matrix_ops_correct(n):
    rng = np.random.default_rng(n)
    A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    Al, Bl = A.tolist(), B.tolist()
    assert np.allclose(M.transpose_naive(Al), A.T)
    assert np.allclose(M.transpose_recursive(Al), A.T)
    assert np.allclose(M.transpose_recursive(Al, leaf=4), A.T)
    assert np.allclose(M.matmul_naive(Al, Bl, "ijk"), A @ B)
    assert np.allclose(M.matmul_naive(Al, Bl, "ikj"), A @ B)
    assert np.allclose(M.matmul_recursive(Al, Bl), A @ B)
    for f in (M.transpose_naive_np, M.transpose_blocked_np, M.transpose_recursive_np):
        assert np.allclose(f(A), A.T)
    assert np.allclose(M.matmul_naive_np(A, B), A @ B)
    assert np.allclose(M.matmul_recursive_np(A, B, leaf=4), A @ B)


def test_non_square():
    A = np.arange(15.0).reshape(3, 5)
    assert np.allclose(M.transpose_recursive(A.tolist()), A.T)
    assert np.allclose(M.transpose_recursive_np(A, leaf=2), A.T)
    B = np.arange(10.0).reshape(5, 2)
    assert np.allclose(M.matmul_recursive(A.tolist(), B.tolist()), A @ B)


def test_recursive_traversal_visits_everything_once():
    for n in (1, 4, 7):
        order = M.recursive_order(n)
        assert sorted(order) == [(i, j) for i in range(n) for j in range(n)]
    A = np.arange(36.0).reshape(6, 6).tolist()
    assert M.traverse(A, "recursive") == M.traverse(A, "column") == M.traverse(A, "row") == 630.0
    seen = []
    M.traverse(A, "row", visit=lambda i, j, v: seen.append(v))
    assert seen == list(range(36))


def test_traces_have_the_right_length():
    for m in ("naive", "blocked", "recursive"):
        assert len(list(M.trace_transpose(9, m))) == 2 * 81
    for m in ("ijk", "ikj", "blocked", "recursive"):
        assert len(list(M.trace_matmul(6, m))) == 3 * 216


def test_recursive_transpose_reaches_compulsory_misses():
    n = 256                         # 2 x 512 KiB of data, cache 8 KiB
    compulsory = 2 * n * n * 8 // 64
    naive = simulate(M.trace_transpose(n, "naive"), 8 * 1024, 64)
    rec = simulate(M.trace_transpose(n, "recursive"), 8 * 1024, 64)
    assert rec == compulsory and naive > 4 * compulsory


def test_recursive_matmul_fewer_misses():
    n = 48
    naive = simulate(M.trace_matmul(n, "ijk"), 8 * 1024, 64)
    rec = simulate(M.trace_matmul(n, "recursive"), 8 * 1024, 64)
    assert rec < naive / 3


def test_column_traversal_worst_recursive_close_to_row():
    n = 256
    row = simulate(M.trace_traversal(n, "row"), 8 * 1024, 64)
    col = simulate(M.trace_traversal(n, "column"), 8 * 1024, 64)
    rec = simulate(M.trace_traversal(n, "recursive"), 8 * 1024, 64)
    assert row == rec == n * n * 8 // 64 and col == n * n


# --- layouts -----------------------------------------------------------------

def test_veb_order_small():
    assert veb_order(1).tolist() == [0]
    assert veb_order(2).tolist() == [0, 1, 2]
    assert veb_order(4).tolist() == [0, 1, 2, 3, 7, 8, 4, 9, 10, 5, 11, 12, 6, 13, 14]
    for h in range(1, 13):
        assert sorted(veb_order(h).tolist()) == list(range(2 ** h - 1))


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_search_correct_in_every_layout(layout):
    keys = list(range(0, 3000, 3))         # 1000 keys, padded to 1023
    t = StaticSearchTree(keys, layout)
    assert all(t.contains(k) for k in keys)
    assert not any(t.contains(k + 1) for k in keys[:300])
    assert t.real_size == 1000 and t.height == 10


def test_veb_fewer_misses_for_every_block_size():
    rows = compare_layouts(16, num_queries=400, block_sizes=(64, 512, 4096))
    by = {(r["layout"], r["block_bytes"]): r["cold_misses"] for r in rows}
    for b in (64, 512, 4096):
        assert by[("van Emde Boas", b)] < 0.65 * by[("BFS / heap", b)]
        assert by[("van Emde Boas", b)] < 0.65 * by[("sorted (binary search)", b)]
