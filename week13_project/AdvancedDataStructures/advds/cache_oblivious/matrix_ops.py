"""
matrix_ops.py
----------------
Part 3.1 of the AdvancedDataStructures project: cache-oblivious matrix
algorithms, compared with the naive loop orders.

A cache-OBLIVIOUS algorithm never mentions the cache size M or line
size B, yet incurs an optimal number of cache misses for every (M, B)
at once. The trick is divide and conquer: keep halving the problem;
at some depth of the recursion the subproblems happen to fit in the
cache, whatever its size, and from then on they run without further
misses. Because that's true at every level of the hierarchy
simultaneously (L1, L2, L3, RAM pages), one algorithm is tuned for all
of them. Compare a cache-AWARE blocked algorithm, which needs the block
size as a parameter and is tuned for one level.

Functions here come in three forms:

  1. Pure-Python reference implementations on lists of lists
     (transpose_naive, transpose_recursive, matmul_naive,
     matmul_recursive, traverse_*), used for correctness and to define
     the exact order of memory accesses.
  2. Address-trace generators (trace_*) that yield, in program order,
     the byte address of every element each algorithm reads or writes,
     assuming row-major n x n matrices of 8-byte floats. These are fed to
     cache_sim.IdealCache to count misses: the main experiment.
  3. numpy versions with a large leaf (transpose_recursive_np,
     transpose_naive_np, transpose_blocked_np, matmul_recursive_np), where
     each leaf is one vectorized numpy operation, so the interpreter
     overhead is small enough for real cache effects to show up in
     wall-clock time on matrices bigger than the cache.

Matrices are square with side n; recursive versions work for any n
(they split the larger dimension, sizes need not be powers of two).
"""

import numpy as np

ELEM = 8  # bytes per element (float64)


# ===========================================================================
# 1. reference implementations
# ===========================================================================

def zeros(n, m=None):
    m = n if m is None else m
    return [[0.0] * m for _ in range(n)]


def transpose_naive(A):
    """B[j][i] = A[i][j], reading A row by row (so B is written column by column)."""
    n, m = len(A), len(A[0])
    B = zeros(m, n)
    for i in range(n):
        Ai = A[i]
        for j in range(m):
            B[j][i] = Ai[j]
    return B


def transpose_recursive(A, leaf=1):
    """
    Cache-oblivious transpose: split the larger of the two dimensions in
    half and transpose each half, down to `leaf` x `leaf` tiles. With
    leaf = 1 nothing about the cache is assumed.
    """
    n, m = len(A), len(A[0])
    B = zeros(m, n)

    def rec(r0, r1, c0, c1):
        if (r1 - r0) <= leaf and (c1 - c0) <= leaf:
            for i in range(r0, r1):
                Ai = A[i]
                for j in range(c0, c1):
                    B[j][i] = Ai[j]
            return
        if (r1 - r0) >= (c1 - c0):
            mid = (r0 + r1) // 2
            rec(r0, mid, c0, c1)
            rec(mid, r1, c0, c1)
        else:
            mid = (c0 + c1) // 2
            rec(r0, r1, c0, mid)
            rec(r0, r1, mid, c1)

    rec(0, n, 0, m)
    return B


def matmul_naive(A, B, order="ijk"):
    """
    Triple loop. "ijk" is the textbook order: the inner loop walks down a
    COLUMN of B, one cache line per element once the matrix is big.
    "ikj" reorders so the inner loop walks along a row of B and of C:
    same arithmetic, sequential access.
    """
    n, p, m = len(A), len(B), len(B[0])
    C = zeros(n, m)
    if order == "ijk":
        for i in range(n):
            Ai, Ci = A[i], C[i]
            for j in range(m):
                s = 0.0
                for k in range(p):
                    s += Ai[k] * B[k][j]
                Ci[j] = s
    elif order == "ikj":
        for i in range(n):
            Ai, Ci = A[i], C[i]
            for k in range(p):
                a = Ai[k]
                Bk = B[k]
                for j in range(m):
                    Ci[j] += a * Bk[j]
    else:
        raise ValueError("order must be 'ijk' or 'ikj'")
    return C


def matmul_recursive(A, B, leaf=1):
    """
    Cache-oblivious divide-and-conquer multiplication C += A B: split the
    largest of the three dimensions (rows of A, shared k, columns of B)
    in half and recurse. Same n^3 multiply-adds as the triple loop; the
    order is what changes. Misses: O(n^3 / (B sqrt(M))) for every cache
    level, versus O(n^3 / B) or worse for the naive order.
    """
    n, p, m = len(A), len(B), len(B[0])
    C = zeros(n, m)

    def rec(i0, i1, k0, k1, j0, j1):
        di, dk, dj = i1 - i0, k1 - k0, j1 - j0
        if di <= leaf and dk <= leaf and dj <= leaf:
            for i in range(i0, i1):
                Ai, Ci = A[i], C[i]
                for k in range(k0, k1):
                    a = Ai[k]
                    Bk = B[k]
                    for j in range(j0, j1):
                        Ci[j] += a * Bk[j]
            return
        if di >= dk and di >= dj:
            mid = (i0 + i1) // 2
            rec(i0, mid, k0, k1, j0, j1)
            rec(mid, i1, k0, k1, j0, j1)
        elif dj >= dk:
            mid = (j0 + j1) // 2
            rec(i0, i1, k0, k1, j0, mid)
            rec(i0, i1, k0, k1, mid, j1)
        else:
            mid = (k0 + k1) // 2
            rec(i0, i1, k0, mid, j0, j1)
            rec(i0, i1, mid, k1, j0, j1)

    rec(0, n, 0, p, 0, m)
    return C


def recursive_order(n, m=None):
    """
    The (i, j) visiting order of a recursive (Z-order / quadrant)
    traversal: split the larger dimension, visit each half. Every
    aligned sub-square is finished before the traversal leaves it.
    """
    m = n if m is None else m
    out = []

    def rec(r0, r1, c0, c1):
        if r1 - r0 == 1 and c1 - c0 == 1:
            out.append((r0, c0))
            return
        if r1 - r0 >= c1 - c0:
            mid = (r0 + r1) // 2
            rec(r0, mid, c0, c1)
            rec(mid, r1, c0, c1)
        else:
            mid = (c0 + c1) // 2
            rec(r0, r1, c0, mid)
            rec(r0, r1, mid, c1)

    if n and m:
        rec(0, n, 0, m)
    return out


def traverse(A, order="recursive", visit=None):
    """
    Visit every element of A in the given order ('row', 'column',
    'recursive'), calling visit(i, j, value). Returns the sum of the
    values if no callback is given (a simple reduction to time).
    """
    n, m = len(A), len(A[0])
    if order == "row":
        idx = ((i, j) for i in range(n) for j in range(m))
    elif order == "column":
        idx = ((i, j) for j in range(m) for i in range(n))
    elif order == "recursive":
        idx = recursive_order(n, m)
    else:
        raise ValueError("order must be row, column or recursive")
    total = 0.0
    for i, j in idx:
        v = A[i][j]
        if visit is not None:
            visit(i, j, v)
        else:
            total += v
    return total


# ===========================================================================
# 2. address traces (row-major, 8-byte elements)
# ===========================================================================

def _addr(base, n, i, j):
    return base + (i * n + j) * ELEM


def trace_traversal(n, order="row"):
    """Addresses read by traverse(A, order) for an n x n matrix at address 0."""
    if order == "row":
        for i in range(n):
            for j in range(n):
                yield (i * n + j) * ELEM
    elif order == "column":
        for j in range(n):
            for i in range(n):
                yield (i * n + j) * ELEM
    else:
        for i, j in recursive_order(n):
            yield (i * n + j) * ELEM


def trace_transpose(n, method="naive", leaf=1, block=None):
    """
    Read A[i][j], write B[j][i] for every element, in the order of the
    chosen method: 'naive' (row order), 'recursive' (cache-oblivious,
    leaf tiles), or 'blocked' (cache-aware tiles of side `block`).
    A at address 0, B right after it.
    """
    base_b = n * n * ELEM
    if method == "naive":
        for i in range(n):
            for j in range(n):
                yield _addr(0, n, i, j)
                yield _addr(base_b, n, j, i)
    elif method == "blocked":
        b = block or 8
        for i0 in range(0, n, b):
            for j0 in range(0, n, b):
                for i in range(i0, min(i0 + b, n)):
                    for j in range(j0, min(j0 + b, n)):
                        yield _addr(0, n, i, j)
                        yield _addr(base_b, n, j, i)
    elif method == "recursive":
        stack = [(0, n, 0, n)]
        while stack:
            r0, r1, c0, c1 = stack.pop()
            if (r1 - r0) <= leaf and (c1 - c0) <= leaf:
                for i in range(r0, r1):
                    for j in range(c0, c1):
                        yield _addr(0, n, i, j)
                        yield _addr(base_b, n, j, i)
                continue
            if (r1 - r0) >= (c1 - c0):
                mid = (r0 + r1) // 2
                stack.append((mid, r1, c0, c1))
                stack.append((r0, mid, c0, c1))
            else:
                mid = (c0 + c1) // 2
                stack.append((r0, r1, mid, c1))
                stack.append((r0, r1, c0, mid))
    else:
        raise ValueError("method must be naive, blocked or recursive")


def trace_matmul(n, method="ijk", leaf=1, block=None):
    """
    Address trace of C = A B for n x n matrices (A at 0, B, then C).
    Each multiply-add reads A[i][k] and B[k][j] and read-modify-writes
    C[i][j] (one access is recorded for C, since both halves of a
    read-modify-write hit the same line).
    """
    nn = n * n * ELEM
    a0, b0, c0 = 0, nn, 2 * nn

    def mac(i, k, j):
        yield a0 + (i * n + k) * ELEM
        yield b0 + (k * n + j) * ELEM
        yield c0 + (i * n + j) * ELEM

    if method == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    yield from mac(i, k, j)
    elif method == "ikj":
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    yield from mac(i, k, j)
    elif method == "blocked":
        b = block or 8
        for i0 in range(0, n, b):
            for k0 in range(0, n, b):
                for j0 in range(0, n, b):
                    for i in range(i0, min(i0 + b, n)):
                        for k in range(k0, min(k0 + b, n)):
                            for j in range(j0, min(j0 + b, n)):
                                yield from mac(i, k, j)
    elif method == "recursive":
        stack = [(0, n, 0, n, 0, n)]
        while stack:
            i0, i1, k0, k1, j0, j1 = stack.pop()
            di, dk, dj = i1 - i0, k1 - k0, j1 - j0
            if di <= leaf and dk <= leaf and dj <= leaf:
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        for j in range(j0, j1):
                            yield from mac(i, k, j)
                continue
            if di >= dk and di >= dj:
                mid = (i0 + i1) // 2
                stack.append((mid, i1, k0, k1, j0, j1))
                stack.append((i0, mid, k0, k1, j0, j1))
            elif dj >= dk:
                mid = (j0 + j1) // 2
                stack.append((i0, i1, k0, k1, mid, j1))
                stack.append((i0, i1, k0, k1, j0, mid))
            else:
                mid = (k0 + k1) // 2
                stack.append((i0, i1, mid, k1, j0, j1))
                stack.append((i0, i1, k0, mid, j0, j1))
    else:
        raise ValueError("method must be ijk, ikj, blocked or recursive")


# ===========================================================================
# 3. numpy versions with vectorized leaves (for real timings)
# ===========================================================================

def transpose_naive_np(A):
    """Copy row i of A into column i of B, one row at a time: every write
    to B strides by a whole row, touching a new cache line per element."""
    n, m = A.shape
    B = np.empty((m, n), dtype=A.dtype)
    for i in range(n):
        B[:, i] = A[i, :]
    return B


def transpose_blocked_np(A, block=64):
    """Cache-aware: tiles of a fixed side chosen for one cache size."""
    n, m = A.shape
    B = np.empty((m, n), dtype=A.dtype)
    for i in range(0, n, block):
        for j in range(0, m, block):
            B[j:j + block, i:i + block] = A[i:i + block, j:j + block].T
    return B


def transpose_recursive_np(A, leaf=32):
    """Cache-oblivious recursion; leaves of side <= `leaf` are one numpy
    copy each (the leaf only bounds interpreter overhead; correctness and
    the miss bound don't depend on it)."""
    n, m = A.shape
    B = np.empty((m, n), dtype=A.dtype)

    def rec(r0, r1, c0, c1):
        if (r1 - r0) <= leaf and (c1 - c0) <= leaf:
            B[c0:c1, r0:r1] = A[r0:r1, c0:c1].T
            return
        if (r1 - r0) >= (c1 - c0):
            mid = (r0 + r1) // 2
            rec(r0, mid, c0, c1)
            rec(mid, r1, c0, c1)
        else:
            mid = (c0 + c1) // 2
            rec(r0, r1, c0, mid)
            rec(r0, r1, mid, c1)

    rec(0, n, 0, m)
    return B


def matmul_naive_np(A, B):
    """Row of C at a time with the 'ijk' inner product over a COLUMN of B:
    C[i, j] = A[i, :] @ B[:, j] for each j. Column access of B is strided."""
    n = A.shape[0]
    m = B.shape[1]
    C = np.empty((n, m))
    Bt_cols = [B[:, j] for j in range(m)]     # strided views, not copies
    for i in range(n):
        Ai = A[i]
        for j in range(m):
            C[i, j] = Ai @ Bt_cols[j]
    return C


def matmul_recursive_np(A, B, leaf=64):
    """Cache-oblivious recursion with numpy leaves (C += A_sub @ B_sub)."""
    n, p = A.shape
    m = B.shape[1]
    C = np.zeros((n, m))

    def rec(i0, i1, k0, k1, j0, j1):
        di, dk, dj = i1 - i0, k1 - k0, j1 - j0
        if di <= leaf and dk <= leaf and dj <= leaf:
            C[i0:i1, j0:j1] += A[i0:i1, k0:k1] @ B[k0:k1, j0:j1]
            return
        if di >= dk and di >= dj:
            mid = (i0 + i1) // 2
            rec(i0, mid, k0, k1, j0, j1)
            rec(mid, i1, k0, k1, j0, j1)
        elif dj >= dk:
            mid = (j0 + j1) // 2
            rec(i0, i1, k0, k1, j0, mid)
            rec(i0, i1, k0, k1, mid, j1)
        else:
            mid = (k0 + k1) // 2
            rec(i0, i1, k0, mid, j0, j1)
            rec(i0, i1, mid, k1, j0, j1)

    rec(0, n, 0, p, 0, m)
    return C
