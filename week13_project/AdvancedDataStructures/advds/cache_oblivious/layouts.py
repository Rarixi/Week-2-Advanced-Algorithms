"""
layouts.py
-------------
Part 3.2 of the AdvancedDataStructures project: memory layouts for a
static binary search tree, and how many cache misses a search costs in
each.

The tree is a complete binary search tree over N = 2^h - 1 sorted keys.
Its nodes can be stored in an array in different orders; the search
algorithm (go left or right at each node) is the same, only the
addresses it touches change:

  * sorted / in-order:  the plain sorted array, searched by binary
    search. The first probes jump n/2, n/4, ... apart: every probe
    until the last ~log2(B) is a different cache line, ~log2(N) - log2(B)
    misses.
  * BFS / heap (Eytzinger): level by level, node i has children 2i+1
    and 2i+2, like an array-based binary heap. The top few levels share
    lines, but below them each level is again one new line:
    ~log2(N) - log2(B) misses, same order as binary search.
  * van Emde Boas (vEB): cut the tree at half its height. Store the top
    subtree (height h/2) recursively in vEB order, then each of the
    ~sqrt(N) bottom subtrees recursively, one after another. At the
    recursion level where subtrees have about B nodes, every subtree is
    contiguous, and a root-to-leaf path crosses only about
    log2(N) / log2(B) of them: O(log_B N) misses, for EVERY B at once
    without knowing B. That's the cache-oblivious layout, and it matches
    a B-tree's bound without being tuned to a block size.

Each layout is represented by `pos[i]` = array position of the node that
has heap (BFS) index i, so one search routine works for all of them:
walk heap indices, read address pos[i] * node_bytes.
"""

import numpy as np

from .cache_sim import IdealCache


def bfs_positions(height):
    """Heap layout: node i is stored at position i."""
    return np.arange((1 << height) - 1, dtype=np.int64)


def inorder_positions(height):
    """Sorted-array layout: node at heap index i is stored at its in-order rank."""
    n = (1 << height) - 1
    pos = np.empty(n, dtype=np.int64)
    # in-order rank of heap index i at depth d with offset o (0..2^d-1):
    # (2o + 1) * 2^(h - d - 1) - 1
    for d in range(height):
        first = (1 << d) - 1
        offsets = np.arange(1 << d, dtype=np.int64)
        pos[first:first + (1 << d)] = (2 * offsets + 1) * (1 << (height - d - 1)) - 1
    return pos


def veb_order(height):
    """
    Heap indices of a complete tree of the given height, listed in van
    Emde Boas order (position k of the array holds node veb_order[k]).
    """
    out = []

    def lay(root, h):
        if h == 1:
            out.append(root)
            return
        top = h // 2                 # top subtree height
        bottom = h - top             # bottom subtrees' height
        lay(root, top)
        # roots of the bottom subtrees: descendants of `root` at depth `top`
        first = (root + 1) * (1 << top) - 1
        for r in range(first, first + (1 << top)):
            lay(r, bottom)

    if height > 0:
        lay(0, height)
    return np.array(out, dtype=np.int64)


def veb_positions(height):
    """pos[i] = vEB array position of heap node i (inverse of veb_order)."""
    order = veb_order(height)
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    return pos


LAYOUTS = {
    "sorted (binary search)": inorder_positions,
    "BFS / heap": bfs_positions,
    "van Emde Boas": veb_positions,
}


class StaticSearchTree:
    """
    A complete BST over sorted keys stored in one of the layouts.

        t = StaticSearchTree(range(0, 2000, 2), layout="van Emde Boas")
        t.contains(1000)        # True
        t.search_path(1000)     # array positions touched, root first
    """

    def __init__(self, keys, layout="van Emde Boas", node_bytes=8):
        keys = np.sort(np.asarray(list(keys)))
        if keys.size == 0:
            raise ValueError("need at least one key")
        self.height = int(np.ceil(np.log2(keys.size + 1)))
        n_full = (1 << self.height) - 1
        if keys.size < n_full:
            # pad with +infinity so the tree is complete; padding is never "found"
            pad_value = np.inf if keys.dtype.kind == "f" else np.iinfo(np.int64).max
            keys = np.concatenate([keys, np.full(n_full - keys.size, pad_value, dtype=keys.dtype)])
        self.real_size = int(np.count_nonzero(keys != (np.inf if keys.dtype.kind == "f"
                                                        else np.iinfo(np.int64).max)))
        self.layout = layout
        self.node_bytes = node_bytes
        self.pos = LAYOUTS[layout](self.height)
        inorder = inorder_positions(self.height)       # sorted rank of each heap node
        self.array = np.empty(n_full, dtype=keys.dtype)
        self.array[self.pos] = keys[inorder]           # key of heap node i lives at pos[i]
        self._pos_list = self.pos.tolist()
        self._arr_list = self.array.tolist()

    def search_path(self, key):
        """Array positions read while searching for key (root to leaf or hit)."""
        path = []
        i = 0
        n = len(self._pos_list)
        while i < n:
            p = self._pos_list[i]
            path.append(p)
            v = self._arr_list[p]
            if key == v:
                break
            i = 2 * i + 1 if key < v else 2 * i + 2
        return path

    def contains(self, key):
        p = self.search_path(key)[-1]
        return self._arr_list[p] == key

    def addresses(self, key, base=0):
        return [base + p * self.node_bytes for p in self.search_path(key)]


def search_misses(tree, queries, cache_bytes=32 * 1024, block_bytes=64, cold=True):
    """
    Cache misses per search, averaged over `queries`.

    cold=True flushes the cache before every search, so the number is the
    count of distinct blocks a single search touches: exactly the quantity
    the O(log_B N) bound is about. cold=False keeps one warm cache across
    all searches (the realistic case, where the top of the tree stays
    cached).
    """
    cache = IdealCache(cache_bytes, block_bytes)
    for q in queries:
        if cold:
            cache.flush()
        cache.run(tree.addresses(q))
    return cache.misses / max(len(queries), 1)


def compare_layouts(height, num_queries=2000, block_sizes=(64, 256, 4096),
                    cache_bytes=32 * 1024, node_bytes=8, seed=0):
    """
    For each layout and block size: average cold misses per search, and
    warm misses with a `cache_bytes` cache. Returns a list of dicts.
    """
    rng = np.random.default_rng(seed)
    n = (1 << height) - 1
    keys = np.arange(n, dtype=np.int64) * 2
    queries = (rng.integers(0, n, num_queries) * 2).tolist()   # successful searches
    rows = []
    for name in LAYOUTS:
        tree = StaticSearchTree(keys, name, node_bytes)
        for b in block_sizes:
            rows.append({
                "layout": name, "height": height, "n": n, "block_bytes": b,
                "cold_misses": search_misses(tree, queries, max(cache_bytes, b), b, cold=True),
                "warm_misses": search_misses(tree, queries, max(cache_bytes, b), b, cold=False),
            })
    return rows
