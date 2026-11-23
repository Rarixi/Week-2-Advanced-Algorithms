"""
persistent_tree.py
---------------------
Part 1.3 of the AdvancedDataStructures project: persistent (immutable,
versioned) trees.

A data structure is *persistent* if an update produces a new version
while every old version stays readable. The technique here is PATH
COPYING: nodes are never modified after they're created. An update
copies only the nodes on the root-to-leaf path it touches (O(log n) of
them) and points the copies at the old, untouched subtrees. Every version
is just a root pointer, and all versions together form a DAG of shared
nodes rather than a forest of separate copies. Memory per update is
O(log n) instead of O(n) for a full copy.

Two structures:

  * PersistentSegmentTree - the persistent version of segment_tree.py.
    Point assignment and range add create a new version; sum/min/max
    queries work on any version; diff(v1, v2) lists changed positions
    while skipping every subtree the two versions share.

    Range add can't use ordinary lazy propagation, because pushing a tag
    down would MODIFY children that other versions share. Instead it
    uses "permanent" tags: a fully covered node gets a copy with its tag
    increased, the tag is never pushed down, and queries add up the tags
    on their way down from the root. Each node's cached sum/min/max
    includes its own tag but not its ancestors'.

  * PersistentMap - a persistent hash trie (a 16-way trie on the SHA-1
    of each key, collapsed so a subtree holding one key is just that
    leaf). Every node carries a Merkle hash of its contents, so two maps
    with identical contents have identical root hashes no matter how
    they were built, and diffing two versions only descends into
    subtrees whose hashes differ. That is exactly how Git compares
    trees, and it's what the version-control prototype stores file
    states in.
"""

import hashlib


# ===========================================================================
# persistent segment tree
# ===========================================================================

class PersistentSegmentTree:
    """
        pst = PersistentSegmentTree([1, 2, 3, 4])   # version 0
        v1 = pst.update(0, 2, 10)                   # version 0 with a[2] = 10
        v2 = pst.range_add(v1, 0, 3, 1)             # version 1 plus 1 everywhere
        pst.query_sum(0, 0, 3)   # 10   (version 0 is untouched)
        pst.query_sum(v2, 0, 3)  # 21
        pst.diff(0, v2)          # [(0, 1, 2), (1, 2, 3), (2, 3, 11), (3, 4, 5)]

    Any version can be updated, not just the newest, so versions form a
    tree (branching history); `parents[v]` records where each came from.
    """

    def __init__(self, values):
        values = list(values)
        if not values:
            raise ValueError("PersistentSegmentTree needs at least one value")
        self.n = len(values)
        # node pool: parallel lists, node id = index
        self._left = []
        self._right = []
        self._sum = []
        self._min = []
        self._max = []
        self._tag = []
        root = self._build(0, self.n - 1, values)
        self.roots = [root]
        self.parents = [None]
        self.labels = ["initial"]
        self.nodes_visited = 0

    # -- node pool ---------------------------------------------------------

    def _new(self, left, right, s, mn, mx, tag):
        self._left.append(left)
        self._right.append(right)
        self._sum.append(s)
        self._min.append(mn)
        self._max.append(mx)
        self._tag.append(tag)
        return len(self._sum) - 1

    def _new_internal(self, left, right, tag, length):
        return self._new(left, right,
                         self._sum[left] + self._sum[right] + tag * length,
                         min(self._min[left], self._min[right]) + tag,
                         max(self._max[left], self._max[right]) + tag,
                         tag)

    def _build(self, lo, hi, values):
        if lo == hi:
            v = values[lo]
            return self._new(-1, -1, v, v, v, 0)
        mid = (lo + hi) // 2
        a = self._build(lo, mid, values)
        b = self._build(mid + 1, hi, values)
        return self._new_internal(a, b, 0, hi - lo + 1)

    @property
    def node_count(self):
        return len(self._sum)

    @property
    def num_versions(self):
        return len(self.roots)

    def _check_version(self, version):
        if not 0 <= version < len(self.roots):
            raise IndexError(f"no version {version}")

    def _check(self, l, r):
        if not (0 <= l <= r < self.n):
            raise IndexError(f"range [{l}, {r}] outside [0, {self.n - 1}]")

    # -- updates (each returns a new version number) -------------------------

    def _set(self, node, lo, hi, i, v, acc):
        if lo == hi:
            base = v - acc            # ancestors' tags will be added back on read
            return self._new(-1, -1, base, base, base, 0)
        tag = self._tag[node]
        mid = (lo + hi) // 2
        left, right = self._left[node], self._right[node]
        if i <= mid:
            left = self._set(left, lo, mid, i, v, acc + tag)
        else:
            right = self._set(right, mid + 1, hi, i, v, acc + tag)
        return self._new_internal(left, right, tag, hi - lo + 1)

    def _add(self, node, lo, hi, l, r, v):
        if r < lo or hi < l:
            return node                                   # shared, not copied
        length = hi - lo + 1
        if l <= lo and hi <= r:
            return self._new(self._left[node], self._right[node],
                             self._sum[node] + v * length,
                             self._min[node] + v, self._max[node] + v,
                             self._tag[node] + v)
        mid = (lo + hi) // 2
        left = self._add(self._left[node], lo, mid, l, r, v)
        right = self._add(self._right[node], mid + 1, hi, l, r, v)
        return self._new_internal(left, right, self._tag[node], length)

    def _commit(self, base, root, label):
        self.roots.append(root)
        self.parents.append(base)
        self.labels.append(label)
        return len(self.roots) - 1

    def update(self, version, i, value, label=None):
        """New version = `version` with a[i] = value. O(log n) new nodes."""
        self._check_version(version)
        self._check(i, i)
        root = self._set(self.roots[version], 0, self.n - 1, i, value, 0)
        return self._commit(version, root, label or f"a[{i}] = {value}")

    def range_add(self, version, l, r, value, label=None):
        """New version = `version` with a[l..r] += value. O(log n) new nodes."""
        self._check_version(version)
        self._check(l, r)
        root = self._add(self.roots[version], 0, self.n - 1, l, r, value)
        return self._commit(version, root, label or f"a[{l}..{r}] += {value}")

    # -- queries -----------------------------------------------------------

    def _query(self, node, lo, hi, l, r, acc):
        self.nodes_visited += 1
        if r < lo or hi < l:
            return None
        if l <= lo and hi <= r:
            return (self._sum[node] + acc * (hi - lo + 1),
                    self._min[node] + acc, self._max[node] + acc)
        acc += self._tag[node]
        mid = (lo + hi) // 2
        a = self._query(self._left[node], lo, mid, l, r, acc)
        b = self._query(self._right[node], mid + 1, hi, l, r, acc)
        if a is None:
            return b
        if b is None:
            return a
        return a[0] + b[0], min(a[1], b[1]), max(a[2], b[2])

    def query(self, version, l, r):
        """(sum, min, max) of a[l..r] as of `version`. O(log n)."""
        self._check_version(version)
        self._check(l, r)
        return self._query(self.roots[version], 0, self.n - 1, l, r, 0)

    def query_sum(self, version, l, r):
        return self.query(version, l, r)[0]

    def query_min(self, version, l, r):
        return self.query(version, l, r)[1]

    def query_max(self, version, l, r):
        return self.query(version, l, r)[2]

    def get(self, version, i):
        return self.query(version, i, i)[0]

    def to_list(self, version):
        self._check_version(version)
        out = [0] * self.n

        def walk(node, lo, hi, acc):
            if lo == hi:
                out[lo] = self._sum[node] + acc
                return
            acc += self._tag[node]
            mid = (lo + hi) // 2
            walk(self._left[node], lo, mid, acc)
            walk(self._right[node], mid + 1, hi, acc)

        walk(self.roots[version], 0, self.n - 1, 0)
        return out

    # -- version comparison ------------------------------------------------

    def diff(self, v1, v2):
        """
        Positions whose value differs between two versions, as
        [(index, value_in_v1, value_in_v2), ...]. Shared subtrees (same
        node id, same accumulated tags above) are skipped without being
        entered, so the cost is proportional to the size of the change,
        not to n. `last_diff_nodes` records how many nodes were visited.
        """
        self._check_version(v1)
        self._check_version(v2)
        out = []
        visited = [0]

        def values_under(node, lo, hi, acc, sink):
            if lo == hi:
                sink.append(self._sum[node] + acc)
                return
            acc += self._tag[node]
            mid = (lo + hi) // 2
            values_under(self._left[node], lo, mid, acc, sink)
            values_under(self._right[node], mid + 1, hi, acc, sink)

        def walk(a, b, lo, hi, acc_a, acc_b):
            visited[0] += 1
            if a == b:
                if acc_a == acc_b:
                    return                                  # identical shared subtree
                # same subtree under different ancestor tags: every element
                # differs by the same nonzero amount
                vals = []
                values_under(a, lo, hi, 0, vals)
                for k, x in enumerate(vals):
                    out.append((lo + k, x + acc_a, x + acc_b))
                return
            if lo == hi:
                x, y = self._sum[a] + acc_a, self._sum[b] + acc_b
                if x != y:
                    out.append((lo, x, y))
                return
            ta, tb = acc_a + self._tag[a], acc_b + self._tag[b]
            mid = (lo + hi) // 2
            walk(self._left[a], self._left[b], lo, mid, ta, tb)
            walk(self._right[a], self._right[b], mid + 1, hi, ta, tb)

        walk(self.roots[v1], self.roots[v2], 0, self.n - 1, 0, 0)
        self.last_diff_nodes = visited[0]
        out.sort()
        return out

    def nodes_reachable(self, version):
        """Distinct nodes in one version's tree (2n - 1 for a full tree)."""
        return self._reachable([self.roots[version]])

    def _reachable(self, roots):
        seen = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node < 0 or node in seen:
                continue
            seen.add(node)
            stack.append(self._left[node])
            stack.append(self._right[node])
        return len(seen)

    def sharing_stats(self):
        """
        How much path copying saves: nodes actually stored for all
        versions vs. what storing every version as a separate full tree
        would cost.
        """
        stored = self._reachable(self.roots)
        full = (2 * self.n - 1) * len(self.roots)
        return {"versions": len(self.roots), "nodes_stored": stored,
                "nodes_if_full_copies": full, "saving_factor": full / stored}

    def version_history(self, version):
        """The chain of versions from 0 to `version` (following parents)."""
        chain = []
        v = version
        while v is not None:
            chain.append(v)
            v = self.parents[v]
        return chain[::-1]


# ===========================================================================
# persistent hash trie with Merkle hashes
# ===========================================================================

_FANOUT_BITS = 4
_FANOUT = 1 << _FANOUT_BITS          # 16 children per branch
_DIGITS = 160 // _FANOUT_BITS        # SHA-1 gives 160 bits = 40 hex digits


def _sha1(data):
    return hashlib.sha1(data.encode() if isinstance(data, str) else data).hexdigest()


class _Leaf:
    __slots__ = ("key", "value", "path", "hash")

    def __init__(self, key, value, path):
        self.key = key
        self.value = value
        self.path = path            # hex SHA-1 of the key: its route through the trie
        self.hash = _sha1(f"L\0{key}\0{value}")

    size = 1


class _Branch:
    __slots__ = ("children", "size", "hash")

    def __init__(self, children):
        self.children = children     # tuple of 16: None, _Leaf or _Branch
        self.size = sum(c.size for c in children if c is not None)
        self.hash = _sha1("B" + "".join(c.hash if c is not None else "-" for c in children))


class PersistentMap:
    """
    Immutable string -> string map. Every update returns a new map that
    shares all untouched nodes with the old one.

        m0 = PersistentMap()
        m1 = m0.set("README.md", "blob-a").set("main.py", "blob-b")
        m2 = m1.set("main.py", "blob-c")
        m1.get("main.py")      # 'blob-b'  (m1 is unchanged)
        m1.diff(m2)            # {'added': [], 'removed': [], 'modified': ['main.py']}

    Keys are routed by the hex digits of SHA-1(key), one digit per level.
    A subtree that holds a single key is stored as that key's leaf, so
    the shape (and therefore the Merkle root hash) depends only on the
    contents, never on insertion order.
    """

    __slots__ = ("_root",)

    def __init__(self, _root=None):
        self._root = _root

    # -- basic info ------------------------------------------------------

    def __len__(self):
        return 0 if self._root is None else self._root.size

    @property
    def root_hash(self):
        return _sha1("") if self._root is None else self._root.hash

    def __eq__(self, other):
        return isinstance(other, PersistentMap) and self.root_hash == other.root_hash

    def __hash__(self):
        return hash(self.root_hash)

    def __contains__(self, key):
        return self.get(key) is not None

    # -- lookup ------------------------------------------------------------

    def get(self, key, default=None):
        path = _sha1(key)
        node = self._root
        depth = 0
        while isinstance(node, _Branch):
            node = node.children[int(path[depth], 16)]
            depth += 1
        if node is not None and node.key == key:
            return node.value
        return default

    def items(self):
        out = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if isinstance(node, _Leaf):
                out.append((node.key, node.value))
            else:
                stack.extend(node.children)
        return sorted(out)

    def keys(self):
        return [k for k, _ in self.items()]

    # -- updates -----------------------------------------------------------

    @staticmethod
    def _merge_leaves(a, b, depth):
        """Branches needed to separate two leaves whose paths agree up to `depth`."""
        da, db = int(a.path[depth], 16), int(b.path[depth], 16)
        children = [None] * _FANOUT
        if da == db:
            children[da] = PersistentMap._merge_leaves(a, b, depth + 1)
        else:
            children[da] = a
            children[db] = b
        return _Branch(tuple(children))

    def _insert(self, node, leaf, depth):
        if node is None:
            return leaf
        if isinstance(node, _Leaf):
            if node.key == leaf.key:
                return leaf if node.value != leaf.value else node
            return self._merge_leaves(node, leaf, depth)
        d = int(leaf.path[depth], 16)
        child = node.children[d]
        new_child = self._insert(child, leaf, depth + 1)
        if new_child is child:
            return node
        children = list(node.children)
        children[d] = new_child
        return _Branch(tuple(children))

    def set(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("PersistentMap keys and values are strings")
        root = self._insert(self._root, _Leaf(key, value, _sha1(key)), 0)
        return self if root is self._root else PersistentMap(root)

    def _remove(self, node, key, path, depth):
        if node is None:
            return None
        if isinstance(node, _Leaf):
            return None if node.key == key else node
        d = int(path[depth], 16)
        child = node.children[d]
        new_child = self._remove(child, key, path, depth + 1)
        if new_child is child:
            return node
        children = list(node.children)
        children[d] = new_child
        remaining = [c for c in children if c is not None]
        # keep the canonical form: a subtree holding one key is that leaf
        if len(remaining) == 1 and isinstance(remaining[0], _Leaf):
            return remaining[0]
        if not remaining:
            return None
        return _Branch(tuple(children))

    def delete(self, key):
        root = self._remove(self._root, key, _sha1(key), 0)
        return self if root is self._root else PersistentMap(root)

    def update(self, mapping):
        m = self
        for k, v in mapping.items():
            m = m.set(k, v)
        return m

    # -- comparison --------------------------------------------------------

    def diff(self, other):
        """
        {'added': [...], 'removed': [...], 'modified': [...]} going from
        self to other. Descends only into children whose Merkle hashes
        differ; `diff_nodes_visited` on the result counts how many nodes
        that took.
        """
        added, removed, modified = [], [], []
        visited = [0]

        def leaves(node):
            if node is None:
                return {}
            if isinstance(node, _Leaf):
                return {node.key: node.value}
            out = {}
            for c in node.children:
                out.update(leaves(c))
            return out

        def walk(a, b):
            visited[0] += 1
            if a is b or (a is not None and b is not None and a.hash == b.hash):
                return
            if isinstance(a, _Branch) and isinstance(b, _Branch):
                for ca, cb in zip(a.children, b.children):
                    walk(ca, cb)
                return
            la, lb = leaves(a), leaves(b)
            for k in la.keys() - lb.keys():
                removed.append(k)
            for k in lb.keys() - la.keys():
                added.append(k)
            for k in la.keys() & lb.keys():
                if la[k] != lb[k]:
                    modified.append(k)

        walk(self._root, other._root)
        result = {"added": sorted(added), "removed": sorted(removed), "modified": sorted(modified)}
        result["nodes_visited"] = visited[0]
        return result

    def node_ids(self):
        """id() of every node reachable from this map (for sharing statistics)."""
        seen = set()
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, _Branch):
                stack.extend(node.children)
        return seen

    def depth(self):
        def d(node):
            if not isinstance(node, _Branch):
                return 0
            return 1 + max(d(c) for c in node.children)
        return d(self._root)
