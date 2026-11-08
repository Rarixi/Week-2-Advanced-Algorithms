"""
suffix_tree.py
------------------
Part 2.2 of the string search project, built as Option A, a compressed
suffix trie, every suffix of the text inserted into a trie and chains of
single-child nodes merged into one edge, rather than Option B's full
Ukkonen online construction. The payoff of either one is the same,
O(1) time substring containment checks and O(occurrence count) listing
of every position a substring occurs at, paid for by an upfront build.

One design choice matters enough to call out directly: every edge here
stores a (start, end) pair of indices into the original text rather than
a copied substring. That's not just a style preference, it's required
for correctness at scale. A string with every character distinct (say
"abcdefgh...") has suffixes that share nothing with each other beyond
the root, so a trie over them has O(n) nodes but, if each edge copied
its own label text, O(n^2) total characters stored, the classic suffix
tree space blowup. Storing index pairs instead keeps total memory at
O(n) integers no matter how the edges end up shaped, recovering the
actual label text only when a query needs to compare against it.

The honest trade off of picking the compressed trie over Ukkonen's
algorithm is construction time, not space. Building this trie inserts
each of the n suffixes one at a time, and an insertion can touch O(n)
existing structure in the worst case, so construction here is O(n^2) in
the worst case rather than Ukkonen's guaranteed O(n). That's the real
reason the benchmark suite in this project runs the suffix tree on
noticeably smaller inputs than KMP, Rabin-Karp, or the suffix array get,
the same honest tradeoff tension as Push-Relabel needing the gap
heuristic in the prior project, a textbook-correct structure with a
practical cost that only shows up once you actually run it at scale.
"""


class _Node:
    __slots__ = ("children", "end_positions")

    def __init__(self):
        self.children = {}  # first_char -> [edge_start, edge_end, child_node]
        self.end_positions = set()  # at most one element, see module notes


def build_suffix_trie(text):
    """
    Inserts every suffix of `text` into a fresh compressed trie and
    returns its root. Call the other functions in this module with the
    same `text` and this root together, the tree only stores index pairs
    into `text`, not copies of it.
    """
    root = _Node()
    n = len(text)
    for start in range(n):
        _insert_suffix(root, text, start)
    return root


def _insert_suffix(root, text, suffix_start):
    node = root
    n = len(text)
    i = suffix_start

    while i < n:
        c = text[i]
        if c not in node.children:
            leaf = _Node()
            leaf.end_positions.add(suffix_start)
            node.children[c] = [i, n, leaf]
            return

        edge_start, edge_end, child = node.children[c]
        edge_len = edge_end - edge_start

        j = 0
        while j < edge_len and i + j < n and text[edge_start + j] == text[i + j]:
            j += 1

        if j == edge_len:
            node = child
            i += j
            if i == n:
                node.end_positions.add(suffix_start)
                return
            continue

        # mismatch (or suffix ran out) partway through this edge, split it
        split_node = _Node()
        split_node.children[text[edge_start + j]] = [edge_start + j, edge_end, child]
        node.children[c] = [edge_start, edge_start + j, split_node]

        if i + j == n:
            split_node.end_positions.add(suffix_start)
        else:
            new_leaf = _Node()
            new_leaf.end_positions.add(suffix_start)
            split_node.children[text[i + j]] = [i + j, n, new_leaf]
        return


def _collect_end_positions(node):
    positions = set(node.end_positions)
    for _, _, child in node.children.values():
        positions |= _collect_end_positions(child)
    return positions


def substring_positions(text, root, pattern):
    """
    Every start index of `pattern` in `text`, found by walking the trie
    character by character and, once the pattern is fully consumed,
    collecting the end positions of every leaf still reachable from
    wherever that walk ended (every suffix passing through that point
    has `pattern` as a prefix, which is exactly what makes it an
    occurrence). O(len(pattern) + number of occurrences).
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")

    node = root
    i = 0
    m = len(pattern)
    target = None

    while i < m:
        c = pattern[i]
        if c not in node.children:
            return []
        edge_start, edge_end, child = node.children[c]
        edge_len = edge_end - edge_start
        compare_len = min(edge_len, m - i)
        if text[edge_start:edge_start + compare_len] != pattern[i:i + compare_len]:
            return []
        i += compare_len
        target = child
        if compare_len == edge_len and i < m:
            node = child

    return sorted(_collect_end_positions(target))


def prefix_positions(text, root, prefix):
    """
    Alias of substring_positions. Listed as its own task in the
    assignment, but in a suffix tree these are mechanically the same
    walk, every substring occurrence IS a prefix match against the path
    from the root, there's no separate structure or algorithm needed for
    "starts with" versus "contains" the way there would be for, say, a
    plain word trie over a list of independent strings (autocomplete.py
    builds exactly that different structure for its own prefix queries).
    """
    return substring_positions(text, root, prefix)


def _branch_count(node):
    """How many distinct ways there are to continue past this exact
    point, every child counts as one, and the node's own end_positions
    marker counts as one more, "stop here" is just as much a distinct
    continuation as "keep going with this next character" is. A node
    where this is 2 or more is where at least two different suffixes
    were still indistinguishable, which is exactly what it means for
    that node's path to be a repeated substring."""
    return len(node.children) + (1 if node.end_positions else 0)


def longest_repeated_substring(text, root):
    """
    The longest substring that occurs at least twice, read off the tree
    rather than the LCP array this time, find the deepest node (by path
    length from the root) with a branch count of 2 or more. Note this
    is NOT simply "2 or more children", a substring can also repeat by
    occurring once as a suffix that ends exactly at this node and again
    as a prefix of some longer suffix that keeps going past it, "banana"
    is the smallest example, "ana" occurs once as the whole suffix
    starting at index 3 and again as a prefix of "anana" starting at
    index 1, so the node for "ana" has only 1 child but also carries its
    own end marker, 2 distinct continuations either way. Used in this
    project's tests to cross check against suffix_array.py's
    longest_repeated_substring_via_lcp, two different data structures
    over the same text should always agree.
    """
    best_depth = 0
    best_node = None

    def dfs(node, depth):
        nonlocal best_depth, best_node
        if _branch_count(node) >= 2 and depth > best_depth:
            best_depth = depth
            best_node = node
        for edge_start, edge_end, child in node.children.values():
            dfs(child, depth + (edge_end - edge_start))

    dfs(root, 0)

    if best_node is None or best_depth == 0:
        return ""

    positions = _collect_end_positions(best_node)
    start = min(positions)
    return text[start:start + best_depth]


def count_distinct_substrings(text, root):
    """
    The number of distinct substrings of `text`, computed as the sum of
    every edge's length across the whole trie. This works because every
    point you could stop at while walking down from the root, whether
    exactly at a node or partway along an edge, corresponds to exactly
    one distinct substring, and the trie's compression guarantees no two
    different stopping points ever spell out the same string. Summing
    edge lengths is just counting all of those stopping points at once,
    O(number of nodes).
    """
    total = 0

    def dfs(node):
        nonlocal total
        for edge_start, edge_end, child in node.children.values():
            total += edge_end - edge_start
            dfs(child)

    dfs(root)
    return total


def count_nodes(root):
    """Total node count, used by the benchmark suite as a rough stand in
    for this structure's memory footprint alongside tracemalloc."""
    total = 1  # the root itself

    def dfs(node):
        nonlocal total
        for _, _, child in node.children.values():
            total += 1
            dfs(child)

    dfs(root)
    return total
