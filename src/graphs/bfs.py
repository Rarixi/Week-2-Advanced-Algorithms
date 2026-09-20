"""
bfs.py
------
Breadth-First Search over a Graph (src/graphs/graph.py).

Provides:
    bfs(graph, start)  -> traversal order from one start node, level by level, using a queue
    bfs_full(graph)    -> traversal order across the ENTIRE graph, including nodes not
                          reachable from any single start node -- this is what makes
                          disconnected graphs work correctly

Both return a plain list of nodes in the order they were first visited -- useful for
validating against an expected order in tests, or for visualizing "which node got touched
at which step."

Time complexity: O(V + E) for a graph built with representation="list" -- every node enters
and leaves the queue exactly once, and get_neighbors() is called exactly once per node, so the
total work sums to (number of nodes) + (number of edges examined). As with DFS, a
representation="matrix" graph makes get_neighbors() itself O(V) per call, so a full traversal
there is really O(V^2) -- see benchmarks/traversal_benchmark.py for the sparse-vs-dense,
list-vs-matrix comparison.
"""

from collections import deque


def _bfs_from(graph, start, visited, order):
    """Internal helper that does the actual queue-based traversal, starting at `start`,
    mutating `visited` and `order` IN PLACE rather than returning fresh ones.

    This is shared by both bfs() and bfs_full() for an important reason: bfs_full() has
    to run BFS from more than one start node (once per connected component), and every
    one of those calls needs to see nodes already visited by EARLIER calls -- otherwise,
    in a directed graph where multiple nodes can reach the same downstream node (e.g.
    A -> C and B -> C), a node like C would get rediscovered and fully re-processed once
    for every root that reaches it. That's not just wasted work: it breaks the O(V+E)
    guarantee (a node whose in-degree is k could get reprocessed k times) and it means
    the SAME node shows up more than once in the returned order, when "visit order" is
    supposed to mean each node appears exactly once.
    """
    if start in visited:
        return

    visited.add(start)
    queue = deque([start])

    while queue:
        node = queue.popleft()
        # ^ FIFO: take whichever node has been waiting longest. That's what makes this
        #   breadth-FIRST -- we fully drain one "ring" of nodes before moving out to the
        #   next ring -- compare to dfs's stack.pop(), which always grabs the newest
        #   arrival instead.

        order.append(node)

        for neighbor in graph.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


def bfs(graph, start):
    """Breadth-first traversal starting at `start`. Visits every one of start's direct
    neighbors before visiting any of THEIR neighbors -- nearest nodes first -- which is what
    makes BFS the right tool whenever "fewest edges to get there" matters (shortest path in
    an unweighted graph, "friends of friends" style queries, and so on). Returns nodes in the
    order they were first visited."""
    if start not in graph.nodes:
        return []

    visited = set()
    order = []
    _bfs_from(graph, start, visited, order)
    return order


def bfs_full(graph):
    """Breadth-first traversal that covers the ENTIRE graph, one connected component at a
    time, so disconnected graphs are handled gracefully instead of a plain bfs() call
    silently only covering whatever happens to be reachable from a single start node.

    Uses ONE shared `visited` set across every component (see _bfs_from's docstring) --
    without that, a directed graph where several nodes can reach the same downstream node
    would re-traverse shared regions of the graph once per root that reaches them.
    """
    visited = set()
    order = []

    for node in graph.nodes:
        # Walking every node in the graph (not just ones reachable from a single start) is
        # what guarantees disconnected components -- and, for directed graphs, nodes with
        # no incoming edges from anywhere already explored -- don't get silently skipped.
        if node not in visited:
            _bfs_from(graph, node, visited, order)

    return order
