"""
dfs.py
------
Depth-First Search over a Graph (src/graphs/graph.py).

Provides:
    dfs_iterative(graph, start)  -> traversal order from one start node, using an explicit
                                    stack (no Python recursion involved)
    dfs_recursive(graph, start)  -> the same traversal, using actual function-call recursion
    dfs_full(graph, algorithm)   -> traversal order across the ENTIRE graph, including nodes
                                    not reachable from any single start node -- this is what
                                    makes disconnected graphs work correctly

All three return a plain list of nodes in the order they were first visited. That list is
exactly what you diff against an expected order in tests, or feed to a visualization (e.g.
animating "this node lit up at step 4").

Time complexity: O(V + E) for a graph built with representation="list". Every node is visited
at most once, and get_neighbors() is only ever called once per visited node, so summed across
the whole traversal, the total work is proportional to (number of nodes) + (number of edges
examined). NOTE: if the graph is built with representation="matrix" instead, get_neighbors()
itself costs O(V) per call (it has to scan an entire row looking for non-zero cells), so a full
traversal over a matrix-backed graph is really O(V^2), not O(V+E) -- see
benchmarks/traversal_benchmark.py for how that difference actually shows up on sparse vs. dense
graphs.
"""

import sys


def _dfs_iterative_from(graph, start, visited, order):
    """Internal helper: explicit-stack DFS starting at `start`, mutating `visited` and
    `order` IN PLACE rather than returning fresh ones.

    Shared by dfs_iterative() and dfs_full() for the same reason bfs.py's _bfs_from is
    shared: dfs_full() has to start fresh traversals from more than one node (one per
    component), and each of those needs to see everything already visited by EARLIER
    traversals. In a directed graph, two different "unvisited" starting nodes can both
    be able to reach the same downstream node (A -> C and B -> C, say) -- without a
    shared visited set, C would get fully re-explored once per root that reaches it,
    which both breaks the O(V+E) guarantee and makes C show up more than once in the
    returned order.
    """
    if start in visited:
        return

    stack = [start]

    while stack:
        node = stack.pop()
        # ^ LIFO: pop the most recently pushed node. That's what makes this depth-FIRST --
        #   we keep chasing the newest branch as deep as it goes before backing up, instead
        #   of working outward level by level (that's BFS's job, using a queue instead).

        if node in visited:
            # A node can end up pushed onto the stack more than once, if two different
            # already-visited neighbors both pointed at it before it was ever popped.
            # Skip it the second time instead of visiting it twice.
            continue

        visited.add(node)
        order.append(node)

        neighbors = list(graph.get_neighbors(node).keys())
        for neighbor in reversed(neighbors):
            # Pushed in reverse order so that, once popped one at a time, they come off the
            # stack in the same left-to-right order dfs_recursive visits them in below. This
            # isn't required for correctness -- any order is a valid DFS -- but it makes the
            # iterative and recursive versions directly comparable in tests.
            if neighbor not in visited:
                stack.append(neighbor)


def dfs_iterative(graph, start):
    """Depth-first traversal starting at `start`, using an explicit stack instead of
    recursion. Returns the list of nodes in the order they were first visited."""
    if start not in graph.nodes:
        # Nothing to traverse if the start node doesn't even exist in the graph.
        return []

    visited = set()
    order = []
    _dfs_iterative_from(graph, start, visited, order)
    return order


def _dfs_recursive_from(graph, node, visited, order):
    """Internal helper: the recursive half of dfs_recursive() / dfs_full(algorithm="recursive"),
    mutating `visited` and `order` in place for the same sharing reason described in
    _dfs_iterative_from's docstring above."""
    visited.add(node)
    order.append(node)
    for neighbor in graph.get_neighbors(node):
        if neighbor not in visited:
            _dfs_recursive_from(graph, neighbor, visited, order)
            # ^ This recursive call IS the stack. Each call that hasn't returned yet is
            #   one frame "pushed" on Python's call stack, in exactly the same role as
            #   stack.append(...) above -- which is also why a sufficiently deep or wide
            #   graph can hit Python's recursion limit here in a way dfs_iterative never
            #   will (it uses a plain list, not the call stack, so it isn't bounded by it).


def dfs_recursive(graph, start):
    """The same traversal as dfs_iterative, but using real recursion: Python's own call
    stack stands in for the explicit `stack` list above. Returns nodes in visit order."""
    if start not in graph.nodes:
        return []

    visited = set()
    order = []
    _dfs_recursive_from(graph, start, visited, order)
    return order


def dfs_full(graph, algorithm="iterative"):
    """Depth-first traversal that covers the ENTIRE graph, not just whatever's reachable from
    one start node. A single dfs_iterative/dfs_recursive call from one start node will
    silently miss every node that lives in a different connected component (or, for a
    directed graph, every node that isn't reachable by following edges forward from start) --
    this function is what makes disconnected graphs work correctly.

    algorithm: "iterative" (default) or "recursive" -- picks which single-start DFS is used
    to explore each component.

    Uses ONE shared `visited` set across every component (see _dfs_iterative_from's
    docstring) -- without that, a directed graph where several nodes can reach the same
    downstream node would re-traverse shared regions of the graph once per root that
    reaches them, which both breaks the O(V+E) guarantee and duplicates nodes in the
    returned order.

    Returns a list of nodes in visit order. Nodes within one component appear together (in
    normal DFS order); once a component is exhausted, traversal picks an arbitrary unvisited
    node and starts a fresh DFS from there, repeating until every node has been visited.
    """
    visit_from = _dfs_iterative_from if algorithm == "iterative" else _dfs_recursive_from

    visited = set()
    order = []

    for node in graph.nodes:
        # Walking every node in the graph (not just ones reachable from a single start) is
        # what guarantees disconnected components don't get silently skipped.
        if node not in visited:
            visit_from(graph, node, visited, order)

    return order

