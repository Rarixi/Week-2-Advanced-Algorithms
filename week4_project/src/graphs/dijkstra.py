"""
dijkstra.py
-----------
Dijkstra's single-source shortest path algorithm over a Graph (src/graphs/graph.py),
using the MinHeap-backed PriorityQueue built in Week 3 (src/data_structures/heap.py).

Provides:
    dijkstra(graph, source)             -> (distances, predecessors), heap-based, O(E log V)
    dijkstra_list_based(graph, source)  -> the same result, using a plain O(V) linear scan
                                            instead of a heap, purely so
                                            benchmarks/dijkstra_benchmark.py has something
                                            slower to compare the heap version against.
                                            This is NOT the required implementation -- it
                                            exists only to make the heap's speedup measurable.

Both require non-negative edge weights (Dijkstra's algorithm isn't correct with negative
weights -- see the ValueError raised below) and both handle disconnected graphs gracefully:
a node that isn't reachable from `source` just gets distance float('inf') and predecessor
None, rather than being silently dropped from the result or causing a crash.

Returns, from both functions:
    distances:    {node: shortest distance from source to node}, float('inf') if unreachable
    predecessors: {node: the node right before it on the shortest path from source, or None
                   if it IS the source, or if it's unreachable}

To reconstruct an actual path (not just its length), walk predecessors backward from the
destination to the source:

    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()
    # if path[0] != source: destination was unreachable, this isn't a real path

Time complexity: O(E log V) for dijkstra(). Every edge relaxation can push at most one new
entry onto the priority queue (see the "lazy deletion" note in dijkstra() below -- Week 3's
PriorityQueue has no decrease-key operation, so instead of updating an entry in place we push
a new, better one and skip the old one later), so the queue holds at most O(E) entries total,
and each push/pop costs O(log E), which is O(log V) since E is at most V^2 (so log E is at
most 2 log V). dijkstra_list_based() is O(V^2 + E): V rounds of an O(V) linear scan to find
the next-closest unvisited node, plus O(E) total edge relaxations along the way.
"""


from src.data_structures.heap import PriorityQueue


def _validate_non_negative_weights(graph):
    """Dijkstra's algorithm assumes every edge weight is non-negative. Once a node is
    finalized (its shortest distance is treated as settled), the algorithm never revisits
    it -- but a negative edge could still find a shorter path back to an already-finalized
    node later on, which it would then have no mechanism to detect. Rather than silently
    hand back a wrong answer, fail loudly up front instead."""
    for node in graph.nodes:
        for neighbor, weight in graph.get_neighbors(node).items():
            if weight < 0:
                raise ValueError(
                    f"Dijkstra's algorithm requires non-negative edge weights; "
                    f"got {node} -> {neighbor} with weight {weight}"
                )


def dijkstra(graph, source):
    """Heap-based Dijkstra. Returns (distances, predecessors) -- see module docstring."""
    _validate_non_negative_weights(graph)

    distances = {node: float("inf") for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}

    if source not in graph.nodes:
        # Nothing is reachable from a source that doesn't exist -- every distance stays inf,
        # same as "handle disconnected graphs gracefully" for a node with zero connections.
        return distances, predecessors

    distances[source] = 0

    pq = PriorityQueue()
    pq.enqueue(source, priority=0)

    visited = set()

    while not pq.is_empty():
        node = pq.dequeue()

        if node in visited:
            # This is the "lazy deletion" half of the decrease-key workaround: since we
            # can't reach into the heap and update an entry that's already sitting there,
            # we push a brand-new entry every time we find a shorter distance to some node,
            # and just leave the old, now-stale entry where it is. By the time a stale
            # entry finally gets dequeued, that node has already been finalized with a
            # distance at least as good -- so we skip it here instead of processing it
            # (and its outgoing edges) a second time.
            continue

        visited.add(node)

        for neighbor, weight in graph.get_neighbors(node).items():
            candidate = distances[node] + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                pq.enqueue(neighbor, priority=candidate)
                # ^ Push a fresh, better entry rather than trying to modify one already in
                #   the heap -- see the note above. Any older, worse entry for `neighbor`
                #   is still sitting in the heap somewhere; it'll just get skipped by the
                #   `if node in visited` check above whenever it eventually surfaces.

    return distances, predecessors


def dijkstra_list_based(graph, source):
    """The textbook O(V^2) version of Dijkstra: instead of a heap, "the priority queue" is
    just the plain set of not-yet-finalized nodes, and finding the next-closest one means
    scanning that whole collection every single round. This function exists ONLY so
    benchmarks/dijkstra_benchmark.py has a slower baseline to compare the heap-based
    dijkstra() against -- it is not the algorithm this assignment asks you to hand in.
    Returns (distances, predecessors), same shape as dijkstra()."""
    _validate_non_negative_weights(graph)

    distances = {node: float("inf") for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}

    if source not in graph.nodes:
        return distances, predecessors

    distances[source] = 0
    unvisited = set(graph.nodes)
    # ^ "The list-based priority queue" -- just a plain collection of nodes, with no
    #   ordering structure at all. Finding the minimum means checking every element.

    while unvisited:
        node = min(unvisited, key=lambda n: distances[n])
        # ^ This one line is the entire difference from the heap version: an O(V) scan
        #   through every unvisited node, every single round, for up to V rounds -- O(V^2)
        #   total, versus the heap's O(log V) per push/pop.

        if distances[node] == float("inf"):
            # Every remaining node is unreachable from the source -- nothing left to relax,
            # so stop early instead of doing pointless scans over the rest of `unvisited`.
            break

        unvisited.remove(node)

        for neighbor, weight in graph.get_neighbors(node).items():
            if neighbor not in unvisited:
                continue
            candidate = distances[node] + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node

    return distances, predecessors
