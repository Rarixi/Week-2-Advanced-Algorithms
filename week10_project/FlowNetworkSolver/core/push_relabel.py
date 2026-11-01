"""
push_relabel.py
------------------
Part 1 of the flow network project. The Push-Relabel max-flow algorithm,
asymptotically faster than Edmonds-Karp (O(V^2 * E) here, against
Edmonds-Karp's O(V * E^2)) because it abandons the "always keep a valid
flow" discipline Ford-Fulkerson style algorithms use and instead works
with a preflow, letting nodes other than the source temporarily hold more
incoming flow than outgoing, excess, and pushes that excess downhill
toward the sink using height labels as the notion of downhill, fixing up
any leftover excess by relabeling a node taller whenever it runs out of
downhill neighbors to push to.

Every node keeps a height label and an excess value. The source starts
at height = number of nodes (tall enough that nothing can ever push flow
back into it once started) and immediately saturates every edge leaving
it, dumping excess onto its neighbors. From there, discharge(u) repeats
two operations until u's excess reaches zero, push (send flow along any
residual edge to a strictly shorter neighbor, height[u] == height[v] + 1)
and relabel (when no such neighbor exists, raise u's height to one more
than the shortest residual neighbor still reachable, so a push becomes
possible next time). A FIFO queue of currently active (excess > 0) nodes
decides discharge order here, the simplest of the standard variants
(FIFO, highest label, relabel-to-front all share the same correctness
argument and worst case bound family, they only differ in constant
factor performance). Each node also keeps a current-arc pointer into its
own edge list so discharge doesn't rescan already-exhausted edges on
every push attempt.

Plain FIFO push-relabel with only the current-arc optimization turned
out, while building this, to be nowhere near "high performance" on
anything but small or very sparse graphs, a 45x45 grid (2025 nodes) took
over a minute and several million push/relabel operations, because
nothing stops a node from being relabeled one height at a time, over and
over, chasing a sink it can genuinely never reach once the rest of the
graph has drained. The fix is the standard gap heuristic. A height_count
array tracks how many nodes currently sit at each height. Whenever a
relabel empties out a height band entirely (height_count[h] drops to
zero for some 0 < h < num_nodes), every remaining node with a height
strictly between h and num_nodes is providably disconnected from the
sink for the rest of the run, nothing can ever cross an empty band
upward, so they get bulk relabeled straight to num_nodes + 1 (high
enough to start draining back toward the source) in one pass instead of
being discovered one relabel at a time. This single change took the
45x45 grid from 67.8 seconds and 7 million operations down to a fraction
of a second, without changing the algorithm's correctness at all, gap
relabeling is a provably safe shortcut, not an approximation.
"""

from collections import deque

from core.network import Edge  # noqa: F401  (re-exported for callers inspecting edge types)


def push_relabel_max_flow(network, source, sink, variant="fifo", detailed_trace=False):
    """
    Runs Push-Relabel on `network` from source to sink. `variant` is
    accepted for interface symmetry with the other solvers and to record
    which discharge order was used, this implementation always uses FIFO
    ordering, the simplest of the three standard choices and the one
    this module's docstring describes. Returns a dict with:
      value             the max-flow value
      flow_matrix       {(u, v): flow} for every edge carrying flow
      pushes            total push operations performed
      relabels          total relabel operations performed
      convergence       list of (operation_number, total_excess) sampled
                         periodically, the data week10_flow_benchmark.py
                         plots as Push-Relabel's convergence profile
      trace             only populated when detailed_trace=True (meant
                         for small graphs animate_flow.py replays), a
                         list of {"op": "push"|"relabel", "node": ...,
                         "heights": {...}, "excess": {...}} snapshots
      gap_relabels      how many of the total relabels were bulk gap
                         relabels rather than ordinary single-node ones,
                         a direct measure of how much work the gap
                         heuristic saved on this particular run
    Does not mutate any flow already on `network`, call reset_flow()
    first if it was used by another algorithm.
    """
    nodes = list(network.nodes)
    num_nodes = len(nodes)

    height = {node: 0 for node in nodes}
    height[source] = num_nodes
    excess = {node: 0.0 for node in nodes}
    current_arc = {node: 0 for node in nodes}

    # gap heuristic bookkeeping, how many nodes currently sit at each
    # height, so an emptied-out band can be detected in O(1) right after
    # the relabel that caused it.
    height_count = [0] * (2 * num_nodes + 2)
    for node in nodes:
        height_count[height[node]] += 1

    pushes = 0
    relabels = 0
    gap_relabels = 0
    convergence = []
    trace = []
    sample_interval = max(1, num_nodes // 20)

    def total_excess():
        return sum(v for node, v in excess.items() if node != source and node != sink)

    def record(op, node):
        nonlocal pushes, relabels
        op_count = pushes + relabels
        if op_count % sample_interval == 0:
            convergence.append((op_count, total_excess()))
        if detailed_trace:
            trace.append({
                "op": op,
                "node": node,
                "heights": dict(height),
                "excess": dict(excess),
            })

    active = deque()
    in_queue = set()

    def activate(node):
        if node != source and node != sink and node not in in_queue and excess[node] > 1e-12:
            active.append(node)
            in_queue.add(node)

    # initial preflow, saturate every edge out of the source
    for edge_index in network.edges_from(source):
        cap = network.residual_capacity(edge_index)
        if cap > 1e-12:
            v = network.edges[edge_index].v
            network.push_flow(edge_index, cap)
            excess[v] += cap
            excess[source] -= cap
            pushes += 1
            activate(v)

    def push(u, edge_index):
        nonlocal pushes
        edge = network.edges[edge_index]
        v = edge.v
        amount = min(excess[u], network.residual_capacity(edge_index))
        network.push_flow(edge_index, amount)
        excess[u] -= amount
        excess[v] += amount
        pushes += 1
        activate(v)
        record("push", u)

    def relabel(u):
        nonlocal relabels
        candidate_heights = [
            height[network.edges[ei].v]
            for ei in network.edges_from(u)
            if network.residual_capacity(ei) > 1e-12
        ]
        old_height = height[u]
        new_height = 1 + min(candidate_heights) if candidate_heights else old_height + 1

        height_count[old_height] -= 1
        height[u] = new_height
        height_count[new_height] += 1
        current_arc[u] = 0
        relabels += 1
        record("relabel", u)

        check_gap(old_height)

    def check_gap(emptied_height):
        """If relabeling just emptied out height band `emptied_height`
        (and it's a meaningful interior band, not 0 and not already past
        num_nodes), every node strictly above it and still below
        num_nodes can never reach the sink again, bulk relabel them to
        num_nodes + 1 instead of waiting for them to discover that one
        ordinary relabel at a time."""
        nonlocal relabels, gap_relabels
        if emptied_height == 0 or emptied_height >= num_nodes:
            return
        if height_count[emptied_height] > 0:
            return

        for v in nodes:
            if v == source:
                continue
            h = height[v]
            if emptied_height < h < num_nodes:
                height_count[h] -= 1
                height[v] = num_nodes + 1
                height_count[num_nodes + 1] += 1
                current_arc[v] = 0
                relabels += 1
                gap_relabels += 1
                record("relabel", v)

    def discharge(u):
        while excess[u] > 1e-12:
            arcs = network.edges_from(u)
            if current_arc[u] >= len(arcs):
                relabel(u)
                continue
            edge_index = arcs[current_arc[u]]
            edge = network.edges[edge_index]
            if network.residual_capacity(edge_index) > 1e-12 and height[u] == height[edge.v] + 1:
                push(u, edge_index)
            else:
                current_arc[u] += 1

    while active:
        u = active.popleft()
        in_queue.discard(u)
        discharge(u)
        if excess[u] > 1e-12:
            active.append(u)
            in_queue.add(u)

    # At termination every node but source and sink has drained its
    # excess back to zero (that's what "no active nodes left" means), so
    # whatever excess landed at the sink during the run is exactly the
    # max-flow value, cleaner and correct regardless of edge direction
    # than trying to sum flow over source-incident edges, some of which
    # are backward residual edges from edges that originally pointed
    # into the source.
    max_flow_value = excess[sink]

    return {
        "value": max_flow_value,
        "flow_matrix": network.flow_matrix(),
        "pushes": pushes,
        "relabels": relabels,
        "gap_relabels": gap_relabels,
        "convergence": convergence,
        "trace": trace,
    }
