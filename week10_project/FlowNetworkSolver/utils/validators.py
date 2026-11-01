"""
validators.py
----------------
Part 4 of the flow network project. Validation helpers shared across
every algorithm, all of them reading a Network's own internal flow state
directly (every algorithm in core/ pushes flow straight onto the Network
object via push_flow(), there's no separate flow structure to pass
around), flow conservation at internal nodes, capacity constraints on
every edge, residual graph correctness, and min-cost feasibility.
"""


def check_flow_conservation(network, source, sink, tol=1e-6):
    """True if every node besides source and sink has zero net flow,
    whatever flows in leaves again."""
    for node in network.nodes:
        if node == source or node == sink:
            continue
        if abs(network.flow_value_out_of(node)) > tol:
            return False
    return True


def check_capacity_constraints(network, tol=1e-6):
    """True if no edge's flow goes negative or exceeds its own
    capacity. Backward edges (capacity 0 by construction) are skipped,
    their "flow" is really just bookkeeping for how much of their
    forward twin has been undone, checked instead by
    check_residual_correctness below."""
    for i, edge in enumerate(network.edges):
        if edge.capacity > 0:
            if network.flow[i] < -tol or network.flow[i] > edge.capacity + tol:
                return False
    return True


def check_residual_correctness(network, tol=1e-6):
    """True if every forward/backward edge pair's residual capacities
    still sum to that pair's original forward capacity (the invariant
    push_flow's bookkeeping is supposed to maintain, residual_capacity(i)
    + residual_capacity(i ^ 1) == edges[i].capacity for every forward i),
    and neither side has gone negative."""
    for i in range(0, len(network.edges), 2):
        j = i + 1
        forward_capacity = network.edges[i].capacity
        residual_sum = network.residual_capacity(i) + network.residual_capacity(j)
        if abs(residual_sum - forward_capacity) > tol:
            return False
        if network.residual_capacity(i) < -tol or network.residual_capacity(j) < -tol:
            return False
    return True


def check_min_cost_feasibility(network, expected_cost, tol=1e-6):
    """True if the network's current flow has total cost matching
    expected_cost, cross-checking an algorithm's own reported cost
    against the flow it actually left behind on the network."""
    return abs(network.total_cost() - expected_cost) <= tol
