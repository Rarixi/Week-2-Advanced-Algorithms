"""
network.py
-------------
Part 1 of the flow network project. The core graph data structure every
algorithm in this project builds on.

Every edge is stored as a forward/backward pair. add_edge(u, v, capacity,
cost) appends a forward edge u -> v with the given capacity and cost, and
immediately appends its paired backward residual edge v -> u with 0
capacity (and negated cost, so cost bookkeeping stays correct when
min_cost_flow.py pushes flow backward to cancel an earlier choice).
Storing the pair back to back at indices 2k and 2k+1 means each edge's
twin is always index ^ 1, the standard trick for O(1) access to an
edge's reverse without a lookup.

Pushing f units of flow along edge index i means edges[i].flow += f and
edges[i ^ 1].flow -= f, the second line is what makes a backward edge's
residual capacity (capacity - flow, which for a 0 capacity backward edge
becomes -flow, i.e. "how much forward flow can be undone") grow exactly
as the forward edge's residual capacity shrinks. That's the entire
mechanism that lets every algorithm in this project treat "flow along a
backward edge" as ordinary residual capacity rather than a special case.
"""

from collections import defaultdict


class Edge:
    """One direction of a flow edge. capacity and cost are fixed at
    construction, flow changes as algorithms push along it."""

    __slots__ = ("u", "v", "capacity", "cost")

    def __init__(self, u, v, capacity, cost=0.0):
        self.u = u
        self.v = v
        self.capacity = capacity
        self.cost = cost


class Network:
    """
    Flow network representation. Nodes can be any hashable value (ints,
    strings, tuples for grid coordinates, whatever the application
    needs). Edges are stored in a flat list, each forward edge's
    reverse sits at the paired index (index ^ 1), and adj[node] lists
    every edge index whose tail is that node, forward or backward.
    """

    def __init__(self):
        self.adj = defaultdict(list)
        self.edges = []
        self.flow = []
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, u, v, capacity, cost=0.0):
        """Adds a forward edge u -> v with the given capacity and cost,
        automatically creating its paired backward residual edge v -> u
        (capacity 0, cost negated). Returns (forward_index,
        backward_index). Calling add_edge(u, v, ...) more than once
        between the same pair is allowed and creates parallel edges,
        each with its own residual pair."""
        self.add_node(u)
        self.add_node(v)

        forward = Edge(u, v, capacity, cost)
        backward = Edge(v, u, 0.0, -cost)
        self.edges.append(forward)
        self.edges.append(backward)
        self.flow.append(0.0)
        self.flow.append(0.0)

        forward_index = len(self.edges) - 2
        backward_index = len(self.edges) - 1
        self.adj[u].append(forward_index)
        self.adj[v].append(backward_index)
        return forward_index, backward_index

    def reverse_of(self, edge_index):
        """The paired reverse edge's index, index ^ 1 by construction."""
        return edge_index ^ 1

    def residual_capacity(self, edge_index):
        edge = self.edges[edge_index]
        return edge.capacity - self.flow[edge_index]

    def push_flow(self, edge_index, amount):
        """Pushes `amount` units of flow along edges[edge_index],
        crediting the paired reverse edge's residual capacity by the
        same amount. amount must not exceed the edge's current residual
        capacity."""
        if amount < 0:
            raise ValueError("push_flow amount must be non-negative")
        if amount > self.residual_capacity(edge_index) + 1e-9:
            raise ValueError("push_flow amount exceeds residual capacity")
        self.flow[edge_index] += amount
        self.flow[self.reverse_of(edge_index)] -= amount

    def reset_flow(self):
        """Zeroes every edge's flow, so the same Network (and its
        capacities/costs) can be reused across multiple algorithm runs,
        which is exactly what the benchmark needs to compare
        Edmonds-Karp, Push-Relabel, and Min-Cost Flow on identical
        instances."""
        self.flow = [0.0 for _ in self.edges]

    def get_residual_graph(self):
        """Returns {u: {v: residual_capacity}}, aggregating every edge
        (forward or backward, including parallel edges) from u to v into
        one combined residual capacity. This is a read-only view built
        fresh from the current flow, not a structure algorithms mutate
        directly, they go through push_flow() on specific edge indices
        instead, since two edges between the same pair of nodes can have
        different per-edge residual capacities worth tracking
        separately."""
        residual = defaultdict(lambda: defaultdict(float))
        for edge_index, edge in enumerate(self.edges):
            cap = self.residual_capacity(edge_index)
            if cap > 1e-12:
                residual[edge.u][edge.v] += cap
        return {u: dict(neighbors) for u, neighbors in residual.items()}

    def edges_from(self, node):
        """Every edge index whose tail is `node`, forward and backward
        alike, the adjacency view every augmenting-path search walks."""
        return self.adj[node]

    def flow_value_out_of(self, node):
        """Net flow leaving `node` across every edge touching it, summed
        over forward edges (positive contribution) and backward edges
        (their flow is negative by construction, so they subtract
        correctly without special casing)."""
        return sum(self.flow[i] for i in self.adj[node])

    def total_cost(self):
        """Sum of flow * cost across every edge with positive flow
        (backward edges carry negative flow and negative cost, so they'd
        double count the same shipment if included, only forward-sign
        flow contributes)."""
        return sum(
            self.flow[i] * self.edges[i].cost
            for i in range(len(self.edges))
            if self.flow[i] > 1e-12
        )

    def flow_matrix(self):
        """Returns {(u, v): flow} for every edge actually carrying
        positive flow, the human-readable flow assignment the
        algorithms return alongside their max-flow / min-cost value."""
        result = {}
        for i, edge in enumerate(self.edges):
            if self.flow[i] > 1e-9:
                key = (edge.u, edge.v)
                result[key] = result.get(key, 0.0) + self.flow[i]
        return result


def is_valid_flow(network, source, sink, tol=1e-6):
    """
    Flow validation helper. True if every edge respects its capacity (0
    <= flow <= capacity for every forward edge) and every node besides
    source and sink has zero net flow (what comes in leaves again). Used
    by utils/validators.py and by every algorithm's own test suite to
    confirm a returned flow is actually legal, not just high value.
    """
    for i, edge in enumerate(network.edges):
        if edge.capacity > 0:
            if network.flow[i] < -tol or network.flow[i] > edge.capacity + tol:
                return False

    for node in network.nodes:
        if node == source or node == sink:
            continue
        if abs(network.flow_value_out_of(node)) > tol:
            return False

    return True
