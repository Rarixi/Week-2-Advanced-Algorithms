"""
graph.py
--------
A Graph that can be backed by either of two internal representations, chosen at
construction time:

    representation="list"    -> adjacency list: {node: {neighbor: weight}}
    representation="matrix"  -> adjacency matrix: a 2D list indexed by node_index[node]

Both support directed and undirected graphs (set via `directed`) and weighted edges
(default weight=1 for unweighted use).

Complexity, list vs. matrix (V = number of nodes, E = number of edges):
    add_node       O(1) list   | O(V) matrix (every existing row gains one column)
    add_edge       O(1) list   | O(1) matrix
    remove_edge    O(1) list   | O(1) matrix
    remove_node    O(V) list   | O(V^2) matrix (a full row + column shift)
    get_neighbors  O(1) list   | O(V) matrix (must scan the whole row)

The list representation is the better default for sparse graphs, which is why
get_neighbors -- the operation BFS/DFS/Dijkstra all call once per node -- is O(1)
there but O(V) on a matrix (see bfs.py/dfs.py/dijkstra.py docstrings for how that
turns a full traversal from O(V+E) into O(V^2)).

Matrix cells use None, not 0, to mean "no edge": 0 is a legal edge weight (e.g. a
free-cost edge for Dijkstra), so treating 0 as "empty" would make a real zero-weight
edge indistinguishable from no edge at all.
"""


class Graph:
    def __init__(self, directed=False, representation="list"):
        """directed: one-way edges (A->B doesn't imply B->A) if True, else edges mirror
        both ways automatically. representation: "list" or "matrix", see module docstring."""
        self.directed = directed
        self.representation = representation
        self.nodes = set()

        if self.representation == "list":
            self.adj_list = {}
        else:
            self.adj_matrix = []
            # Maps node -> its row/column index, since the matrix itself only understands
            # integer positions, not node names.
            self.node_index = {}

    def add_node(self, node):
        """Add `node` with no edges yet. No-op if it already exists (so calling this
        doesn't wipe out a node's existing edges)."""
        if node in self.nodes:
            return

        self.nodes.add(node)

        if self.representation == "list":
            self.adj_list[node] = {}
        else:
            self.node_index[node] = len(self.node_index)

            for row in self.adj_matrix:
                row.append(None)  # every existing row grows by one column for the new node

            self.adj_matrix.append([None] * len(self.node_index))

    def add_edge(self, u, v, weight=1):
        """Connect u -> v with `weight` (and v -> u too, if undirected). Creates u and/or v
        first if they don't already exist, rather than raising."""
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        if self.representation == "list":
            self.adj_list[u][v] = weight
            if not self.directed:
                self.adj_list[v][u] = weight
        else:
            i, j = self.node_index[u], self.node_index[v]
            self.adj_matrix[i][j] = weight
            if not self.directed:
                self.adj_matrix[j][i] = weight

    def remove_edge(self, u, v):
        """Remove the edge u -> v (and v -> u too, if undirected). Fine to call on a pair
        that isn't actually connected."""
        if self.representation == "list":
            self.adj_list[u].pop(v, None)
            if not self.directed:
                self.adj_list[v].pop(u, None)
        else:
            i, j = self.node_index[u], self.node_index[v]
            self.adj_matrix[i][j] = None
            if not self.directed:
                self.adj_matrix[j][i] = None

    def remove_node(self, node):
        """Remove `node` and every edge touching it. No-op if it doesn't exist."""
        if node not in self.nodes:
            return

        self.nodes.remove(node)

        if self.representation == "list":
            self.adj_list.pop(node, None)
            # Other nodes may still list `node` as a neighbor -- strip those too, or
            # they'd become dangling references to a node that no longer exists.
            for neighbors in self.adj_list.values():
                neighbors.pop(node, None)
        else:
            idx = self.node_index.pop(node)
            del self.adj_matrix[idx]
            for row in self.adj_matrix:
                del row[idx]

            # Every node indexed after the removed one has shifted down by one
            # row/column; the index map has to shift with it.
            for other_node, other_idx in self.node_index.items():
                if other_idx > idx:
                    self.node_index[other_node] = other_idx - 1

    def get_neighbors(self, node):
        """Return {neighbor: weight} for `node`. O(1) on the list representation
        (direct dict lookup); O(V) on the matrix representation (scans the whole row --
        see module docstring)."""
        if self.representation == "list":
            return self.adj_list.get(node, {})
        else:
            i = self.node_index[node]
            return {
                other_node: self.adj_matrix[i][j]
                for other_node, j in self.node_index.items()
                if self.adj_matrix[i][j] is not None
            }

    def __str__(self):
        """Human-readable "node -> {neighbor: weight, ...}", one line per node."""
        lines = []
        for node in self.nodes:
            neighbors = self.get_neighbors(node)
            lines.append(f"{node} -> {neighbors}")
        return "\n".join(lines)
