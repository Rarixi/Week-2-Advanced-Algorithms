"""
matrix_utils.py
-----------------
Shared helpers for building test/benchmark inputs for the matrix- and graph-based
algorithms in src/dp_advanced/: matrix_chain_multiplication.py, floyd_warshall.py, and
bitmask_traveling_salesman.py.

Provides:
    generate_chain_dimensions(n, rng, min_dim=1, max_dim=100)
        -- a random dims list of length n+1 for an n-matrix chain (see
           matrix_chain_multiplication.py's module docstring for the encoding).

    generate_distance_matrix(n, rng, min_dist=1, max_dist=100, symmetric=True)
        -- a random n x n distance matrix for TSP (bitmask_traveling_salesman.py):
           every pair of distinct cities gets a real finite distance (TSP assumes a
           complete graph), and the diagonal is 0.

    generate_sparse_weighted_matrix(n, rng, edge_probability=0.3, min_w=1, max_w=20)
        -- a random n x n adjacency matrix for Floyd-Warshall (floyd_warshall.py),
           using float('inf') for "no direct edge" -- unlike TSP's distance matrix,
           Floyd-Warshall's whole point is finding shortest paths THROUGH intermediate
           nodes, so the input graph should actually be sparse/incomplete, or every
           shortest path would just be its one direct edge.

    print_matrix(matrix, label=None)
        -- a readable, fixed-width printout, with "inf" shown for float('inf') cells
           (used by examples/week6_demo.py and ad-hoc debugging, not by the tests).
"""


def generate_chain_dimensions(n, rng, min_dim=1, max_dim=100):
    """A random dims list of length n+1 for a chain of n matrices: matrix i has shape
    dims[i-1] x dims[i], so consecutive matrices are automatically compatible
    (matrix i's column count IS matrix i+1's row count, by construction)."""
    return [rng.randint(min_dim, max_dim) for _ in range(n + 1)]


def generate_distance_matrix(n, rng, min_dist=1, max_dist=100, symmetric=True):
    """A random n x n distance matrix for TSP: every pair of distinct cities (i, j)
    gets a real, finite distance (the graph is complete, as TSP requires), and
    matrix[i][i] = 0. If `symmetric`, distance(i, j) == distance(j, i) (an undirected
    "map" of cities); if not, the two directions are chosen independently."""
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if symmetric and j < i:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = rng.randint(min_dist, max_dist)
    return matrix


def generate_sparse_weighted_matrix(n, rng, edge_probability=0.3, min_w=1, max_w=20):
    """A random n x n adjacency matrix for Floyd-Warshall: matrix[i][i] = 0, and each
    OTHER ordered pair (i, j) independently gets a direct edge with probability
    `edge_probability` (weight chosen uniformly from [min_w, max_w]); pairs with no
    edge get float('inf'). Deliberately incomplete/sparse -- Floyd-Warshall's purpose
    is finding shortest paths that ROUTE THROUGH intermediate nodes, which only
    matters when direct edges don't already connect every pair."""
    matrix = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < edge_probability:
                matrix[i][j] = rng.randint(min_w, max_w)
    return matrix


def print_matrix(matrix, label=None):
    """Readable, fixed-width printout of a 2D matrix -- float('inf') cells print as
    "inf" instead of the much wider "inf" Python would otherwise format inconsistently
    alongside integers."""
    if label:
        print(label)
    for row in matrix:
        cells = [("inf" if v == float("inf") else str(v)) for v in row]
        width = max(len(c) for c in cells) if cells else 0
        print(" ".join(c.rjust(width) for c in cells))
