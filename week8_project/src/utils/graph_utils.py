"""
graph_utils.py
---------------
Shared helpers for generating test instances and doing basic bookkeeping on
them. Covers the three instance types this week actually needs, graphs for
vertex cover, cnf formulas for sat, and point sets for tsp, plus the little
pieces of math (distances, tour length, cover validity) that show up in more
than one of the complexity modules.
"""

import math
import random


def random_graph(num_vertices, edge_probability, rng=None):
    """
    Build a random undirected graph on num_vertices labeled 0..num_vertices-1.
    Each possible edge is included independently with edge_probability.
    Returns (vertices, edges) where vertices is a list of ints and edges is a
    sorted list of (u, v) tuples with u < v (no self loops, no duplicates).
    """
    rng = rng or random.Random()
    vertices = list(range(num_vertices))
    edges = []
    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if rng.random() < edge_probability:
                edges.append((u, v))
    return vertices, edges


def connected_random_graph(num_vertices, edge_probability, rng=None):
    """
    Same as random_graph, but first lays down a random spanning path so every
    vertex has at least one edge, then adds extra random edges on top. Vertex
    cover and tsp instances are a lot less interesting (and less comparable
    across sizes) if a chunk of vertices just float around disconnected.
    """
    rng = rng or random.Random()
    vertices = list(range(num_vertices))
    order = vertices[:]
    rng.shuffle(order)

    edge_set = set()
    for i in range(len(order) - 1):
        u, v = order[i], order[i + 1]
        edge_set.add((min(u, v), max(u, v)))

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if (u, v) not in edge_set and rng.random() < edge_probability:
                edge_set.add((u, v))

    return vertices, sorted(edge_set)


def build_adjacency(vertices, edges):
    """Return {vertex: set(neighbors)} for the given vertex/edge lists."""
    adjacency = {v: set() for v in vertices}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def is_vertex_cover(cover, edges):
    """True if every edge has at least one endpoint in cover (a set of vertices)."""
    cover = set(cover)
    return all(u in cover or v in cover for u, v in edges)


def random_points(num_points, width=1000.0, height=1000.0, rng=None):
    """Random 2d points for a tsp instance, as a list of (x, y) floats."""
    rng = rng or random.Random()
    return [(rng.uniform(0, width), rng.uniform(0, height)) for _ in range(num_points)]


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def tour_length(tour, points):
    """Total distance of visiting `tour` (a list of point indices) in order
    and returning to the start, the standard closed tsp tour cost."""
    total = 0.0
    for i in range(len(tour)):
        a = points[tour[i]]
        b = points[tour[(i + 1) % len(tour)]]
        total += euclidean_distance(a, b)
    return total


def random_cnf(num_vars, num_clauses, clause_size=3, rng=None):
    """
    Build a random cnf formula over variables 1..num_vars. Each clause is a
    list of clause_size distinct literals (an int k means variable k, -k
    means its negation), with the variable for each literal chosen without
    repeats within a clause and the polarity chosen at random. Returns a list
    of clauses.
    """
    rng = rng or random.Random()
    clauses = []
    for _ in range(num_clauses):
        chosen_vars = rng.sample(range(1, num_vars + 1), min(clause_size, num_vars))
        clause = [v if rng.random() < 0.5 else -v for v in chosen_vars]
        clauses.append(clause)
    return clauses


def random_subset_sum_instance(n, max_value=100, rng=None):
    """
    Build a random subset sum instance, n positive integers plus a target
    that is the sum of a random subset of them, so a solution is always
    guaranteed to exist (useful for measuring worst case style runtime
    without every instance secretly being the easy "no solution" case).
    """
    rng = rng or random.Random()
    numbers = [rng.randint(1, max_value) for _ in range(n)]
    subset_size = rng.randint(1, n)
    chosen_indices = rng.sample(range(n), subset_size)
    target = sum(numbers[i] for i in chosen_indices)
    return numbers, target
