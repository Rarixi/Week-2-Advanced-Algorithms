"""
graph_utils.py
-----------------
Shared helpers for generating test instances and doing basic bookkeeping
on them, covering the four instance types week 9 needs, graphs for vertex
cover and max cut, point sets for metric TSP, and facility/client layouts
plus set cover universes for the LP based problems.
"""

import math
import random


# ---------------------------------------------------------------------------
# Graphs (vertex cover, max cut)
# ---------------------------------------------------------------------------

def connected_random_graph(num_vertices, edge_probability, rng=None):
    """
    Random undirected graph on num_vertices labeled 0..num_vertices-1. Lays
    down a random spanning path first so every vertex has at least one
    edge, then adds extra random edges on top. Returns (vertices, edges)
    with edges a sorted list of (u, v) tuples, u < v.
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
    """True if every edge has at least one endpoint in cover."""
    cover = set(cover)
    return all(u in cover or v in cover for u, v in edges)


# ---------------------------------------------------------------------------
# Metric point sets (TSP)
# ---------------------------------------------------------------------------

def random_points(num_points, width=1000.0, height=1000.0, rng=None):
    """Random 2d points for a tsp instance, a list of (x, y) floats."""
    rng = rng or random.Random()
    return [(rng.uniform(0, width), rng.uniform(0, height)) for _ in range(num_points)]


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def tour_length(tour, points):
    """Total distance of visiting `tour` (a list of point indices) in
    order and returning to the start, the standard closed tsp tour cost."""
    total = 0.0
    for i in range(len(tour)):
        a = points[tour[i]]
        b = points[tour[(i + 1) % len(tour)]]
        total += euclidean_distance(a, b)
    return total


# ---------------------------------------------------------------------------
# Facility location
# ---------------------------------------------------------------------------

def random_facility_location_instance(num_facilities, num_clients, width=1000.0, height=1000.0,
                                       opening_cost_range=(50.0, 200.0), rng=None):
    """
    Random Uncapacitated Facility Location instance. Facilities and clients
    both get random 2d coordinates (so connection cost is just euclidean
    distance, keeping the instance genuinely metric), and each facility
    gets a random opening cost. Returns (facility_costs, distances), where
    distances[client][facility] is the connection cost.
    """
    rng = rng or random.Random()
    facility_points = random_points(num_facilities, width, height, rng=rng)
    client_points = random_points(num_clients, width, height, rng=rng)
    facility_costs = [rng.uniform(*opening_cost_range) for _ in range(num_facilities)]
    distances = [
        [euclidean_distance(client, facility) for facility in facility_points]
        for client in client_points
    ]
    return facility_costs, distances


# ---------------------------------------------------------------------------
# Set cover
# ---------------------------------------------------------------------------

def random_set_cover_instance(num_elements, num_sets, set_size_range=(2, 5),
                               cost_range=(1.0, 10.0), rng=None):
    """
    Random Set Cover instance guaranteed to actually be coverable, every
    element is force-assigned to at least one random set before the rest
    of each set is filled in randomly, so there's never an element no set
    can reach. Returns (universe, sets, costs), universe a set of element
    ids, sets a list of frozensets, costs a parallel list of floats.
    """
    rng = rng or random.Random()
    universe = set(range(num_elements))

    sets = [set() for _ in range(num_sets)]
    # guarantee coverability, every element goes into at least one random set
    for element in universe:
        sets[rng.randrange(num_sets)].add(element)

    # round out each set's size with more random elements
    for s in sets:
        target_size = rng.randint(*set_size_range)
        while len(s) < target_size:
            s.add(rng.randrange(num_elements))

    sets = [frozenset(s) for s in sets]
    costs = [rng.uniform(*cost_range) for _ in range(num_sets)]
    return universe, sets, costs
