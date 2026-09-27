"""
test_graph_generator.py
------------------------
pytest tests for the four graph generators in src/utils/graph_generator.py:
generate_sparse_graph, generate_dense_graph, generate_random_graph, generate_weighted_graph.

These aren't algorithm-correctness tests in the same sense as test_bfs.py/test_dfs.py/
test_dijkstra.py -- there's no single "right answer" for a random graph. What IS checkable,
and what these tests focus on, is whether each generator actually delivers the shape it
promises: right node count, roughly the right edge count/density, weights in the right
range, no self-loops, and reproducibility given the same random source. If a generator
silently produced the wrong density, benchmarks/week4_performance.py's whole comparison
across "sparse vs. dense vs. random vs. weighted" would be built on a false premise.

"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.utils.graph_generator import (
    generate_sparse_graph,
    generate_dense_graph,
    generate_random_graph,
    generate_weighted_graph,
)


def edge_count(graph):
    return sum(len(graph.get_neighbors(node)) for node in graph.nodes)


def all_weights(graph):
    weights = []
    for node in graph.nodes:
        weights.extend(graph.get_neighbors(node).values())
    return weights


class TestGenerateSparseGraph:
    def test_has_the_right_number_of_nodes(self):
        g = generate_sparse_graph(50, rng=random.Random(1))
        assert g.nodes == set(range(50))

    def test_edge_count_exactly_matches_the_target(self):
        # _add_random_edges rejects duplicate (u, v) picks rather than silently
        # overwriting them, so the edge count should land on the target exactly, not
        # just "close to" it -- see graph_generator.py's docstring for why that
        # distinction matters for a density-comparison benchmark.
        g = generate_sparse_graph(200, edges_per_node=2, rng=random.Random(1))
        assert edge_count(g) == 200 * 2

    def test_is_meaningfully_sparser_than_a_dense_graph_of_the_same_size(self):
        sparse = generate_sparse_graph(100, rng=random.Random(1))
        dense = generate_dense_graph(100, rng=random.Random(1))
        # The whole point of having separate generators: these two should not be
        # remotely close in edge count for the same n.
        assert edge_count(sparse) < edge_count(dense) / 10

    def test_no_self_loops(self):
        g = generate_sparse_graph(30, rng=random.Random(1))
        for node in g.nodes:
            assert node not in g.get_neighbors(node)

    def test_weighted_false_gives_every_edge_weight_one(self):
        g = generate_sparse_graph(30, weighted=False, rng=random.Random(1))
        assert all(w == 1 for w in all_weights(g))

    def test_default_weight_range_is_respected(self):
        g = generate_sparse_graph(100, edges_per_node=4, min_weight=5, max_weight=9, rng=random.Random(1))
        weights = all_weights(g)
        assert weights  # sanity check the graph actually has edges to inspect
        assert all(5 <= w <= 9 for w in weights)

    def test_same_rng_seed_reproduces_the_same_graph(self):
        g1 = generate_sparse_graph(40, rng=random.Random(7))
        g2 = generate_sparse_graph(40, rng=random.Random(7))
        # Same seed fed to independent random.Random instances -> identical sequence of
        # random choices -> identical resulting edges. This is what makes a benchmark
        # run reproducible across two different machines or two different days.
        assert {n: g1.get_neighbors(n) for n in g1.nodes} == {n: g2.get_neighbors(n) for n in g2.nodes}


class TestGenerateDenseGraph:
    def test_has_the_right_number_of_nodes(self):
        g = generate_dense_graph(50, rng=random.Random(1))
        assert g.nodes == set(range(50))

    def test_edge_count_exactly_matches_the_requested_density(self):
        n = 100
        g = generate_dense_graph(n, density=0.5, rng=random.Random(1))
        assert edge_count(g) == int(0.5 * n * (n - 1))

    def test_higher_density_means_more_edges(self):
        sparse_density = generate_dense_graph(80, density=0.1, rng=random.Random(1))
        dense_density = generate_dense_graph(80, density=0.8, rng=random.Random(2))
        assert edge_count(dense_density) > edge_count(sparse_density)

    def test_no_self_loops(self):
        g = generate_dense_graph(30, density=0.5, rng=random.Random(1))
        for node in g.nodes:
            assert node not in g.get_neighbors(node)


class TestGenerateRandomGraph:
    def test_has_the_right_number_of_nodes(self):
        g = generate_random_graph(40, rng=random.Random(1))
        assert g.nodes == set(range(40))

    def test_zero_probability_means_no_edges(self):
        g = generate_random_graph(30, edge_probability=0.0, rng=random.Random(1))
        assert edge_count(g) == 0

    def test_probability_one_means_every_possible_edge_exists(self):
        # With p=1.0, EVERY (u, v) pair with u != v gets an edge -- a complete directed
        # graph. This is a clean, exactly-checkable case, unlike the probabilistic ones.
        n = 15
        g = generate_random_graph(n, edge_probability=1.0, rng=random.Random(1))
        assert edge_count(g) == n * (n - 1)
        for node in g.nodes:
            assert set(g.get_neighbors(node).keys()) == g.nodes - {node}

    def test_higher_probability_means_more_edges_on_average(self):
        low = generate_random_graph(60, edge_probability=0.05, rng=random.Random(1))
        high = generate_random_graph(60, edge_probability=0.5, rng=random.Random(2))
        assert edge_count(high) > edge_count(low)

    def test_no_self_loops(self):
        g = generate_random_graph(30, edge_probability=0.3, rng=random.Random(1))
        for node in g.nodes:
            assert node not in g.get_neighbors(node)

    def test_same_rng_seed_reproduces_the_same_graph(self):
        g1 = generate_random_graph(40, edge_probability=0.1, rng=random.Random(7))
        g2 = generate_random_graph(40, edge_probability=0.1, rng=random.Random(7))
        assert {n: g1.get_neighbors(n) for n in g1.nodes} == {n: g2.get_neighbors(n) for n in g2.nodes}


class TestGenerateWeightedGraph:
    def test_has_the_right_number_of_nodes(self):
        g = generate_weighted_graph(50, rng=random.Random(1))
        assert g.nodes == set(range(50))

    def test_uses_a_much_wider_weight_range_than_the_other_generators(self):
        # The defining feature of this generator: min_weight=1, max_weight=1000 by
        # default, versus 1-20 everywhere else. With enough edges, the weights actually
        # observed should span a meaningfully wide range, not cluster in a narrow band.
        g = generate_weighted_graph(200, edges_per_node=5, rng=random.Random(1))
        weights = all_weights(g)
        assert weights
        assert min(weights) <= 50          # at least one edge is cheap
        assert max(weights) >= 500         # at least one edge is expensive
        assert all(1 <= w <= 1000 for w in weights)

    def test_custom_weight_range_is_respected(self):
        g = generate_weighted_graph(100, edges_per_node=4, min_weight=200, max_weight=210, rng=random.Random(1))
        weights = all_weights(g)
        assert weights
        assert all(200 <= w <= 210 for w in weights)

    def test_no_self_loops(self):
        g = generate_weighted_graph(30, rng=random.Random(1))
        for node in g.nodes:
            assert node not in g.get_neighbors(node)


class TestGeneratorsShareTheSameGraphInterface:
    """All four generators are meant to be interchangeable as far as callers (like
    week4_performance.py) are concerned -- same Graph interface, just different
    topology/weights underneath. bfs_full/dfs_full/dijkstra shouldn't need to know or
    care which generator built the graph they're given."""

    GENERATORS = [generate_sparse_graph, generate_dense_graph, generate_random_graph, generate_weighted_graph]

    @pytest.mark.parametrize("generator", GENERATORS)
    def test_produces_a_graph_usable_by_bfs_dfs_dijkstra(self, generator):
        from src.graphs.bfs import bfs_full
        from src.graphs.dfs import dfs_full
        from src.graphs.dijkstra import dijkstra

        g = generator(25, rng=random.Random(3))

        bfs_order = bfs_full(g)
        dfs_order = dfs_full(g)
        distances, predecessors = dijkstra(g, 0)

        assert set(bfs_order) == g.nodes
        assert set(dfs_order) == g.nodes
        assert set(distances.keys()) == g.nodes
        assert set(predecessors.keys()) == g.nodes
