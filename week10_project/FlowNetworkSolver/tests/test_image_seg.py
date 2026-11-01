"""
test_image_seg.py
--------------------
Tests for applications/image_seg.py. Builds tiny synthetic images with an
obvious correct segmentation (a dark block next to a light block) and
checks the graph cut actually finds it, plus checks the mask's shape
and that build_pixel_graph wires up the terminals and neighbor edges
correctly on a minimal 1x2 image.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from applications.image_seg import (
    build_pixel_graph,
    segment_image,
    default_similarity,
    default_likelihoods,
)


class TestBuildPixelGraph:
    def test_minimal_two_pixel_image_has_terminals_and_one_shared_edge(self):
        image = [[10, 200]]
        similarity_fn = default_similarity(sigma=30.0)
        fg_fn, bg_fn = default_likelihoods(foreground_value=10, background_value=200)

        network, fg_terminal, bg_terminal = build_pixel_graph(image, similarity_fn, fg_fn, bg_fn)

        assert (0, 0) in network.nodes
        assert (1, 0) in network.nodes
        assert fg_terminal in network.nodes
        assert bg_terminal in network.nodes
        # 2 t-links per pixel (4 total) + 2 n-links between the pair (one
        # each direction) = 6 forward edges, 12 edge list entries total
        assert len(network.edges) == 12


class TestSegmentCleanlySeparableImage:
    def test_dark_left_light_right_splits_down_the_middle(self):
        # a clean 4x2 image, left half dark (foreground), right half
        # light (background), strongly separable so the cut should land
        # exactly on the boundary column regardless of the smoothness
        # term's exact pull
        image = [
            [10, 10, 200, 200],
            [10, 10, 200, 200],
        ]
        similarity_fn = default_similarity(sigma=15.0)
        fg_fn, bg_fn = default_likelihoods(foreground_value=10, background_value=200, scale=5.0)

        mask = segment_image(image, similarity_fn, fg_fn, bg_fn)

        assert len(mask) == 2
        assert len(mask[0]) == 4
        for row in mask:
            assert row[0] == 255 and row[1] == 255
            assert row[2] == 0 and row[3] == 0

    def test_uniform_image_segments_entirely_one_way(self):
        # every pixel identical, nothing to split, the whole image
        # should land on whichever label its own likelihood favors, here
        # foreground since the pixel value exactly matches foreground_value
        image = [[50, 50], [50, 50]]
        similarity_fn = default_similarity(sigma=10.0)
        fg_fn, bg_fn = default_likelihoods(foreground_value=50, background_value=200)

        mask = segment_image(image, similarity_fn, fg_fn, bg_fn)
        assert all(value == 255 for row in mask for value in row)
