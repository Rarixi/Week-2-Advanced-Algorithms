"""
image_seg.py
---------------
Part 2 of the flow network project. Image segmentation via graph cuts,
the classic Boykov-Jolly construction. Every pixel is a node. Two
terminal nodes represent the two labels, a foreground terminal (playing
the role of "source") and a background terminal (playing the role of
"sink"). Every pixel gets a t-link from the foreground terminal, weighted
by how strongly that pixel's intensity matches the foreground model, and
a t-link to the background terminal, weighted by how strongly it matches
the background model. Every pair of 4-connected neighboring pixels also
gets a pair of n-links (one each direction, same weight, a similarity
score that's high when the two pixels look alike).

The key fact that makes this work, a minimum cut partitions every node
into exactly one of two sets, the side still reachable from the
foreground terminal in the residual graph after max-flow (call it S) and
everything else (T). A pixel's t-link to the foreground terminal only
gets cut if that pixel ends up in T, so ending up in T costs exactly that
pixel's foreground evidence. Symmetrically, a pixel's t-link to the
background terminal only gets cut if the pixel ends up in S, so ending
up in S costs its background evidence. Minimizing the cut therefore
means every pixel gets assigned whichever label costs less in isolation,
while a cut n-link between two differently labeled, visually similar
neighbors adds its own penalty too, which is exactly what keeps the
segmentation smooth instead of a speckled pixel by pixel decision. Since
min s-t cut and max-flow share the same value (the max-flow min-cut
theorem), this project solves the cut by running max-flow and reading
off which side of the residual graph each pixel ended up on.
"""

import math
from collections import deque

from core.edmonds_karp import edmonds_karp_max_flow
from core.network import Network

_FG_TERMINAL = "__fg_terminal__"
_BG_TERMINAL = "__bg_terminal__"


def build_pixel_graph(image, similarity_fn, foreground_likelihood_fn, background_likelihood_fn):
    """
    Builds the pixel grid graph described above from a 2d image, a list
    of rows, each row a list of pixel values. `similarity_fn(a, b)`
    scores how alike two neighboring pixel values are (used as every
    n-link's capacity, both directions), `foreground_likelihood_fn(v)`
    and `background_likelihood_fn(v)` score a single pixel's value
    against each label's model (used as that pixel's two t-link
    capacities). Returns (network, foreground_terminal, background_terminal).
    """
    height = len(image)
    width = len(image[0]) if height else 0

    network = Network()
    network.add_node(_FG_TERMINAL)
    network.add_node(_BG_TERMINAL)

    for y in range(height):
        for x in range(width):
            node = (x, y)
            value = image[y][x]
            network.add_edge(_FG_TERMINAL, node, foreground_likelihood_fn(value))
            network.add_edge(node, _BG_TERMINAL, background_likelihood_fn(value))

            if x + 1 < width:
                weight = similarity_fn(image[y][x], image[y][x + 1])
                network.add_edge(node, (x + 1, y), weight)
                network.add_edge((x + 1, y), node, weight)
            if y + 1 < height:
                weight = similarity_fn(image[y][x], image[y + 1][x])
                network.add_edge(node, (x, y + 1), weight)
                network.add_edge((x, y + 1), node, weight)

    return network, _FG_TERMINAL, _BG_TERMINAL


def _residual_reachable(network, source):
    """BFS over the post-max-flow residual graph, every node still
    reachable from `source` is the S side of the minimum cut."""
    seen = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for edge_index in network.edges_from(u):
            if network.residual_capacity(edge_index) <= 1e-12:
                continue
            v = network.edges[edge_index].v
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def segment_image(image, similarity_fn, foreground_likelihood_fn, background_likelihood_fn, solver=edmonds_karp_max_flow):
    """
    Runs the graph cut and returns a segmented mask, same shape as
    `image`, 255 for pixels on the foreground (source reachable) side of
    the minimum cut, 0 for background. `solver` defaults to
    edmonds_karp_max_flow, push_relabel_max_flow also works and is
    usually the better choice once an image has more than a handful of
    pixels, since the pixel grid graph gets large fast.
    """
    network, fg_terminal, bg_terminal = build_pixel_graph(
        image, similarity_fn, foreground_likelihood_fn, background_likelihood_fn
    )
    solver(network, fg_terminal, bg_terminal)
    foreground_side = _residual_reachable(network, fg_terminal)

    height = len(image)
    width = len(image[0]) if height else 0
    return [
        [255 if (x, y) in foreground_side else 0 for x in range(width)]
        for y in range(height)
    ]


def default_similarity(sigma=30.0):
    """A simple intensity-distance similarity, exp(-(a - b)^2 / (2 *
    sigma^2)), close to 1 for nearly identical pixel values and close to
    0 for very different ones, the standard smoothness weight used in
    graph cut segmentation demos."""

    def similarity_fn(a, b):
        diff = a - b
        return math.exp(-(diff * diff) / (2.0 * sigma * sigma))

    return similarity_fn


def default_likelihoods(foreground_value, background_value, scale=1.0):
    """
    Builds a (foreground_likelihood_fn, background_likelihood_fn) pair
    from two representative intensity values, foreground_value typical
    of the object you want segmented out and background_value typical
    of everything else. A pixel's foreground likelihood is how close it
    sits to foreground_value relative to how far it sits from
    background_value (and vice versa for background), so a pixel
    squarely between the two seeds gets a weak, easily overridden pull
    toward either terminal, while one clearly matching a seed gets a
    strong one.
    """

    def foreground_likelihood_fn(value):
        return scale / (1.0 + abs(value - foreground_value))

    def background_likelihood_fn(value):
        return scale / (1.0 + abs(value - background_value))

    return foreground_likelihood_fn, background_likelihood_fn
