"""
draw_suffix_tree.py
-----------------------
Part 2.2's visualization task. Draws the compressed suffix trie built
by core/suffix_tree.py directly with matplotlib, a custom layout rather
than reaching for Graphviz, every leaf gets an evenly spaced x position
(assigned by a left to right DFS), every internal node sits at the
average x of its children, and depth sets the y position, which is
enough structure to make a small trie's branching and edge labels
genuinely readable without an external Graphviz dependency this project
doesn't otherwise need.

This is meant for small, illustrative texts, a demo string, a short
gene fragment, a sentence. The suffix tree built for the benchmark's
larger inputs has thousands of nodes and would just render as an
unreadable smear, `max_nodes` exists specifically to catch that mistake
early with a clear error instead of producing a plot nobody can read.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.suffix_tree import count_nodes


def _compute_layout(root):
    positions = {}
    next_leaf_x = [0]

    def dfs(node, depth):
        if not node.children:
            x = next_leaf_x[0]
            next_leaf_x[0] += 1
            positions[id(node)] = (x, -depth)
            return x
        child_xs = [dfs(child, depth + 1) for _, _, child in node.children.values()]
        x = sum(child_xs) / len(child_xs)
        positions[id(node)] = (x, -depth)
        return x

    dfs(root, 0)
    return positions


def draw_suffix_tree(text, root, out_path=None, max_label_length=12, max_nodes=400, title=None):
    """
    Draws the compressed trie rooted at `root` over `text`. Raises
    ValueError if the tree has more than `max_nodes` nodes, this
    function is for small demonstration texts, not the benchmark's
    stress test inputs, pass a shorter text or a higher max_nodes if
    you're sure you want a bigger plot.
    """
    total_nodes = count_nodes(root)
    if total_nodes > max_nodes:
        raise ValueError(
            f"this tree has {total_nodes} nodes, over the max_nodes={max_nodes} limit, "
            f"draw_suffix_tree is meant for small demonstration texts, pass a shorter text "
            f"or raise max_nodes if a bigger plot is really what you want"
        )

    positions = _compute_layout(root)
    fig, ax = plt.subplots(figsize=(max(8, total_nodes * 0.35), max(5, total_nodes * 0.2)))

    def draw(node):
        x0, y0 = positions[id(node)]
        for edge_start, edge_end, child in node.children.values():
            x1, y1 = positions[id(child)]
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=1, zorder=1)
            label = text[edge_start:edge_end]
            if len(label) > max_label_length:
                label = label[:max_label_length] + "…"
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=3,
            )
            draw(child)

        is_leaf = not node.children
        color = "tomato" if node.end_positions else ("steelblue" if is_leaf else "lightgray")
        ax.scatter([x0], [y0], s=45 if is_leaf else 25, color=color, zorder=2)
        if node.end_positions:
            ax.annotate(
                str(min(node.end_positions)), (x0, y0), textcoords="offset points",
                xytext=(0, -11), fontsize=6, ha="center", color="darkred",
            )

    draw(root)
    ax.set_title(title or f"Compressed suffix trie ({len(text)} character text, {total_nodes} nodes)")
    ax.axis("off")
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        return None
    return fig
