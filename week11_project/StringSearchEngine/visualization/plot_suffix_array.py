"""
plot_suffix_array.py
------------------------
Part 2.1's visualization task. Two aligned bar charts, the suffix array
itself (which text position each rank in sorted order points to) on top,
and the LCP array (how much each suffix shares with its neighbor in
sorted order) underneath, sharing the same x axis (rank in the sorted
order) so a tall spike in the LCP chart lines up directly under the two
suffix array bars it's describing the overlap between.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_suffix_array(text, sa, lcp=None, out_path=None, max_display=80, title=None):
    """
    Plots `sa` (and `lcp`, if given) as bar charts. Texts longer than
    `max_display` suffixes get truncated to the first `max_display`
    ranks, the plot is meant to make the structure legible, not to
    render tens of thousands of bars.
    """
    n = len(sa)
    display_n = min(n, max_display)
    ranks = list(range(display_n))
    sa_display = sa[:display_n]

    if lcp is not None:
        fig, (ax_sa, ax_lcp) = plt.subplots(2, 1, figsize=(max(8, display_n * 0.15), 6), sharex=True)
    else:
        fig, ax_sa = plt.subplots(1, 1, figsize=(max(8, display_n * 0.15), 4))
        ax_lcp = None

    ax_sa.bar(ranks, sa_display, color="steelblue")
    ax_sa.set_ylabel("suffix start index")
    truncated_note = f" (first {display_n} of {n})" if display_n < n else f" ({n} suffixes)"
    ax_sa.set_title(title or f"Suffix array{truncated_note}")

    if ax_lcp is not None:
        lcp_display = lcp[:display_n]
        ax_lcp.bar(ranks, lcp_display, color="darkorange")
        ax_lcp.set_ylabel("LCP with previous suffix")
        ax_lcp.set_xlabel("rank in sorted suffix order")
    else:
        ax_sa.set_xlabel("rank in sorted suffix order")

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        return None
    return fig
