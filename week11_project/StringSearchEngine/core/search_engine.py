"""
search_engine.py
-------------------
Part 3 of the string search project. A single StringSearchEngine class
wrapping KMP, Rabin-Karp, the suffix array, and the suffix tree behind
one interface,

    engine = StringSearchEngine(text)
    engine.find("pattern", method="kmp")

so every application in this project (plagiarism, DNA search,
autocomplete) and the benchmark suite can swap which algorithm answers a
query without touching any other code.

The one piece of bookkeeping that actually matters here, beyond just
dispatching to the right function, is tracking preprocessing time
separately from search time per method. KMP and Rabin-Karp do their
"preprocessing" (the LPS table, the pattern hash) fresh on every call,
scaled to the pattern, not the text, so there's nothing to cache. The
suffix array and suffix tree are the opposite, their preprocessing scales
with the TEXT and only needs to happen once no matter how many patterns
get searched afterward, so this engine builds each one lazily on first
use and reuses it for every later call. That's exactly the tradeoff this
project's report calls out, SA and the suffix tree pay more upfront and
then answer repeated queries against the same text faster than
redoing a linear scan every time, which is also what recommend() below
is built to surface from the engine's own recorded history rather than
just asserting it.
"""

import time

from core.kmp import kmp_search
from core.rabin_karp import rabin_karp_search, rabin_karp_search_multi
from core.suffix_array import build_suffix_array, build_lcp_array, sa_search
from core.suffix_tree import build_suffix_trie, substring_positions

_METHODS = ("kmp", "rk", "sa", "tree")


class StringSearchEngine:
    def __init__(self, text):
        self.text = text
        self._sa = None
        self._lcp = None
        self._tree_root = None
        self._preprocessing_time = {}  # method -> seconds spent building its index, once
        self._search_times = {method: [] for method in _METHODS}
        self._match_counts = {method: [] for method in _METHODS}

    def _ensure_sa(self):
        if self._sa is None:
            start = time.perf_counter()
            self._sa = build_suffix_array(self.text)
            self._lcp = build_lcp_array(self.text, self._sa)
            self._preprocessing_time["sa"] = time.perf_counter() - start

    def _ensure_tree(self):
        if self._tree_root is None:
            start = time.perf_counter()
            self._tree_root = build_suffix_trie(self.text)
            self._preprocessing_time["tree"] = time.perf_counter() - start

    def find(self, pattern, method="kmp"):
        """
        Runs one pattern search with the chosen method ("kmp", "rk",
        "sa", or "tree") and returns a dict with the match positions,
        the match count, how long this particular call took, and (for
        sa/tree) the one-time preprocessing cost paid the first time
        this engine instance used that method.
        """
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}, expected one of {_METHODS}")

        start = time.perf_counter()
        if method == "kmp":
            positions = kmp_search(self.text, pattern)
        elif method == "rk":
            positions = rabin_karp_search(self.text, pattern)["matches"]
        elif method == "sa":
            self._ensure_sa()
            positions = sa_search(self.text, self._sa, pattern)
        else:  # "tree"
            self._ensure_tree()
            positions = substring_positions(self.text, self._tree_root, pattern)
        elapsed = time.perf_counter() - start

        self._search_times[method].append(elapsed)
        self._match_counts[method].append(len(positions))

        return {
            "method": method,
            "positions": positions,
            "count": len(positions),
            "search_time": elapsed,
            "preprocessing_time": self._preprocessing_time.get(method, 0.0),
        }

    def find_multi(self, patterns):
        """
        Batch matching convenience, always uses Rabin-Karp's multi
        pattern mode since that's the one method here actually built to
        amortize work across many patterns in a single pass rather than
        running find() once per pattern.
        """
        start = time.perf_counter()
        result = rabin_karp_search_multi(self.text, patterns)
        elapsed = time.perf_counter() - start
        self._search_times["rk"].append(elapsed)
        self._match_counts["rk"].append(sum(len(v) for v in result.values()))
        return result

    def stats(self):
        """
        Per method summary of everything this engine instance has done
        so far: number of calls, average search time per call,
        preprocessing time (0.0 if that method was never used, or for
        kmp/rk which never have any), and total matches returned.
        """
        summary = {}
        for method in _METHODS:
            times = self._search_times[method]
            counts = self._match_counts[method]
            summary[method] = {
                "calls": len(times),
                "average_search_time": (sum(times) / len(times)) if times else 0.0,
                "preprocessing_time": self._preprocessing_time.get(method, 0.0),
                "total_matches": sum(counts),
            }
        return summary

    def recommend(self):
        """
        A short, plain English recommendation built from this engine
        instance's own recorded history, not a canned string. Needs at
        least one call recorded for a method to say anything about it,
        falls back to general guidance for any method not yet used.
        """
        summary = self.stats()
        lines = []

        used = {m: s for m, s in summary.items() if s["calls"] > 0}
        if used:
            fastest_avg = min(used.items(), key=lambda kv: kv[1]["average_search_time"])
            lines.append(
                f"{fastest_avg[0]} had the lowest average search time so far "
                f"({fastest_avg[1]['average_search_time']:.6f}s over {fastest_avg[1]['calls']} call(s))."
            )

        for method in ("sa", "tree"):
            s = summary[method]
            if s["calls"] >= 2:
                amortized = s["preprocessing_time"] / s["calls"]
                lines.append(
                    f"{method} paid {s['preprocessing_time']:.6f}s of preprocessing once, "
                    f"now down to {amortized:.6f}s amortized per call across {s['calls']} queries, "
                    f"the more queries against this same text, the better that trade looks."
                )
            elif s["calls"] == 1:
                lines.append(
                    f"{method} just paid {s['preprocessing_time']:.6f}s of preprocessing on its first "
                    f"call, that cost is sunk now, a second query against this same text would be "
                    f"effectively free by comparison."
                )

        for method in ("kmp", "rk"):
            s = summary[method]
            if s["calls"] == 0:
                continue
            lines.append(
                f"{method} has no preprocessing to amortize, every one of its {s['calls']} call(s) "
                f"redid the full O(n) scan, fine for a one off search, repeated searches against the "
                f"same text are where sa/tree pull ahead instead."
            )

        if not lines:
            lines.append(
                "No searches recorded yet. As a starting point, kmp is a safe default with no setup "
                "cost, rk is strongest for batch matching many patterns at once, and sa/tree are worth "
                "their preprocessing cost once you're running many queries against this same text."
            )

        return " ".join(lines)
