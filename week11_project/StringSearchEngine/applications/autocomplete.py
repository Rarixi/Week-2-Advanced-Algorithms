"""
autocomplete.py
-------------------
Part 4.3 of the string search project, built as the "prefix-based
search" option the assignment offers as an alternative to a compressed
suffix tree. That's a deliberate choice, not just the easier path,
core/suffix_tree.py indexes every suffix of one single text, which is
the right structure when the question is "where does this substring
occur inside this one document", but autocomplete asks a genuinely
different question, "which of these many independent words or queries
start with this prefix", over a vocabulary of separate strings rather
than positions in one shared text. Reusing the suffix tree for that
would mean concatenating every vocabulary word together with separators
just to force it into a shape it wasn't built for. A plain prefix trie
over the vocabulary directly, one node per character, is simpler, is
exactly the structure autocomplete is usually taught with, and leaves
no ambiguity about what a "word boundary" even means.

Query frequency weighting is just an integer stored at whichever node
completes a word, bumped by a weight every time that word gets added or
searched again elsewhere, so a popular query naturally outranks a rare
one with the same prefix once top_k_completions sorts by it.
"""


class _Node:
    __slots__ = ("children", "frequency")

    def __init__(self):
        self.children = {}
        self.frequency = 0  # > 0 means a word ends exactly here


def insert(root, word, weight=1):
    """Adds `word` to the trie rooted at `root`, incrementing its
    frequency by `weight` (so inserting the same word again, or seeding
    it with a known popularity count, just accumulates)."""
    if not word:
        raise ValueError("word must be non-empty")
    node = root
    for ch in word:
        node = node.children.setdefault(ch, _Node())
    node.frequency += weight


def _collect_words(node, prefix):
    results = []
    if node.frequency > 0:
        results.append((prefix, node.frequency))
    for ch, child in node.children.items():
        results.extend(_collect_words(child, prefix + ch))
    return results


def prefix_query(root, prefix):
    """Every (word, frequency) in the trie that starts with `prefix`,
    in no particular order, descending straight to the prefix's node
    (O(len(prefix))) and then collecting every completed word in its
    subtree (O(number of matches))."""
    node = root
    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]
    return _collect_words(node, prefix)


def top_k_completions(root, prefix, k=5):
    """The `k` completions of `prefix` with the highest frequency,
    ties broken alphabetically so results are stable and reproducible
    rather than depending on dict insertion order."""
    matches = prefix_query(root, prefix)
    matches.sort(key=lambda word_freq: (-word_freq[1], word_freq[0]))
    return matches[:k]


class AutocompleteEngine:
    """A thin, stateful wrapper around the trie functions above, the
    shape an actual autocomplete feature would hold onto across many
    queries, add_query incrementally growing the vocabulary (and its
    frequency weights) as real queries come in, complete() answering
    the "what should I suggest" question that matters."""

    def __init__(self):
        self.root = _Node()

    def add_query(self, word, weight=1):
        insert(self.root, word, weight)

    def add_queries(self, queries):
        """Bulk loads `queries`, accepted as a dict of {word: weight},
        or an iterable of words (each defaulting to weight 1) or
        (word, weight) pairs, whichever is more convenient for the
        caller, both plagiarism.py style text corpora and a plain list
        of search logs show up in one of these two shapes."""
        if isinstance(queries, dict):
            for word, weight in queries.items():
                self.add_query(word, weight)
            return
        for item in queries:
            if isinstance(item, tuple):
                word, weight = item
            else:
                word, weight = item, 1
            self.add_query(word, weight)

    def complete(self, prefix, k=5):
        return top_k_completions(self.root, prefix, k)

    def all_matches(self, prefix):
        return prefix_query(self.root, prefix)


def build_autocomplete_engine(queries):
    """One shot convenience, builds an AutocompleteEngine and loads
    `queries` into it in the same call (see add_queries for the accepted
    shapes)."""
    engine = AutocompleteEngine()
    engine.add_queries(queries)
    return engine
