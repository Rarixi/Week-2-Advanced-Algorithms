"""
plagiarism.py
----------------
Part 4.1 of the string search project. Document similarity built from
two different signals, each playing to the strength of a different core
algorithm.

The first signal is overlapping phrases (shingles), every fixed length
substring of one document, checked for whether it also shows up
somewhere in the other. This is exactly Rabin-Karp's best case, one text
being searched for hundreds or thousands of fixed length patterns at
once, batched into a single pass with rabin_karp_search_multi rather
than hundreds of separate O(n) scans. It's also the technique real
plagiarism detectors actually use (shingling), two documents that share
a lot of identical phrase length chunks are very likely to share real
copied text, even if the surrounding wording was lightly edited.

The second signal is the single longest substring the two documents
share outright, found with a generalized suffix array, concatenate
doc_a, a sentinel character guaranteed not to appear in either document,
and doc_b, build one suffix array and LCP array over the combined text,
and scan for the largest LCP value between two adjacent suffixes that
come from different sides of the sentinel. Any such pair shares a common
prefix that is, by construction, a substring of both original documents,
and the largest one found this way is provably the longest common
substring between them, lifted straight from a fact the suffix array
already proves about itself, no separate alignment algorithm needed.
"""

from core.rabin_karp import rabin_karp_search_multi
from core.suffix_array import build_lcp_array, build_suffix_array


def _pick_sentinel(doc_a, doc_b):
    """A control character (codes 1-31) guaranteed not to appear in
    either document, used to glue them together for the generalized
    suffix array without the two documents ever being able to
    "accidentally" match across the seam."""
    used = set(doc_a) | set(doc_b)
    for code in range(1, 32):
        candidate = chr(code)
        if candidate not in used:
            return candidate
    raise ValueError("could not find an unused sentinel character, documents use every control character")


def shingles(text, phrase_length=8):
    """Every fixed length substring of `text`, as {phrase: [positions]},
    using a dict rather than a list so identical phrases that repeat
    within the same document collapse into one entry with every
    position recorded."""
    positions = {}
    for i in range(len(text) - phrase_length + 1):
        phrase = text[i:i + phrase_length]
        positions.setdefault(phrase, []).append(i)
    return positions


def find_overlapping_phrases(doc_a, doc_b, phrase_length=8):
    """
    Every phrase_length-character phrase from `doc_a` that also occurs
    somewhere in `doc_b`, found in one Rabin-Karp batch pass over doc_b
    rather than one search per phrase. Returns a list of
    {"phrase":, "positions_in_a": [...], "positions_in_b": [...]},
    sorted by where the phrase first appears in doc_a.
    """
    phrases_a = shingles(doc_a, phrase_length)
    if not phrases_a:
        return []

    matches_in_b = rabin_karp_search_multi(doc_b, list(phrases_a.keys()))

    overlaps = [
        {"phrase": phrase, "positions_in_a": positions_a, "positions_in_b": matches_in_b[phrase]}
        for phrase, positions_a in phrases_a.items()
        if matches_in_b[phrase]
    ]
    overlaps.sort(key=lambda o: o["positions_in_a"][0])
    return overlaps


def longest_common_substring(doc_a, doc_b):
    """
    The longest substring occurring in both documents, found via a
    generalized suffix array over doc_a + sentinel + doc_b. Returns
    {"substring":, "position_in_a":, "position_in_b":, "length":},
    an empty substring and length 0 if the documents share nothing at
    all (not even a single character, only possible if they use
    disjoint alphabets).
    """
    if not doc_a or not doc_b:
        return {"substring": "", "position_in_a": None, "position_in_b": None, "length": 0}

    sentinel = _pick_sentinel(doc_a, doc_b)
    combined = doc_a + sentinel + doc_b
    boundary = len(doc_a)  # combined[boundary] is the sentinel itself

    sa = build_suffix_array(combined)
    lcp = build_lcp_array(combined, sa)

    best_len = 0
    best_a = best_b = None
    for idx in range(1, len(sa)):
        s1, s2 = sa[idx - 1], sa[idx]
        if (s1 < boundary) == (s2 < boundary):
            continue  # both suffixes from the same document, not a cross match
        if lcp[idx] > best_len:
            best_len = lcp[idx]
            best_a, best_b = (s1, s2) if s1 < boundary else (s2, s1)

    if best_len == 0:
        return {"substring": "", "position_in_a": None, "position_in_b": None, "length": 0}

    position_in_b = best_b - boundary - 1
    return {
        "substring": doc_a[best_a:best_a + best_len],
        "position_in_a": best_a,
        "position_in_b": position_in_b,
        "length": best_len,
    }


def similarity_score(doc_a, doc_b, phrase_length=8):
    """
    Combines both signals into one report. `score` is a simple average
    of two ratios, what fraction of doc_a's phrases also turned up in
    doc_b, and how much of the shorter document the single longest
    shared substring accounts for, each in [0, 1], so the combined score
    is too. It's a deliberately simple blend, not a tuned metric, the
    point of this project is demonstrating the two algorithms doing real
    work, not building a production plagiarism scorer.
    """
    overlaps = find_overlapping_phrases(doc_a, doc_b, phrase_length)
    distinct_phrases = shingles(doc_a, phrase_length)
    distinct_count = len(distinct_phrases)
    phrase_overlap_ratio = (len(overlaps) / distinct_count) if distinct_count else 0.0

    lcs = longest_common_substring(doc_a, doc_b)
    shorter_length = min(len(doc_a), len(doc_b))
    lcs_ratio = (lcs["length"] / shorter_length) if shorter_length else 0.0

    return {
        "score": (phrase_overlap_ratio + lcs_ratio) / 2.0,
        "phrase_overlap_ratio": phrase_overlap_ratio,
        "matched_phrase_count": len(overlaps),
        "distinct_phrase_count": distinct_count,
        "longest_common_substring": lcs,
    }


def highlight_matches(doc_a, doc_b, phrase_length=8, overlaps=None):
    """
    `doc_a` with every matched phrase wrapped in [[double brackets]],
    adjacent or overlapping matched ranges merged first so a long run of
    copied text reads as one highlighted block instead of a string of
    separately bracketed, overlapping phrase_length chunks.
    """
    if overlaps is None:
        overlaps = find_overlapping_phrases(doc_a, doc_b, phrase_length)

    ranges = sorted(
        (pos, pos + phrase_length)
        for overlap in overlaps
        for pos in overlap["positions_in_a"]
    )
    if not ranges:
        return doc_a

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    pieces = []
    cursor = 0
    for start, end in merged:
        pieces.append(doc_a[cursor:start])
        pieces.append("[[" + doc_a[start:end] + "]]")
        cursor = end
    pieces.append(doc_a[cursor:])
    return "".join(pieces)
