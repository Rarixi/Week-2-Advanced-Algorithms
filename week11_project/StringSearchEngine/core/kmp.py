"""
kmp.py
---------
Part 1.1 of the string search project. Knuth-Morris-Pratt, the
deterministic linear time pattern matcher every other algorithm in this
project gets measured against. The whole trick is the failure function
(also called the LPS table, longest proper prefix which is also a
suffix), computed once per pattern in O(m), which tells the matcher
exactly how far it can slide the pattern forward after a mismatch without
ever re-comparing a character it already knows matches.

Why that works: say the pattern has matched the text up through index
i - 1 and then mismatches at pattern index j. Naive matching would slide
the whole pattern one character to the right and start over from
scratch, throwing away everything it just learned. KMP instead asks, of
the j characters just matched, what's the longest prefix of the pattern
that's also a suffix of what matched. That prefix is guaranteed to still
line up with the text without any further comparison, since the matched
text segment already ends in exactly that string, so the matcher can
jump pattern index straight to lps[j - 1] and keep comparing from there,
never stepping the text pointer backward. That's what keeps the whole
scan at O(n) text comparisons on top of the O(m) table build, O(n + m)
total, regardless of how adversarial the input is.
"""


def compute_lps(pattern):
    """
    Builds the LPS (longest proper prefix which is also a suffix) table
    for `pattern`, lps[i] is the length of the longest string that is
    both a proper prefix of pattern[:i+1] and a proper suffix of it.
    This is itself just KMP matching the pattern against itself, same
    fallback idea, one character at a time, reusing the table being
    built so far to skip ahead on a mismatch instead of restarting the
    comparison from the beginning.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")

    m = len(pattern)
    lps = [0] * m
    length = 0  # length of the previous longest prefix-suffix
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            # fall back to the next shorter candidate prefix-suffix
            # instead of giving up and starting over at length 0
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1

    return lps


def kmp_search(text, pattern):
    """
    Returns every start index in `text` where `pattern` occurs, in order,
    in O(len(text) + len(pattern)) time. A pattern longer than the text
    can never occur, so that case short circuits to an empty list rather
    than doing any work.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")
    n, m = len(text), len(pattern)
    if m > n:
        return []

    lps = compute_lps(pattern)
    matches = []
    i = j = 0  # i walks the text, j walks the pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]  # look for the next, possibly overlapping, match
        elif j > 0:
            # mismatch with partial progress, fall back using the LPS
            # table instead of resetting j to 0 and re-sliding one at a
            # time, this is the whole point of the algorithm
            j = lps[j - 1]
        else:
            i += 1

    return matches


def kmp_search_with_trace(text, pattern):
    """
    Same result as kmp_search, plus a step by step trace of every
    comparison, useful for a visualization or report that wants to
    illustrate the mismatch to fallback behavior directly rather than
    just asserting it happens. Each trace entry is one of:
      {"op": "match",    "text_index": i, "pattern_index": j}
      {"op": "complete_match", "start": start_index}
      {"op": "mismatch_fallback", "text_index": i, "from_j": j, "to_j": new_j}
      {"op": "mismatch_advance", "text_index": i}
    `mismatch_fallback` is the event worth highlighting, the pattern
    pointer jumps from `from_j` to `to_j` using the LPS table instead of
    the text pointer ever moving backward.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")
    n, m = len(text), len(pattern)
    trace = []
    if m > n:
        return {"matches": [], "comparisons": 0, "trace": trace}

    lps = compute_lps(pattern)
    matches = []
    comparisons = 0
    i = j = 0

    while i < n:
        comparisons += 1
        if text[i] == pattern[j]:
            trace.append({"op": "match", "text_index": i, "pattern_index": j})
            i += 1
            j += 1
            if j == m:
                start = i - j
                matches.append(start)
                trace.append({"op": "complete_match", "start": start})
                j = lps[j - 1]
        elif j > 0:
            new_j = lps[j - 1]
            trace.append({"op": "mismatch_fallback", "text_index": i, "from_j": j, "to_j": new_j})
            j = new_j
        else:
            trace.append({"op": "mismatch_advance", "text_index": i})
            i += 1

    return {"matches": matches, "comparisons": comparisons, "trace": trace}
