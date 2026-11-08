"""
suffix_array.py
-------------------
Part 2.1 of the string search project. A suffix array is just every
suffix of a text, sorted lexicographically and stored as the starting
index of each suffix rather than the suffix string itself (storing the
actual substrings would be O(n^2) space, since the suffixes themselves
total that many characters). Once it's built, every occurrence of a
pattern in the text corresponds to a contiguous run of suffixes in the
sorted order, exactly the ones that start with that pattern, which is
what makes binary search over the array a valid way to find them in
O(m log n) instead of scanning the text.

Construction here uses prefix doubling. Start by ranking every suffix by
its first character alone (lots of ties). Then repeatedly double a
window size k, re-rank every suffix by the pair (its current rank, the
current rank of the suffix starting k positions later), and re-sort.
Each round that pair fully determines the correct relative order of
every suffix over a 2k character window using only the previous round's
O(n) ranks, rather than re-comparing 2k characters from scratch, so the
window length doubles every round while the work per round stays
O(n log n) (the cost of one sort), and the whole thing finishes once
every suffix has a distinct rank, in at most log2(n) rounds, O(n log n)
overall (true O(n) construction exists, Kärkkäinen-Sanders among them,
the assignment calls it out explicitly as optional, prefix doubling is
the one actually built here).

Once the array exists, Kasai's algorithm builds the LCP (longest common
prefix) array alongside it in a single O(n) pass, LCP[i] is how many
characters suffix sa[i] and suffix sa[i-1] share as a common prefix,
which is what the suffix tree would have told you directly as shared
path length, the LCP array is the suffix array's version of that same
information.
"""


def build_suffix_array(text):
    """
    Builds the suffix array of `text` via prefix doubling, O(n log n).
    Returns a list of the n suffix start indices, sorted so that
    text[sa[0]:] < text[sa[1]:] < ... lexicographically.
    """
    n = len(text)
    if n == 0:
        return []

    sa = list(range(n))
    rank = [ord(c) for c in text]
    k = 1

    while True:
        def key(i, _rank=rank, _k=k, _n=n):
            return (_rank[i], _rank[i + _k] if i + _k < _n else -1)

        sa.sort(key=lambda i: key(i))

        new_rank = [0] * n
        new_rank[sa[0]] = 0
        for i in range(1, n):
            new_rank[sa[i]] = new_rank[sa[i - 1]] + (1 if key(sa[i]) > key(sa[i - 1]) else 0)
        rank = new_rank

        if rank[sa[-1]] == n - 1:
            break
        k *= 2
        if k > n:
            # safety net, should be unreachable since distinct ranks are
            # always achieved by the time the window covers the whole
            # string, but better to stop than loop forever on a bug
            break

    return sa


def build_lcp_array(text, sa):
    """
    Kasai's algorithm. Given `text` and its suffix array `sa`, returns
    the LCP array in the same O(n) pass, lcp[i] = the length of the
    longest common prefix between suffix sa[i] and suffix sa[i-1]
    (lcp[0] is always 0 by convention, there's no predecessor).

    The trick that keeps this O(n) rather than O(n^2) in the worst case,
    walking the text positions in original order (not sorted order)
    while tracking the previous match length k, the LCP of consecutive
    suffixes in rank order can only drop by at most how much the
    previous one dropped, so k only ever needs to decrease by one
    between positions instead of being recomputed from zero.
    """
    n = len(text)
    if n == 0:
        return []

    rank = [0] * n
    for i, s in enumerate(sa):
        rank[s] = i

    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + k < n and j + k < n and text[i + k] == text[j + k]:
                k += 1
            lcp[rank[i]] = k
            if k > 0:
                k -= 1
        else:
            k = 0

    return lcp


def sa_search(text, sa, pattern):
    """
    Returns every start index of `pattern` in `text`, using two binary
    searches over the suffix array to find the contiguous band of
    suffixes that start with `pattern` (the lower bound, the first
    suffix not lexicographically less than `pattern`, and the upper
    bound, the first suffix strictly greater than it), O(m log n).
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        raise ValueError("pattern must be non-empty")
    if m > n:
        return []

    def prefix_at(idx):
        start = sa[idx]
        return text[start:start + m]

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_at(mid) < pattern:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    lo, hi = left, n
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_at(mid) <= pattern:
            lo = mid + 1
        else:
            hi = mid
    right = lo

    return sorted(sa[left:right])


def longest_repeated_substring_via_lcp(text, sa, lcp):
    """
    The longest substring that occurs at least twice in `text`, read
    straight off the LCP array, it's just the text at the suffix array
    position with the largest LCP value, for exactly that many
    characters. Used mainly to cross check against suffix_tree.py's own
    longest_repeated_substring, two completely different data
    structures should always agree on this answer.
    """
    if not lcp:
        return ""
    best_idx = max(range(len(lcp)), key=lambda i: lcp[i])
    best_len = lcp[best_idx]
    if best_len == 0:
        return ""
    start = sa[best_idx]
    return text[start:start + best_len]
