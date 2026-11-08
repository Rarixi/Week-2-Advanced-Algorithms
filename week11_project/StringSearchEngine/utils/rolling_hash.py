"""
rolling_hash.py
-------------------
Part of the string search project. The hash function Rabin-Karp runs on,
pulled out into its own module since suffix construction code and the
benchmark's collision analysis both want to reuse it without importing
rabin_karp.py itself.

The idea is a polynomial hash, treat a length m window of text as a base-b
number, hash(s[i..i+m-1]) = s[i] * b^(m-1) + s[i+1] * b^(m-2) + ... +
s[i+m-1] * b^0, taken mod a prime. That's an O(m) computation on its own,
but sliding the window by one character only needs O(1) work, drop the
leading term's contribution (s[i] * b^(m-1)), multiply what's left by b to
shift every remaining term up one power, and add the new trailing
character. That rolling update is the entire point of a rolling hash,
without it Rabin-Karp would just be a slower way to do what KMP already
does in O(n + m).

The modulus here defaults to 2^61 - 1, a Mersenne prime comfortably inside
a 64 bit range, which keeps collision probability low (roughly 1 in the
modulus for two unrelated strings) without needing modular arithmetic
libraries beyond Python's own big integers. Real collisions still happen
on large inputs purely from the pigeonhole principle, rabin_karp.py always
double checks a hash match against the actual substring before reporting
it, and the benchmark suite's collision_analysis.png plot exists
specifically to show how often that double check catches a false
positive.
"""


class RollingHash:
    """
    A single rolling hash family, parameterized by a base and a modulus.
    Two RollingHash instances with different (base, modulus) pairs can be
    run side by side as a cheap way to cut collision probability further
    (rabin_karp.py's `double_hash=True` option does exactly that), since
    two independent hash functions colliding on the same pair of strings
    is far less likely than either one colliding alone.
    """

    DEFAULT_BASE = 257
    DEFAULT_MODULUS = (1 << 61) - 1  # a Mersenne prime

    def __init__(self, base=None, modulus=None):
        self.base = base if base is not None else self.DEFAULT_BASE
        self.modulus = modulus if modulus is not None else self.DEFAULT_MODULUS
        self._pow_cache = [1]

    def _power(self, exponent):
        """base^exponent mod modulus, memoized since the same window
        length gets queried over and over while scanning a text."""
        while len(self._pow_cache) <= exponent:
            self._pow_cache.append((self._pow_cache[-1] * self.base) % self.modulus)
        return self._pow_cache[exponent]

    def hash_of(self, s):
        """Direct O(len(s)) hash of a whole string, used once up front to
        hash the pattern and to seed the first window of the text."""
        h = 0
        for ch in s:
            h = (h * self.base + ord(ch)) % self.modulus
        return h

    def roll(self, old_hash, old_char, new_char, window_length):
        """Given the hash of a window and the character leaving on the
        left and the character entering on the right, returns the hash of
        the window shifted one position right, in O(1)."""
        leading_term = (ord(old_char) * self._power(window_length - 1)) % self.modulus
        rolled = (old_hash - leading_term) % self.modulus
        rolled = (rolled * self.base) % self.modulus
        rolled = (rolled + ord(new_char)) % self.modulus
        return rolled

    def window_hashes(self, text, window_length):
        """Yields (start_index, hash) for every window of `window_length`
        in `text`, left to right, computing the first directly and every
        subsequent one with a single roll, O(n) total rather than O(n*m)
        from re-hashing every window from scratch."""
        n = len(text)
        if window_length <= 0 or window_length > n:
            return
        current = self.hash_of(text[:window_length])
        yield 0, current
        for i in range(1, n - window_length + 1):
            current = self.roll(current, text[i - 1], text[i + window_length - 1], window_length)
            yield i, current
