"""
cache_sim.py
---------------
A simulated cache for Part 3 of the AdvancedDataStructures project.

Python can't observe hardware cache misses directly, and its interpreter
overhead hides most memory effects in timings. So the cache-oblivious
experiments replay the exact sequence of memory addresses an algorithm
touches through a model of the *ideal cache* from Frigo, Leiserson,
Prokop & Ramachandran (1999), the model the cache-oblivious theory is
stated in:

  * memory is split into blocks (cache lines) of B bytes;
  * the cache holds M / B blocks, fully associative;
  * on a miss the block is loaded, evicting the least recently used
    block (LRU, which the paper shows is within a constant factor of the
    optimal replacement policy).

A miss is counted whenever an address falls in a block that isn't
resident. Nothing else (prefetching, associativity conflicts, TLB) is
modelled; the point is to count block transfers, which is what the
O(...) bounds in the report are about.

`CacheHierarchy` stacks several caches (e.g. an L1 of 32 KiB and an L2 of
256 KiB, both with 64-byte lines) so one replay reports misses at every
level: the "memory hierarchy behaviour" the assignment asks about.
"""

from collections import OrderedDict


class IdealCache:
    """
        cache = IdealCache(cache_bytes=32 * 1024, block_bytes=64)
        for addr in trace:
            cache.access(addr)
        cache.misses, cache.hits, cache.miss_rate
    """

    def __init__(self, cache_bytes=32 * 1024, block_bytes=64):
        if cache_bytes < block_bytes:
            raise ValueError("cache must hold at least one block")
        self.cache_bytes = cache_bytes
        self.block_bytes = block_bytes
        self.capacity = cache_bytes // block_bytes
        self._lines = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._last = None             # fast path: repeated access to the same block

    def access(self, addr):
        """Touch one byte address; returns True on a hit."""
        block = addr // self.block_bytes
        if block == self._last:
            self.hits += 1
            return True
        self._last = block
        lines = self._lines
        if block in lines:
            lines.move_to_end(block)
            self.hits += 1
            return True
        self.misses += 1
        lines[block] = None
        if len(lines) > self.capacity:
            lines.popitem(last=False)
        return False

    def run(self, trace):
        """Replay an iterable of addresses; returns the number of misses it caused."""
        before = self.misses
        access = self.access
        for addr in trace:
            access(addr)
        return self.misses - before

    def flush(self):
        self._lines.clear()
        self._last = None

    def reset_counters(self):
        self.hits = 0
        self.misses = 0

    @property
    def accesses(self):
        return self.hits + self.misses

    @property
    def miss_rate(self):
        return self.misses / self.accesses if self.accesses else 0.0


class CacheHierarchy:
    """
    Inclusive multi-level cache: every access goes to L1; an L1 miss is
    an access to L2; and so on. levels = [(name, cache_bytes, block_bytes), ...].
    """

    def __init__(self, levels=(("L1", 32 * 1024, 64), ("L2", 256 * 1024, 64))):
        self.names = [name for name, _, _ in levels]
        self.caches = [IdealCache(size, block) for _, size, block in levels]

    def access(self, addr):
        for cache in self.caches:
            if cache.access(addr):
                return

    def run(self, trace):
        for addr in trace:
            self.access(addr)
        return self.report()

    def report(self):
        return {name: c.misses for name, c in zip(self.names, self.caches)}


def simulate(trace, cache_bytes=32 * 1024, block_bytes=64):
    """Convenience: misses of one trace on a fresh ideal cache."""
    cache = IdealCache(cache_bytes, block_bytes)
    cache.run(trace)
    return cache.misses
