"""
hash_table.py
-------------
Two hash table implementations, both supporting insert(key, value),
get(key), and delete(key):

    - ChainingHashTable: separate chaining, each bucket holds a linked
      list of (key, value) nodes. Collisions just get appended to the
      list for that bucket.
    - OpenAddressingHashTable: linear probing, everything lives in one
      flat array. On a collision, we scan forward for the next open slot.

Both track their load factor and automatically rehash (grow + rebuild)
once it crosses a threshold, so performance doesn't quietly degrade as
more items get added.

Average-case vs. worst-case complexity (n = number of stored items):
    insert() / get() / delete()
        Average case: O(1) -- assuming a decent hash function spreads
        keys evenly across buckets/slots, each bucket only ever holds
        a small constant number of items, so there's nothing to scan.

        Worst case: O(n) -- if every key happens to hash to the same
        bucket (chaining) or the same probe sequence (open addressing),
        you degrade into scanning a list of up to n items. This is why
        keeping the load factor low and rehashing matters: it keeps the
        "average case" assumption realistic instead of theoretical.
"""

from typing import Any, List, Optional  # import the typing tools this file uses for type hints


# ===========================================================================
# Separate chaining
# ===========================================================================

class _ChainNode:  # one link in a bucket's chain
    """A single key/value node in a bucket's linked list."""

    def __init__(self, key: Any, value: Any, next_node: Optional["_ChainNode"] = None) -> None:  # constructor
        self.key: Any = key  # the key this node holds
        self.value: Any = value  # the value associated with that key
        self.next: Optional["_ChainNode"] = next_node  # pointer to the next node in this bucket's chain


class ChainingHashTable:  # hash table using separate chaining for collisions
    """Hash table where each bucket is a linked list of (key, value)
    pairs. Multiple keys hashing to the same bucket just get appended
    to that bucket's chain instead of overwriting anything."""

    def __init__(self, capacity: int = 8, load_factor_threshold: float = 0.75) -> None:  # constructor
        self._capacity: int = capacity  # number of buckets in the array
        self._buckets: List[Optional[_ChainNode]] = [None] * capacity  # each slot is the head of a chain, or None
        self._size: int = 0  # total number of key/value pairs stored across all buckets
        self._load_factor_threshold: float = load_factor_threshold  # grow once size/capacity crosses this

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, key: Any, value: Any) -> None:  # add or update a key/value pair
        """Insert or update `key` with `value`. O(1) average, O(n) worst case."""
        index = self._hash(key)  # figure out which bucket this key belongs in
        node = self._buckets[index]  # start at the head of that bucket's chain
        while node is not None:  # walk the chain looking for an existing copy of this key
            if node.key == key:  # key already exists in this bucket
                node.value = value  # just update its value in place
                return  # done, no need to grow anything
            node = node.next  # keep walking the chain
        self._buckets[index] = _ChainNode(key, value, next_node=self._buckets[index])  # new node becomes the new head
        self._size += 1  # one more item is now stored overall
        if self.load_factor() > self._load_factor_threshold:  # check if we've grown too crowded
            self._rehash()  # grow the table and redistribute everything

    def get(self, key: Any) -> Any:  # look up the value for a key
        """Return the value stored for `key`. O(1) average, O(n) worst case.
        Raises KeyError if the key isn't present."""
        index = self._hash(key)  # find which bucket to look in
        node = self._buckets[index]  # start at the head of that bucket's chain
        while node is not None:  # walk the chain
            if node.key == key:  # found it
                return node.value  # hand back the stored value
            node = node.next  # keep walking
        raise KeyError(key)  # walked the whole chain, key isn't here

    def delete(self, key: Any) -> None:  # remove a key/value pair
        """Delete `key` from the table. O(1) average, O(n) worst case.
        Raises KeyError if the key isn't present."""
        index = self._hash(key)  # find which bucket to look in
        node = self._buckets[index]  # start at the head of that bucket's chain
        previous: Optional[_ChainNode] = None  # track the node before `node`, needed to unlink it
        while node is not None:  # walk the chain
            if node.key == key:  # found the node to remove
                if previous is None:  # it's the head of the chain
                    self._buckets[index] = node.next  # bucket now points straight past it
                else:  # it's somewhere in the middle/end of the chain
                    previous.next = node.next  # skip over it by relinking around it
                self._size -= 1  # one fewer item stored overall
                return  # done
            previous = node  # remember this node as "previous" for the next iteration
            node = node.next  # keep walking
        raise KeyError(key)  # walked the whole chain, key isn't here

    def load_factor(self) -> float:  # how full the table currently is
        """size / capacity. Higher means more collisions on average."""
        return self._size / self._capacity  # simple ratio, this is the standard definition

    def max_chain_length(self) -> int:  # diagnostic: longest chain currently in the table
        """Return the length of the longest chain currently in the
        table. Useful for illustrating the worst-case scenario: if this
        number is close to `len(self)`, nearly everything hashed into
        one bucket and lookups there are effectively O(n)."""
        longest = 0  # track the longest chain seen so far
        for head in self._buckets:  # check every bucket
            length = 0  # length of the chain in this particular bucket
            node = head  # start at this bucket's head
            while node is not None:  # walk the whole chain
                length += 1  # count this node
                node = node.next  # move to the next one
            longest = max(longest, length)  # keep the largest chain length seen
        return longest  # hand back the worst bucket's chain length

    def __len__(self) -> int:  # lets Python's built-in len() work on this object
        return self._size  # total items stored

    def __contains__(self, key: Any) -> bool:  # lets Python's `in` operator work on this object
        try:  # reuse get()'s lookup logic
            self.get(key)  # will raise KeyError if not found
            return True  # found it without raising
        except KeyError:  # get() couldn't find it
            return False  # so it's not in the table

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash(self, key: Any) -> int:  # map a key to a bucket index
        return hash(key) % self._capacity  # Python's % always returns a non-negative result here since capacity > 0

    def _rehash(self) -> None:  # grow the table and redistribute every existing item
        """Double the number of buckets and reinsert every existing
        item. O(n), but this only happens occasionally (amortized cost
        stays low), the same idea as a dynamic array's resizing."""
        old_buckets = self._buckets  # keep a reference to the old (smaller) bucket array
        self._capacity *= 2  # double the number of buckets
        self._buckets = [None] * self._capacity  # brand-new, empty, bigger bucket array
        self._size = 0  # will be recounted as we reinsert everything below
        for head in old_buckets:  # go through every old bucket
            node = head  # start at that bucket's head
            while node is not None:  # walk the whole chain
                self.insert(node.key, node.value)  # reinsert into the new, bigger table (recomputes its new bucket)
                node = node.next  # move to the next node in the OLD chain


# ===========================================================================
# Open addressing (linear probing)
# ===========================================================================

class OpenAddressingHashTable:  # hash table using linear probing for collisions
    """Hash table where every key/value pair lives directly in a flat
    array. On a collision, we scan forward slot by slot until we find
    an open spot (linear probing).

    Deleted slots are marked with a "tombstone" instead of being reset
    to genuinely empty -- otherwise a later lookup could incorrectly
    stop probing early and report "not found" for a key that actually
    exists further down the probe sequence, past the deleted slot.
    """

    def __init__(self, capacity: int = 8, load_factor_threshold: float = 0.5) -> None:  # constructor
        self._capacity: int = capacity  # total number of slots in the array
        self._keys: List[Any] = [None] * capacity  # None means "never used" unless also marked deleted
        self._values: List[Any] = [None] * capacity  # values parallel to _keys by index
        self._deleted: List[bool] = [False] * capacity  # tombstone flags, True means "used to hold something"
        self._size: int = 0  # number of live (non-deleted) key/value pairs currently stored
        self._tombstone_count: int = 0  # number of deleted-but-not-yet-cleared slots
        # open addressing needs a lower threshold than chaining, since a
        # nearly-full array causes long probe sequences even before it's
        # technically "full" -- 0.5 keeps probing short on average
        self._load_factor_threshold: float = load_factor_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, key: Any, value: Any) -> None:  # add or update a key/value pair
        """Insert or update `key` with `value`. O(1) average, O(n) worst case."""
        if (self._size + self._tombstone_count) / self._capacity > self._load_factor_threshold:
            self._rehash()  # grow (and clear out tombstones) before inserting

        index = self._hash(key)  # ideal starting slot for this key
        first_tombstone = -1  # remember the first tombstone we pass, in case we need to insert there instead

        for offset in range(self._capacity):  # never probe more than one full lap around the array
            probe = (index + offset) % self._capacity  # wrap around using modulo

            if self._keys[probe] is None and not self._deleted[probe]:  # genuinely empty slot, never used
                target = first_tombstone if first_tombstone != -1 else probe  # prefer reusing an earlier tombstone
                self._keys[target] = key  # place the key
                self._values[target] = value  # place the value
                if self._deleted[target]:  # if we're reusing a tombstone slot
                    self._tombstone_count -= 1  # one fewer tombstone now
                self._deleted[target] = False  # this slot is live now, not a tombstone
                self._size += 1  # one more item stored
                return  # done

            if self._deleted[probe] and first_tombstone == -1:  # first tombstone seen along this probe sequence
                first_tombstone = probe  # remember it as a candidate insert location

            elif not self._deleted[probe] and self._keys[probe] == key:  # key already exists here
                self._values[probe] = value  # just update the value
                return  # done, no growth needed

        self._rehash()
        self.insert(key, value)

    def get(self, key: Any) -> Any:  # look up the value for a key
        """Return the value stored for `key`. O(1) average, O(n) worst case.
        Raises KeyError if the key isn't present."""
        index = self._hash(key)  # ideal starting slot for this key
        for offset in range(self._capacity):  # never probe more than one full lap
            probe = (index + offset) % self._capacity  # wrap around using modulo
            if self._keys[probe] is None and not self._deleted[probe]:  # hit a genuinely empty slot
                raise KeyError(key)  # key can't be further along, probing would have placed it before here
            if not self._deleted[probe] and self._keys[probe] == key:  # found a live match
                return self._values[probe]  # hand back its value
        raise KeyError(key)  # probed the entire table, never found it

    def delete(self, key: Any) -> None:  # remove a key/value pair
        """Delete `key` from the table. O(1) average, O(n) worst case.
        Raises KeyError if the key isn't present."""
        index = self._hash(key)  # ideal starting slot for this key
        for offset in range(self._capacity):  # never probe more than one full lap
            probe = (index + offset) % self._capacity  # wrap around using modulo
            if self._keys[probe] is None and not self._deleted[probe]:  # hit a genuinely empty slot
                raise KeyError(key)  # key isn't in the table
            if not self._deleted[probe] and self._keys[probe] == key:  # found the live key to remove
                self._keys[probe] = None  # clear the key
                self._values[probe] = None  # clear the value
                self._deleted[probe] = True  # leave a tombstone so later probes don't stop early
                self._size -= 1  # one fewer item stored
                self._tombstone_count += 1  # one more tombstone sitting in the array
                return  # done
        raise KeyError(key)  # probed the entire table, never found it

    def load_factor(self) -> float:  # how full the table currently is (live items only)
        """size / capacity. Higher means longer average probe sequences."""
        return self._size / self._capacity  # standard definition, based on live items

    def __len__(self) -> int:  # lets Python's built-in len() work on this object
        return self._size  # number of live items

    def __contains__(self, key: Any) -> bool:  # lets Python's `in` operator work on this object
        try:  # reuse get()'s lookup logic
            self.get(key)  # will raise KeyError if not found
            return True  # found it without raising
        except KeyError:  # get() couldn't find it
            return False  # so it's not in the table

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash(self, key: Any) -> int:  # map a key to a starting slot index
        return hash(key) % self._capacity  # Python's % always returns a non-negative result here since capacity > 0

    def _rehash(self) -> None:  # grow the table, clear tombstones, and reinsert every live item
        """Double the capacity and reinsert every live key. This also
        clears out all tombstones, which is important: tombstones don't
        shrink on their own and would otherwise make probe sequences
        keep getting longer over time even if `size` stays constant."""
        old_keys = self._keys  # keep references to the old arrays
        old_values = self._values
        old_deleted = self._deleted
        old_capacity = self._capacity

        self._capacity *= 2  # double the number of slots
        self._keys = [None] * self._capacity  # brand-new, empty key array
        self._values = [None] * self._capacity  # brand-new, empty value array
        self._deleted = [False] * self._capacity  # brand-new, no tombstones yet
        self._size = 0  # will be recounted as we reinsert below
        self._tombstone_count = 0  # tombstones are gone after a rehash

        for i in range(old_capacity):  # walk every slot in the old table
            if old_keys[i] is not None and not old_deleted[i]:  # only reinsert live entries, skip tombstones/empties
                self.insert(old_keys[i], old_values[i])  # reinsert into the new, bigger table
