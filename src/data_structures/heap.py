"""
heap.py
-------

insert(item, priority): Add item with given priority
extract_max(): Remove and return highest priority item
peek(): View highest priority without removing
is_empty(): Check if queue is empty

Binary heap implementations (array-backed, 0-indexed):
    - MinHeap: root is always the smallest element
    - MaxHeap: root is always the largest element
    - PriorityQueue: a queue where items come out in priority order,
      built on top of MinHeap

Index math for a node at position i:
    parent(i)      = (i - 1) // 2
    left_child(i)  = 2*i + 1
    right_child(i) = 2*i + 2

Complexity (n = number of elements currently in the heap):
    peek()          O(1)       -- root is always at index 0
    insert()        O(log n)   -- worst case, bounded by tree height (sift-up)
    extract_min()   O(log n)   -- worst case, bounded by tree height (sift-down)
    extract_max()   O(log n)   -- same as above, MaxHeap version
    heapify()       O(n)       -- Floyd's bottom-up build, NOT O(n log n)
    is_empty()      O(1)

Why heapify() is O(n) and not O(n log n):
    Naively inserting n items one at a time costs O(n log n) total,
    since each insert costs O(log n) and it happens n times.

    The build-heap approach instead starts from the last non-leaf
    node and sifts DOWN, working from the bottom of the tree up
    toward the root. Leaves are skipped entirely since a single
    element is already a valid heap on its own. Most nodes live
    near the bottom of the tree and only sift a short distance;
    only a few nodes near the root sift far. Summed across the
    whole tree, this telescopes down to O(n) total instead of
    O(n log n).
"""

from typing import Any, Callable, List, Optional  # import the typing tools this file uses for type hints


class _BinaryHeap:  # shared base class holding all the array/index/sift logic
    """Shared array-backed heap logic; MinHeap/MaxHeap subclass this."""

    def __init__(self, comparator: Callable[[Any, Any], bool]) -> None:  # constructor, takes a comparator function
        self._data: List[Any] = []  # the underlying array that stores the heap's elements
        self._has_priority_over = comparator  # store the comparator so sift methods can use it later

    @staticmethod  # doesn't need access to self, it's pure math on an index
    def _parent(i: int) -> int:  # given a child index, return its parent's index
        return (i - 1) // 2  # standard array-heap parent formula

    @staticmethod  # doesn't need access to self
    def _left(i: int) -> int:  # given a node index, return its left child's index
        return 2 * i + 1  # standard array-heap left-child formula

    @staticmethod  # doesn't need access to self
    def _right(i: int) -> int:  # given a node index, return its right child's index
        return 2 * i + 2  # standard array-heap right-child formula

    def peek(self) -> Any:  # look at the root without removing it
        """Return the root without removing it. O(1)."""
        if self.is_empty():  # check whether there's anything to look at
            raise IndexError("peek from an empty heap")  # fail loudly instead of returning garbage
        return self._data[0]  # root of the heap always lives at array index 0

    def is_empty(self) -> bool:  # check whether the heap currently has any elements
        """O(1)."""
        return len(self._data) == 0  # true only when the underlying array is empty

    def __len__(self) -> int:  # lets Python's built-in len() work on this object
        return len(self._data)  # just forward to the array's length

    def heapify(self, array: Optional[List[Any]] = None) -> None:  # build a valid heap in O(n)
        """Build a valid heap in O(n) time from an existing array."""
        if array is not None:  # caller gave us new data to build from
            self._data = list(array)  # copy it in, so we don't mutate their original list
        n = len(self._data)  # total number of elements now stored
        last_internal_node = n // 2 - 1  # index of the last node that actually has a child
        for i in range(last_internal_node, -1, -1):  # walk backward from there to the root (index 0)
            self._sift_down(i)  # fix the heap property at each node along the way

    def _insert(self, value: Any) -> None:  # shared insert logic used by both MinHeap and MaxHeap
        self._data.append(value)  # new value goes at the end of the array first
        self._sift_up(len(self._data) - 1)  # then bubble it upward into a valid spot

    def _extract_root(self) -> Any:  # shared "remove the root" logic
        if self.is_empty():  # nothing to remove
            raise IndexError("extract from an empty heap")  # fail loudly instead of crashing weirdly later
        root = self._data[0]  # save the root value, this is what we'll return
        last = self._data.pop()  # remove the last element from the array
        if self._data:  # only if the heap still has elements left
            self._data[0] = last  # move that last element into the now-empty root spot
            self._sift_down(0)  # then push it down into a valid position
        return root  # hand back the value that used to be at the root

    def _sift_up(self, i: int) -> None:  # move element at index i toward the root as needed
        data = self._data  # local shortcut to the array
        above = self._has_priority_over  # local shortcut to the comparator function
        while i > 0:  # keep going until we reach the root
            p = self._parent(i)  # figure out the parent's index
            if above(data[i], data[p]):  # does our node outrank its parent?
                data[i], data[p] = data[p], data[i]  # swap them if so
                i = p  # move our "current position" pointer up to where we just swapped to
            else:  # our node no longer outranks its parent
                break  # heap property holds here, stop looping

    def _sift_down(self, i: int) -> None:  # move element at index i toward the leaves as needed
        data = self._data  # local shortcut to the array
        above = self._has_priority_over  # local shortcut to the comparator function
        n = len(data)  # current size, used to check if children exist
        while True:  # loop until we break out below
            left, right = self._left(i), self._right(i)  # compute both child indices
            top = i  # assume current node is already correctly placed
            if left < n and above(data[left], data[top]):  # left child exists and outranks current top
                top = left  # left child becomes the new candidate for "top"
            if right < n and above(data[right], data[top]):  # right child exists and outranks current top
                top = right  # right child becomes the new candidate for "top"
            if top == i:  # neither child outranked the current node
                break  # heap property holds here, stop looping
            data[i], data[top] = data[top], data[i]  # otherwise swap the node down into place
            i = top  # move our "current position" pointer down to where we just swapped to


class MinHeap(_BinaryHeap):  # a heap where the smallest element is always at the root
    """extract_min() always returns the smallest element, in O(log n)."""

    def __init__(self, items: Optional[List[Any]] = None) -> None:  # constructor, optionally takes starting data
        super().__init__(comparator=lambda a, b: a < b)  # a "outranks" b when a is smaller
        if items:  # if starting data was actually provided
            self.heapify(items)  # build the heap from it in O(n)

    def insert(self, value: Any) -> None:  # add one value to this heap
        """Insert a value. O(log n) worst case."""
        self._insert(value)  # hand off to the shared base-class logic

    def extract_min(self) -> Any:  # remove and return the smallest value
        """Remove and return the smallest value. O(log n) worst case."""
        return self._extract_root()  # hand off to the shared base-class logic


class MaxHeap(_BinaryHeap):  # a heap where the largest element is always at the root
    """extract_max() always returns the largest element, in O(log n)."""

    def __init__(self, items: Optional[List[Any]] = None) -> None:  # constructor, optionally takes starting data
        super().__init__(comparator=lambda a, b: a > b)  # a "outranks" b when a is larger
        if items:  # if starting data was actually provided
            self.heapify(items)  # build the heap from it in O(n)

    def insert(self, value: Any) -> None:  # add one value to this heap
        """Insert a value. O(log n) worst case."""
        self._insert(value)  # hand off to the shared base-class logic

    def extract_max(self) -> Any:  # remove and return the largest value
        """Remove and return the largest value. O(log n) worst case."""
        return self._extract_root()  # hand off to the shared base-class logic


class PriorityQueue:  # a queue where items come out in priority order instead of arrival order
    """Priority queue built on top of MinHeap. Lower priority number = served first."""

    def __init__(self) -> None:  # constructor, starts empty
        self._heap: MinHeap = MinHeap()  # the queue is really just a MinHeap under the hood
        self._counter: int = 0  # increases with each enqueue, used to break priority ties

    def enqueue(self, item: Any, priority: int) -> None:  # add an item with a given priority
        """Add `item` with the given `priority`. O(log n)."""
        entry = (priority, self._counter, item)  # bundle priority, tie-breaker, and the actual item together
        self._counter += 1  # next item enqueued gets a larger tie-breaker value
        self._heap.insert(entry)  # the heap will sort by (priority, counter) automatically via tuple comparison

    def dequeue(self) -> Any:  # remove and return the highest-priority item
        """Remove and return the highest-priority item. O(log n)."""
        _priority, _seq, item = self._heap.extract_min()  # unpack the tuple, ignore priority/tie-breaker
        return item  # give back just the original item

    def peek(self) -> Any:  # look at the highest-priority item without removing it
        """Return the highest-priority item without removing it. O(1)."""
        _priority, _seq, item = self._heap.peek()  # unpack the tuple, ignore priority/tie-breaker
        return item  # give back just the original item

    def is_empty(self) -> bool:  # check whether the queue currently has any items
        return self._heap.is_empty()  # delegate straight to the underlying heap

    def __len__(self) -> int:  # lets Python's built-in len() work on this object
        return len(self._heap)  # delegate straight to the underlying heap
