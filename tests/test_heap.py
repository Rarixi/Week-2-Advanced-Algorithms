"""
test_heap.py
------------
pytest tests for MinHeap, MaxHeap, and PriorityQueue (src/structures/heap.py).

Run with:
    pytest tests/test_heap.py -v
"""

import random
import sys
import os

# Make sure "src" is importable when running pytest from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_structures.heap import MinHeap, MaxHeap, PriorityQueue


class TestMinHeap:
    def test_new_heap_is_empty(self):
        h = MinHeap()
        assert h.is_empty() is True
        assert len(h) == 0

    def test_insert_then_peek_shows_smallest(self):
        h = MinHeap()
        h.insert(5)
        h.insert(3)
        h.insert(8)
        h.insert(1)
        assert h.peek() == 1          # smallest value should be on top
        assert len(h) == 4            # peek must not remove anything

    def test_extract_min_returns_values_in_sorted_order(self):
        h = MinHeap()
        values = [5, 3, 8, 1, 9, 2, 7]
        for v in values:
            h.insert(v)

        extracted = [h.extract_min() for _ in range(len(values))]
        assert extracted == sorted(values)
        assert h.is_empty() is True

    def test_heapify_builds_valid_heap_from_list(self):
        values = [9, 4, 7, 1, 3, 8, 2, 6, 5]
        h = MinHeap(values)          # constructor calls heapify() internally
        extracted = [h.extract_min() for _ in range(len(values))]
        assert extracted == sorted(values)

    def test_peek_on_empty_heap_raises(self):
        h = MinHeap()
        with pytest.raises(IndexError):
            h.peek()

    def test_extract_min_on_empty_heap_raises(self):
        h = MinHeap()
        with pytest.raises(IndexError):
            h.extract_min()

    def test_random_stress(self):
        """Insert a large batch of random values and confirm extraction
        order always matches Python's own sorted()."""
        random.seed(42)
        values = [random.randint(-1000, 1000) for _ in range(500)]
        h = MinHeap()
        for v in values:
            h.insert(v)
        extracted = [h.extract_min() for _ in range(len(values))]
        assert extracted == sorted(values)

    def test_duplicates_are_handled(self):
        h = MinHeap([5, 5, 5, 1, 1, 9])
        extracted = [h.extract_min() for _ in range(6)]
        assert extracted == [1, 1, 5, 5, 5, 9]


class TestMaxHeap:
    def test_new_heap_is_empty(self):
        h = MaxHeap()
        assert h.is_empty() is True

    def test_insert_then_peek_shows_largest(self):
        h = MaxHeap()
        for v in [5, 3, 8, 1]:
            h.insert(v)
        assert h.peek() == 8

    def test_extract_max_returns_values_in_reverse_sorted_order(self):
        values = [5, 3, 8, 1, 9, 2, 7]
        h = MaxHeap()
        for v in values:
            h.insert(v)

        extracted = [h.extract_max() for _ in range(len(values))]
        assert extracted == sorted(values, reverse=True)

    def test_heapify_builds_valid_heap_from_list(self):
        values = [9, 4, 7, 1, 3, 8, 2, 6, 5]
        h = MaxHeap(values)
        extracted = [h.extract_max() for _ in range(len(values))]
        assert extracted == sorted(values, reverse=True)

    def test_extract_max_on_empty_heap_raises(self):
        h = MaxHeap()
        with pytest.raises(IndexError):
            h.extract_max()

    def test_random_stress(self):
        random.seed(7)
        values = [random.randint(-1000, 1000) for _ in range(500)]
        h = MaxHeap()
        for v in values:
            h.insert(v)
        extracted = [h.extract_max() for _ in range(len(values))]
        assert extracted == sorted(values, reverse=True)


class TestPriorityQueue:
    def test_new_queue_is_empty(self):
        pq = PriorityQueue()
        assert pq.is_empty() is True
        assert len(pq) == 0

    def test_lower_priority_number_comes_out_first(self):
        pq = PriorityQueue()
        pq.enqueue("low urgency", priority=10)
        pq.enqueue("high urgency", priority=1)
        pq.enqueue("medium urgency", priority=5)

        assert pq.dequeue() == "high urgency"
        assert pq.dequeue() == "medium urgency"
        assert pq.dequeue() == "low urgency"
        assert pq.is_empty() is True

    def test_equal_priorities_are_served_in_insertion_order(self):
        """Tie-breaker (the internal counter) should keep FIFO order
        among items that share the same priority."""
        pq = PriorityQueue()
        pq.enqueue("first", priority=1)
        pq.enqueue("second", priority=1)
        pq.enqueue("third", priority=1)

        assert pq.dequeue() == "first"
        assert pq.dequeue() == "second"
        assert pq.dequeue() == "third"

    def test_peek_does_not_remove(self):
        pq = PriorityQueue()
        pq.enqueue("task", priority=3)
        assert pq.peek() == "task"
        assert len(pq) == 1  # still there after peek

    def test_dequeue_on_empty_queue_raises(self):
        pq = PriorityQueue()
        with pytest.raises(IndexError):
            pq.dequeue()

    def test_unorderable_items_with_equal_priority_do_not_crash(self):
        """Items like dicts aren't directly comparable with '<'. The
        tie-breaker in PriorityQueue must prevent Python from ever
        trying to compare the raw items themselves."""
        pq = PriorityQueue()
        pq.enqueue({"task": "a"}, priority=1)
        pq.enqueue({"task": "b"}, priority=1)
        # Should not raise TypeError: '<' not supported between instances of 'dict' and 'dict'
        first = pq.dequeue()
        second = pq.dequeue()
        assert first == {"task": "a"}
        assert second == {"task": "b"}
