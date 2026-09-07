"""Shared test configuration and fixtures."""
import pytest
import random
from typing import List, Callable

@pytest.fixture
def sample_arrays():
    """Provide standard test arrays for sorting algorithms."""
    return {
        'empty': [],
        'single': [42],
        'sorted': [1, 2, 3, 4, 5],
        'reverse': [5, 4, 3, 2, 1],
        'duplicates': [3, 1, 4, 1, 5, 9, 2, 6, 5],
        'all_same': [7, 7, 7, 7, 7],
        'negative': [-3, -1, -4, -1, -5],
        'mixed': [3, -1, 4, 0, -2, 7]
    }

@pytest.fixture
def large_random_array():
    """Generate large random array for stress testing."""
    random.seed(42)  # For reproducible tests
    return [random.randint(-1000, 1000) for _ in range(1000)]

def is_sorted(arr: List) -> bool:
    """Check if array is sorted in ascending order."""
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def has_same_elements(arr1: List, arr2: List) -> bool:
    """Check if two arrays contain the same elements (including duplicates)."""
    return sorted(arr1) == sorted(arr2)
