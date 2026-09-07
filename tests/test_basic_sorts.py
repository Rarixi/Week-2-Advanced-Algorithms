from src.sorting.basic_sorts import (
    bubble_sort,
    selection_sort,
    insertion_sort
)


def test_bubble_sort():
    assert bubble_sort([5, 2, 8, 1, 3]) == [1, 2, 3, 5, 8]
    assert bubble_sort([]) == []
    assert bubble_sort([7]) == [7]
    assert bubble_sort([4, 4, 4]) == [4, 4, 4]


def test_selection_sort():
    assert selection_sort([5, 2, 8, 1, 3]) == [1, 2, 3, 5, 8]
    assert selection_sort([]) == []
    assert selection_sort([7]) == [7]
    assert selection_sort([4, 4, 4]) == [4, 4, 4]


def test_insertion_sort():
    assert insertion_sort([5, 2, 8, 1, 3]) == [1, 2, 3, 5, 8]
    assert insertion_sort([]) == []
    assert insertion_sort([7]) == [7]
    assert insertion_sort([4, 4, 4]) == [4, 4, 4]
