
from src.sorting.merge_sort import merge_sort


def test_normal_list():
    assert merge_sort([3, 1, 2]) == [1, 2, 3]


def test_empty_list():
    assert merge_sort([]) == []


def test_single_element():
    assert merge_sort([5]) == [5]


def test_all_equal():
    assert merge_sort([4, 4, 4]) == [4, 4, 4]


def test_negative_numbers():
    assert merge_sort([3, -1, 0, -5]) == [-5, -1, 0, 3]


def test_original_not_modified():
    original = [3, 1, 2]
    merge_sort(original)

    assert original == [3, 1, 2]
