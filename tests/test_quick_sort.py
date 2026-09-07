from src.sorting.quick_sort import quick_sort


def test_normal_list():
    assert quick_sort([5, 2, 8, 1, 3]) == [1, 2, 3, 5, 8]


def test_empty_list():
    assert quick_sort([]) == []


def test_single_element():
    assert quick_sort([7]) == [7]


def test_all_equal():
    assert quick_sort([4, 4, 4, 4]) == [4, 4, 4, 4]


def test_negative_numbers():
    assert quick_sort([3, -1, 0, -5]) == [-5, -1, 0, 3]


def test_already_sorted():
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_original_not_modified():
    original = [5, 2, 8, 1]

    sorted_list = quick_sort(original)

    assert sorted_list == [1, 2, 5, 8]
    assert original == [5, 2, 8, 1]
