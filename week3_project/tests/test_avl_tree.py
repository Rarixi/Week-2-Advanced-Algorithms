"""
test_avl_tree.py
----------------
pytest tests for AVLTree (src/data_structures/avl_tree.py).

Run with:
    pytest tests/test_avl_tree.py -v
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_structures.avl_tree import AVLTree


def assert_is_valid_avl(tree: AVLTree) -> None:
    """Recursively check every node satisfies:
    1. BST ordering (left < node < right)
    2. |balance_factor| <= 1 (the actual AVL invariant)
    3. cached .height matches the real computed height (cache isn't stale)
    """

    def real_height(node):
        if node is None:
            return 0
        return 1 + max(real_height(node.left), real_height(node.right))

    def check(node, low, high):
        if node is None:
            return
        assert low < node.value < high, f"BST ordering violated at {node.value}"

        left_h = real_height(node.left)
        right_h = real_height(node.right)
        balance = left_h - right_h
        assert abs(balance) <= 1, f"AVL invariant violated at node {node.value}: balance={balance}"
        assert node.balance_factor == balance, (
            f"Stale balance_factor at node {node.value}: "
            f"cached={node.balance_factor}, actual={balance}"
        )
        assert node.height == real_height(node), (
            f"Stale height at node {node.value}: "
            f"cached={node.height}, actual={real_height(node)}"
        )

        check(node.left, low, node.value)
        check(node.right, node.value, high)

    check(tree.root, float("-inf"), float("inf"))


def max_avl_height(n: int) -> int:
    """Theoretical upper bound on an AVL tree's height for n nodes."""
    if n == 0:
        return 0
    return math.floor(1.4404 * math.log2(n + 2) - 0.328)


class TestBasicBehavior:
    def test_new_tree_is_empty(self):
        tree = AVLTree()
        assert tree.is_empty() is True
        assert tree.height() == 0
        assert tree.in_order_traversal() == []

    def test_search_on_empty_tree_returns_false(self):
        tree = AVLTree()
        assert tree.search(5) is False

    def test_insert_then_search_found_and_not_found(self):
        tree = AVLTree()
        for v in [50, 30, 70, 20, 40]:
            tree.insert(v)
        assert tree.search(30) is True
        assert tree.search(999) is False

    def test_duplicate_insert_is_ignored(self):
        tree = AVLTree()
        tree.insert(10)
        tree.insert(10)
        tree.insert(10)
        assert tree.in_order_traversal() == [10]

    def test_in_order_traversal_is_always_sorted(self):
        values = [50, 30, 70, 20, 40, 60, 80, 10, 25]
        tree = AVLTree()
        for v in values:
            tree.insert(v)
        assert tree.in_order_traversal() == sorted(values)


class TestRotations:
    def test_left_left_case_triggers_single_right_rotation(self):
        tree = AVLTree()
        for v in [3, 2, 1]:
            tree.insert(v)
        assert tree.root.value == 2
        assert tree.root.left.value == 1
        assert tree.root.right.value == 3
        assert_is_valid_avl(tree)

    def test_right_right_case_triggers_single_left_rotation(self):
        tree = AVLTree()
        for v in [1, 2, 3]:
            tree.insert(v)
        assert tree.root.value == 2
        assert tree.root.left.value == 1
        assert tree.root.right.value == 3
        assert_is_valid_avl(tree)

    def test_left_right_case_triggers_double_rotation(self):
        tree = AVLTree()
        for v in [3, 1, 2]:
            tree.insert(v)
        assert tree.root.value == 2
        assert tree.root.left.value == 1
        assert tree.root.right.value == 3
        assert_is_valid_avl(tree)

    def test_right_left_case_triggers_double_rotation(self):
        tree = AVLTree()
        for v in [1, 3, 2]:
            tree.insert(v)
        assert tree.root.value == 2
        assert tree.root.left.value == 1
        assert tree.root.right.value == 3
        assert_is_valid_avl(tree)


class TestBalanceCorrectness:
    def test_sequential_insert_stays_balanced(self):
        """This is the case that breaks a plain (non-balancing) BST --
        inserting already-sorted data. An AVL tree must NOT degrade
        into a straight line here."""
        tree = AVLTree()
        n = 1000
        for v in range(n):
            tree.insert(v)

        assert_is_valid_avl(tree)
        assert tree.height() <= max_avl_height(n), (
            f"Height {tree.height()} exceeds AVL bound {max_avl_height(n)} "
            f"for n={n} -- tree degraded toward a linked list"
        )

    def test_random_insert_stays_balanced(self):
        random.seed(42)
        values = random.sample(range(100_000), 2000)
        tree = AVLTree()
        for v in values:
            tree.insert(v)

        assert_is_valid_avl(tree)
        assert tree.height() <= max_avl_height(len(values))
        assert tree.in_order_traversal() == sorted(values)

    def test_stays_balanced_after_many_deletions(self):
        random.seed(7)
        values = random.sample(range(10_000), 500)
        tree = AVLTree()
        for v in values:
            tree.insert(v)

        to_delete = values[:250]
        for v in to_delete:
            tree.delete(v)
            assert_is_valid_avl(tree)  # check invariant holds after EVERY delete

        remaining = sorted(set(values) - set(to_delete))
        assert tree.in_order_traversal() == remaining


class TestDeletion:
    def test_delete_leaf_node(self):
        tree = AVLTree()
        for v in [50, 30, 70]:
            tree.insert(v)
        tree.delete(30)
        assert tree.search(30) is False
        assert tree.in_order_traversal() == [50, 70]
        assert_is_valid_avl(tree)

    def test_delete_node_with_one_child(self):
        tree = AVLTree()
        for v in [50, 30, 70, 20]:
            tree.insert(v)
        tree.delete(30)
        assert tree.search(30) is False
        assert tree.search(20) is True
        assert tree.in_order_traversal() == [20, 50, 70]
        assert_is_valid_avl(tree)

    def test_delete_node_with_two_children(self):
        tree = AVLTree()
        for v in [50, 30, 70, 20, 40, 60, 80]:
            tree.insert(v)
        tree.delete(50)
        assert tree.search(50) is False
        assert tree.in_order_traversal() == [20, 30, 40, 60, 70, 80]
        assert_is_valid_avl(tree)

    def test_delete_nonexistent_value_is_a_no_op(self):
        tree = AVLTree()
        for v in [50, 30, 70]:
            tree.insert(v)
        tree.delete(999)
        assert tree.in_order_traversal() == [30, 50, 70]

    def test_delete_from_empty_tree_does_not_crash(self):
        tree = AVLTree()
        tree.delete(5)
        assert tree.is_empty() is True

    def test_delete_all_nodes_empties_the_tree(self):
        values = [50, 30, 70, 20, 40, 60, 80]
        tree = AVLTree()
        for v in values:
            tree.insert(v)
        for v in values:
            tree.delete(v)
        assert tree.is_empty() is True
        assert tree.height() == 0
        assert tree.in_order_traversal() == []
