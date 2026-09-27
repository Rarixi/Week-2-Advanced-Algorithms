"""
avl_tree.py
-----------
A self-balancing binary search tree (AVL tree).

Why balance matters:
    A plain BST can degrade to O(n) height if you insert already-sorted
    data (it becomes a straight line, effectively a linked list). AVL
    trees prevent that by enforcing, after every insert/delete, that
    for any node the heights of its left and right subtrees differ by
    at most 1. This "balance factor" invariant guarantees the tree's
    height stays O(log n) no matter what order data arrives in, which
    is what keeps insert/search/delete all at O(log n).

Complexity (n = number of nodes in the tree):
    insert()              O(log n) -- descend one root-to-leaf path, rebalance on the way back up
    delete()              O(log n) -- same reasoning as insert
    search()              O(log n) -- standard BST descent, height guaranteed O(log n)
    height()              O(1)     -- cached on every node, never recomputed from scratch
    in_order_traversal()  O(n)     -- has to visit every node once
"""

from typing import Any, List, Optional  # import the typing tools this file uses for type hints


class _AVLNode:  # a single node in the tree
    """One node of the AVL tree. Stores its own cached height and
    balance factor so we never have to recompute them from scratch."""

    def __init__(self, value: Any) -> None:  # constructor, takes the value this node holds
        self.value: Any = value  # the actual data stored at this node
        self.left: Optional["_AVLNode"] = None  # left child, starts empty
        self.right: Optional["_AVLNode"] = None  # right child, starts empty
        self.height: int = 1  # a brand-new node is its own leaf, so height starts at 1
        self.balance_factor: int = 0  # a brand-new node has no children, so it's perfectly balanced


class AVLTree:  # the tree itself, wraps the recursive node logic behind a clean public API
    """Self-balancing binary search tree (AVL tree)."""

    def __init__(self) -> None:  # constructor, starts as an empty tree
        self.root: Optional[_AVLNode] = None  # empty tree has no root yet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, value: Any) -> None:  # add a value to the tree
        """Insert `value` into the tree. O(log n)."""
        self.root = self._insert(self.root, value)  # recursively insert, then reassign the (possibly new) root

    def delete(self, value: Any) -> None:  # remove a value from the tree
        """Delete `value` from the tree, if present. O(log n)."""
        self.root = self._delete(self.root, value)  # recursively delete, then reassign the (possibly new) root

    def search(self, value: Any) -> bool:  # check whether a value exists in the tree
        """Return True if `value` exists in the tree. O(log n)."""
        return self._search(self.root, value)  # delegate to the recursive helper

    def in_order_traversal(self) -> List[Any]:  # get all values back out in sorted order
        """Return all values in ascending sorted order. O(n)."""
        result: List[Any] = []  # accumulator list, built up during the traversal
        self._in_order(self.root, result)  # walk the tree, appending into result as we go
        return result  # hand back the fully sorted list

    def height(self) -> int:  # report the height of the whole tree
        """Return the height of the tree (0 for an empty tree). O(1),
        since height is cached on every node instead of recomputed."""
        return self._height(self.root)  # empty tree -> 0, otherwise the root's cached height

    def is_empty(self) -> bool:  # convenience check
        return self.root is None  # tree is empty exactly when there's no root node

    # ------------------------------------------------------------------
    # Internal helpers: height / balance
    # ------------------------------------------------------------------
    @staticmethod
    def _height(node: Optional[_AVLNode]) -> int:  # safe height lookup that handles None
        return node.height if node is not None else 0  # an empty subtree has height 0 by definition

    @staticmethod
    def _balance_factor(node: Optional[_AVLNode]) -> int:  # compute left-height minus right-height
        if node is None:  # an empty subtree has no balance to speak of
            return 0
        return AVLTree._height(node.left) - AVLTree._height(node.right)  # positive = left-heavy, negative = right-heavy

    def _update_node(self, node: _AVLNode) -> None:  # recompute + cache height and balance factor for one node
        node.height = 1 + max(self._height(node.left), self._height(node.right))  # height = 1 + tallest child subtree
        node.balance_factor = self._balance_factor(node)  # store the balance factor the assignment requires

    # ------------------------------------------------------------------
    # Internal helpers: rotations
    # ------------------------------------------------------------------
    def _rotate_left(self, z: _AVLNode) -> _AVLNode:  # standard AVL left rotation, returns new subtree root
        y = z.right  # y will become the new top of this subtree
        assert y is not None  # only ever called when z.right exists (right-heavy case)
        t2 = y.left  # y's left subtree has to move somewhere -- it becomes z's new right subtree

        y.left = z  # z becomes y's left child
        z.right = t2  # z picks up y's old left subtree as its own right subtree

        self._update_node(z)  # z's children changed, so recompute z's height/balance first
        self._update_node(y)  # then y, since y's height depends on z's new height

        return y  # y is now the top of this subtree

    def _rotate_right(self, z: _AVLNode) -> _AVLNode:  # standard AVL right rotation, returns new subtree root
        y = z.left  # y will become the new top of this subtree
        assert y is not None  # only ever called when z.left exists (left-heavy case)
        t3 = y.right  # y's right subtree has to move somewhere -- it becomes z's new left subtree

        y.right = z  # z becomes y's right child
        z.left = t3  # z picks up y's old right subtree as its own left subtree

        self._update_node(z)  # z's children changed, so recompute z's height/balance first
        self._update_node(y)  # then y, since y's height depends on z's new height

        return y  # y is now the top of this subtree

    def _rebalance(self, node: _AVLNode) -> _AVLNode:  # check + fix balance at this node, return new subtree root
        self._update_node(node)  # make sure height/balance_factor are current before checking them
        balance = node.balance_factor  # local copy for readability below

        if balance > 1:  # left-heavy: left subtree is too tall
            if self._balance_factor(node.left) < 0:  # left-right case: left child is itself right-heavy
                node.left = self._rotate_left(node.left)  # rotate left child first to turn this into a left-left case
            return self._rotate_right(node)  # single right rotation fixes the whole subtree

        if balance < -1:  # right-heavy: right subtree is too tall
            if self._balance_factor(node.right) > 0:  # right-left case: right child is itself left-heavy
                node.right = self._rotate_right(node.right)  # rotate right child first to turn this into a right-right case
            return self._rotate_left(node)  # single left rotation fixes the whole subtree

        return node  # already balanced, nothing to do

    # ------------------------------------------------------------------
    # Internal helpers: insert / delete / search / traversal
    # ------------------------------------------------------------------
    def _insert(self, node: Optional[_AVLNode], value: Any) -> _AVLNode:  # recursive insert, returns new subtree root
        if node is None:  # found the empty spot where this value belongs
            return _AVLNode(value)  # create and return a brand-new leaf node here

        if value < node.value:  # value belongs in the left subtree
            node.left = self._insert(node.left, value)  # recurse left, reattach the (possibly rebalanced) result
        elif value > node.value:  # value belongs in the right subtree
            node.right = self._insert(node.right, value)  # recurse right, reattach the (possibly rebalanced) result
        else:  # value already exists in the tree
            return node  # no duplicates allowed -- leave the tree unchanged

        return self._rebalance(node)  # on the way back up, fix balance at every ancestor we passed through

    def _delete(self, node: Optional[_AVLNode], value: Any) -> Optional[_AVLNode]:  # recursive delete
        if node is None:  # value isn't in the tree
            return None  # nothing to delete, nothing to rebalance

        if value < node.value:  # target is in the left subtree
            node.left = self._delete(node.left, value)  # recurse left, reattach the result
        elif value > node.value:  # target is in the right subtree
            node.right = self._delete(node.right, value)  # recurse right, reattach the result
        else:  # this is the node we need to remove
            if node.left is None:  # no left child -- right child (or None) simply replaces this node
                return node.right  # nothing below here needs rebalancing, we just spliced this node out
            if node.right is None:  # no right child -- left child simply replaces this node
                return node.left  # same idea, mirrored

            # Two children: replace this node's value with its in-order successor
            # (smallest value in the right subtree), then delete that successor
            # from the right subtree instead.
            successor = self._min_value_node(node.right)  # smallest value greater than node.value
            node.value = successor.value  # copy successor's value up into this node
            node.right = self._delete(node.right, successor.value)  # remove the now-duplicated successor node

        return self._rebalance(node)  # on the way back up, fix balance at every ancestor we passed through

    @staticmethod
    def _min_value_node(node: _AVLNode) -> _AVLNode:  # find the smallest value in a subtree
        current = node  # start at the given subtree root
        while current.left is not None:  # keep walking left as long as possible
            current = current.left  # smaller values are always further left
        return current  # leftmost node holds the smallest value

    def _search(self, node: Optional[_AVLNode], value: Any) -> bool:  # recursive search
        if node is None:  # fell off the tree without finding it
            return False  # value isn't present
        if value == node.value:  # found it
            return True
        if value < node.value:  # keep looking left
            return self._search(node.left, value)
        return self._search(node.right, value)  # keep looking right

    def _in_order(self, node: Optional[_AVLNode], result: List[Any]) -> None:  # recursive in-order walk
        if node is None:  # nothing here, stop this branch
            return
        self._in_order(node.left, result)  # visit everything smaller first
        result.append(node.value)  # then this node
        self._in_order(node.right, result)  # then everything larger
