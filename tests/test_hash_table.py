"""
test_hash_table.py
-------------------
pytest tests for ChainingHashTable and OpenAddressingHashTable
(src/data_structures/hash_table.py).

Run with:
    pytest tests/test_hash_table.py -v
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_structures.hash_table import ChainingHashTable, OpenAddressingHashTable


@pytest.fixture(params=[ChainingHashTable, OpenAddressingHashTable])
def table(request):
    """Runs every test in this fixture's tests against both hash table
    implementations, one at a time."""
    return request.param()


class TestSharedBehavior:
    def test_new_table_is_empty(self, table):
        assert len(table) == 0

    def test_insert_then_get(self, table):
        table.insert("apple", 1)
        table.insert("banana", 2)
        assert table.get("apple") == 1
        assert table.get("banana") == 2
        assert len(table) == 2

    def test_insert_same_key_updates_value(self, table):
        table.insert("apple", 1)
        table.insert("apple", 99)
        assert table.get("apple") == 99
        assert len(table) == 1

    def test_get_missing_key_raises(self, table):
        table.insert("apple", 1)
        with pytest.raises(KeyError):
            table.get("nonexistent")

    def test_delete_removes_key(self, table):
        table.insert("apple", 1)
        table.insert("banana", 2)
        table.delete("apple")
        assert "apple" not in table
        assert "banana" in table
        assert len(table) == 1

    def test_delete_missing_key_raises(self, table):
        with pytest.raises(KeyError):
            table.delete("nonexistent")

    def test_contains_operator(self, table):
        table.insert("apple", 1)
        assert "apple" in table
        assert "banana" not in table

    def test_reinsert_after_delete_works(self, table):
        table.insert("apple", 1)
        table.delete("apple")
        table.insert("banana", 2)
        assert "apple" not in table
        assert table.get("banana") == 2

    def test_grows_beyond_initial_capacity_without_errors(self, table):
        n = 200
        for i in range(n):
            table.insert(f"key{i}", i)

        assert len(table) == n
        for i in range(n):
            assert table.get(f"key{i}") == i

    def test_matches_a_real_dict_under_random_operations(self, table):
        random.seed(42)
        reference = {}
        keys = [f"k{i}" for i in range(300)]

        for _ in range(2000):
            key = random.choice(keys)
            if random.random() < 0.7:
                value = random.randint(0, 10_000)
                table.insert(key, value)
                reference[key] = value
            else:
                if key in reference:
                    table.delete(key)
                    del reference[key]

        assert len(table) == len(reference)
        for key, value in reference.items():
            assert table.get(key) == value


class TestChainingSpecifics:
    def test_load_factor_tracks_size_over_capacity(self):
        ht = ChainingHashTable(capacity=8)
        assert ht.load_factor() == 0.0
        for i in range(4):
            ht.insert(i, i)
        assert ht.load_factor() == pytest.approx(4 / 8)

    def test_rehash_triggers_and_doubles_capacity(self):
        ht = ChainingHashTable(capacity=4, load_factor_threshold=0.75)
        for i in range(3):
            ht.insert(i, i)
        assert ht._capacity == 4

        ht.insert(3, 3)
        assert ht._capacity == 8
        for i in range(4):
            assert ht.get(i) == i

    def test_forced_collisions_are_handled_via_chaining(self):
        ht = ChainingHashTable(capacity=1, load_factor_threshold=1000)
        for i in range(10):
            ht.insert(i, i * 10)
        assert ht.max_chain_length() == 10
        for i in range(10):
            assert ht.get(i) == i * 10

    def test_max_chain_length_on_empty_table(self):
        ht = ChainingHashTable()
        assert ht.max_chain_length() == 0


class TestOpenAddressingSpecifics:
    def test_load_factor_tracks_size_over_capacity(self):
        ht = OpenAddressingHashTable(capacity=8)
        assert ht.load_factor() == 0.0
        for i in range(4):
            ht.insert(i, i)
        assert ht.load_factor() == pytest.approx(4 / 8)

    def test_rehash_triggers_and_doubles_capacity(self):
        ht = OpenAddressingHashTable(capacity=4, load_factor_threshold=0.5)
        ht.insert(1, "a")
        assert ht._capacity == 4
        ht.insert(2, "b")
        assert ht._capacity == 4
        ht.insert(3, "c")
        assert ht.get(1) == "a"
        assert ht.get(2) == "b"
        assert ht.get(3) == "c"

    def test_forced_collisions_are_handled_via_linear_probing(self):
        ht = OpenAddressingHashTable(capacity=16, load_factor_threshold=1000)
        keys = [i * 16 for i in range(10)]
        for k in keys:
            ht.insert(k, k)
        for k in keys:
            assert ht.get(k) == k

    def test_tombstones_do_not_break_later_lookups(self):
        ht = OpenAddressingHashTable(capacity=16, load_factor_threshold=1000)
        keys = [i * 16 for i in range(5)]
        for k in keys:
            ht.insert(k, f"value-{k}")

        ht.delete(keys[0])

        for k in keys[1:]:
            assert ht.get(k) == f"value-{k}"
        assert keys[0] not in ht

    def test_tombstone_slot_gets_reused_on_next_insert(self):
        """Tombstones only get reused by a LATER insert whose probe
        sequence actually passes over that exact slot -- so this test
        must force a real collision (same home slot) rather than using
        two unrelated keys that likely land in different slots entirely."""
        ht = OpenAddressingHashTable(capacity=8, load_factor_threshold=1000)
        key_a, key_b = 0, 8  # both hash to slot 0 (8 % 8 == 0), guaranteed collision
        ht.insert(key_a, "first")
        ht.insert(key_b, "second")  # forced to probe past key_a's slot
        ht.delete(key_a)  # leaves a tombstone in key_a's original slot
        assert ht._tombstone_count == 1

        key_c = 16  # also hashes to slot 0, so its probe sequence passes the tombstone first
        ht.insert(key_c, "third")  # should land in the reused tombstone slot
        assert ht._tombstone_count == 0
        assert ht.get(key_b) == "second"
        assert ht.get(key_c) == "third"
        assert key_a not in ht
