"""
test_utils.py
-----------------
Tests for utils/rolling_hash.py, utils/text_loader.py, and
utils/generators.py: the rolling hash's O(1) roll matches direct
re-hashing of every window, FASTA parsing round trips correctly, and the
synthetic generators produce exactly the structure they promise (a
planted pattern actually ends up at the positions claimed, periodic and
repeated character strings are exactly as periodic/repeated as asked).
"""

from utils.generators import (
    periodic_string,
    random_dna_sequence,
    random_fasta_records,
    random_string,
    repeated_char_string,
    text_with_planted_pattern,
)
from utils.rolling_hash import RollingHash
from utils.text_loader import parse_fasta, write_fasta, load_fasta


class TestRollingHash:
    def test_roll_matches_direct_hash_of_every_window(self, rng):
        hasher = RollingHash()
        text = random_string(200, alphabet="abcd", rng=rng)
        window = 7
        rolled = [h for _, h in hasher.window_hashes(text, window)]
        direct = [hasher.hash_of(text[i:i + window]) for i in range(len(text) - window + 1)]
        assert rolled == direct

    def test_different_strings_usually_hash_differently(self, rng):
        hasher = RollingHash()
        seen = set()
        collisions = 0
        for _ in range(500):
            s = random_string(10, alphabet="abcdefgh", rng=rng)
            h = hasher.hash_of(s)
            if h in seen:
                collisions += 1
            seen.add(h)
        # with a 2^61-1 modulus, collisions among 500 random strings
        # should be effectively impossible
        assert collisions == 0

    def test_window_hashes_empty_on_window_longer_than_text(self):
        hasher = RollingHash()
        assert list(hasher.window_hashes("ab", 5)) == []


class TestGenerators:
    def test_periodic_string_is_built_from_repeating_the_period(self):
        text = periodic_string(10, "ab")
        assert text == "ababababab"

    def test_periodic_string_handles_non_dividing_length(self):
        text = periodic_string(7, "abc")
        assert text == "abcabca"

    def test_repeated_char_string(self):
        assert repeated_char_string(5, "z") == "zzzzz"

    def test_random_dna_sequence_uses_only_acgt(self, rng):
        seq = random_dna_sequence(500, rng=rng)
        assert len(seq) == 500
        assert set(seq) <= set("ACGT")

    def test_planted_pattern_actually_occurs_at_claimed_positions(self, rng):
        text, positions = text_with_planted_pattern(500, "needle", 5, rng=rng)
        assert len(positions) == 5
        for pos in positions:
            assert text[pos:pos + len("needle")] == "needle"

    def test_planted_pattern_occurrences_do_not_overlap(self, rng):
        text, positions = text_with_planted_pattern(200, "abcde", 10, rng=rng)
        for i in range(len(positions) - 1):
            assert positions[i] + 5 <= positions[i + 1]

    def test_random_fasta_records_shape(self, rng):
        records = random_fasta_records(3, 50, rng=rng)
        assert len(records) == 3
        for header, sequence in records:
            assert len(sequence) == 50
            assert set(sequence) <= set("ACGT")


class TestFastaRoundTrip:
    def test_parse_fasta_basic(self):
        text = ">seq1\nACGT\nACGT\n>seq2\nTTTT\n"
        records = parse_fasta(text)
        assert records == [("seq1", "ACGTACGT"), ("seq2", "TTTT")]

    def test_parse_fasta_ignores_blank_lines(self):
        text = ">seq1\nACGT\n\n\nACGT\n"
        records = parse_fasta(text)
        assert records == [("seq1", "ACGTACGT")]

    def test_write_then_load_round_trips(self, tmp_path, rng):
        records = random_fasta_records(2, 150, rng=rng)
        path = tmp_path / "genome.fasta"
        write_fasta(records, str(path))
        loaded = load_fasta(str(path))
        assert loaded == records

    def test_empty_input_gives_empty_list(self):
        assert parse_fasta("") == []
        assert parse_fasta("   \n  \n") == []
