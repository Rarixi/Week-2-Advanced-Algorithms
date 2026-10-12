"""
test_minhash_similarity.py
-----------------------------
Correctness tests for Part 5: minhash_similarity.py.
"""

import random

from src.randomized.minhash_similarity import (
    generate_hash_functions,
    minhash_signature,
    estimate_jaccard_from_signatures,
)
from src.utils.probability_analysis import true_jaccard
from src.utils.random_utils import random_token_set, related_token_set


class TestSignatureBasics:
    def test_signature_length_matches_num_hashes(self, rng):
        token_set = random_token_set(50, rng)
        hash_functions = generate_hash_functions(32, rng)
        signature = minhash_signature(token_set, hash_functions)
        assert len(signature) == 32

    def test_empty_set_signature_is_all_none(self, rng):
        hash_functions = generate_hash_functions(10, rng)
        signature = minhash_signature(set(), hash_functions)
        assert signature == [None] * 10

    def test_identical_sets_have_identical_signatures(self, rng):
        token_set = random_token_set(50, rng)
        hash_functions = generate_hash_functions(16, rng)
        sig_a = minhash_signature(token_set, hash_functions)
        sig_b = minhash_signature(token_set.copy(), hash_functions)
        assert sig_a == sig_b

    def test_signature_is_deterministic_given_same_hash_functions(self, rng):
        token_set = random_token_set(50, rng)
        hash_functions = generate_hash_functions(16, random.Random(5))
        sig_1 = minhash_signature(token_set, hash_functions)
        sig_2 = minhash_signature(token_set, hash_functions)
        assert sig_1 == sig_2


class TestEstimateJaccard:
    def test_identical_sets_estimate_to_one(self, rng):
        token_set = random_token_set(50, rng)
        hash_functions = generate_hash_functions(64, rng)
        sig = minhash_signature(token_set, hash_functions)
        assert estimate_jaccard_from_signatures(sig, sig) == 1.0

    def test_mismatched_lengths_raises(self):
        with __import__("pytest").raises(ValueError):
            estimate_jaccard_from_signatures([1, 2, 3], [1, 2])

    def test_disjoint_sets_estimate_near_zero(self):
        rng = random.Random(6)
        set_a = {f"a-only-{i}" for i in range(100)}
        set_b = {f"b-only-{i}" for i in range(100)}
        hash_functions = generate_hash_functions(256, rng)
        sig_a = minhash_signature(set_a, hash_functions)
        sig_b = minhash_signature(set_b, hash_functions)
        estimate = estimate_jaccard_from_signatures(sig_a, sig_b)
        assert estimate < 0.15  # should be close to 0, allowing for estimation noise


class TestEstimationAccuracy:
    def test_estimate_converges_toward_true_jaccard_as_k_grows(self):
        # Larger signature sizes should, on average across repeated trials, get
        # closer to the true similarity -- checked by averaging over many trials at
        # a small k and a large k and comparing mean absolute error.
        rng = random.Random(30)
        base = random_token_set(150, rng)
        related = related_token_set(base, 150, overlap_fraction=0.5, rng=rng)
        true_value = true_jaccard(base, related)

        def mean_error(k, num_trials=25):
            errors = []
            for trial in range(num_trials):
                hash_functions = generate_hash_functions(k, random.Random(1000 + trial))
                sig_a = minhash_signature(base, hash_functions)
                sig_b = minhash_signature(related, hash_functions)
                estimate = estimate_jaccard_from_signatures(sig_a, sig_b)
                errors.append(abs(estimate - true_value))
            return sum(errors) / len(errors)

        small_k_error = mean_error(16)
        large_k_error = mean_error(256)
        assert large_k_error < small_k_error
