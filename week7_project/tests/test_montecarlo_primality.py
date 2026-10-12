"""
test_montecarlo_primality.py
-------------------------------
Correctness tests for Part 3: montecarlo_primality.py.
"""

import random

from src.randomized.montecarlo_primality import (
    fermat_test,
    miller_rabin_test,
    deterministic_is_prime,
)

KNOWN_PRIMES = [2, 3, 5, 7, 11, 13, 97, 541, 7919, 104729]
KNOWN_COMPOSITES = [4, 6, 8, 9, 10, 100, 561, 1024, 7921, 104730]
CARMICHAEL_NUMBERS = [561, 1105, 1729, 2465, 2821, 6601, 8911]


class TestDeterministicIsPrime:
    def test_known_primes(self):
        for p in KNOWN_PRIMES:
            assert deterministic_is_prime(p), f"{p} should be prime"

    def test_known_composites(self):
        for c in KNOWN_COMPOSITES:
            assert not deterministic_is_prime(c), f"{c} should be composite"

    def test_edge_cases(self):
        assert not deterministic_is_prime(0)
        assert not deterministic_is_prime(1)
        assert not deterministic_is_prime(-7)
        assert deterministic_is_prime(2)


class TestMillerRabin:
    def test_known_primes_always_pass(self, rng):
        for p in KNOWN_PRIMES:
            assert miller_rabin_test(p, 20, rng)

    def test_known_composites_usually_caught(self, rng):
        for c in KNOWN_COMPOSITES:
            assert not miller_rabin_test(c, 20, rng)

    def test_edge_cases(self, rng):
        assert not miller_rabin_test(0, 10, rng)
        assert not miller_rabin_test(1, 10, rng)
        assert miller_rabin_test(2, 10, rng)

    def test_carmichael_numbers_are_still_caught_at_reasonable_k(self):
        # Miller-Rabin has no Carmichael-number blind spot the way Fermat does --
        # across enough independent trials at k=20, it should essentially never be
        # fooled by any of these.
        rng = random.Random(100)
        fooled = 0
        trials = 20
        for c in CARMICHAEL_NUMBERS:
            for _ in range(trials):
                if miller_rabin_test(c, 20, random.Random(rng.random())):
                    fooled += 1
        assert fooled == 0


class TestFermat:
    def test_known_primes_always_pass(self, rng):
        for p in KNOWN_PRIMES:
            assert fermat_test(p, 20, rng)

    def test_non_carmichael_composites_usually_caught(self, rng):
        non_carmichael = [c for c in KNOWN_COMPOSITES if c not in CARMICHAEL_NUMBERS]
        for c in non_carmichael:
            assert not fermat_test(c, 20, rng)

    def test_edge_cases(self, rng):
        assert not fermat_test(0, 10, rng)
        assert not fermat_test(1, 10, rng)
        assert fermat_test(2, 10, rng)


class TestErrorRateComparison:
    def test_miller_rabin_false_positive_rate_bounded_by_quarter_to_the_k(self):
        # Empirical sanity check (not a proof): across many random composites and
        # many trials, Miller-Rabin's observed false-positive rate at a given k
        # should sit comfortably under the (1/4)^k theoretical worst-case bound.
        import math
        rng = random.Random(200)
        k = 5
        bound = 0.25 ** k
        composites = [c for c in range(100, 2000) if not deterministic_is_prime(c)]
        sampled = rng.sample(composites, 200)
        false_positives = sum(1 for c in sampled if miller_rabin_test(c, k, rng))
        observed_rate = false_positives / len(sampled)
        assert observed_rate <= bound
