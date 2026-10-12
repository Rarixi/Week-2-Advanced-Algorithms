"""
montecarlo_primality.py
-------------------------
Part 3: Monte Carlo Primality Testing -- two probabilistic ("Monte Carlo") primality
tests, Fermat's and Miller-Rabin's, each run for k independent rounds against a
candidate number, plus a deterministic trial-division check to compare both error
rate and speed against.

Both probabilistic tests can only ever say "probably prime" or "definitely
composite" -- a round that fails proves compositeness outright, but a round that
passes never proves primality, only fails to disprove it. Running k independent
rounds is what drives the false "probably prime" rate down, at a cost of k times the
work per round.

Provides:
    fermat_round(n, rng)                 -- one round of Fermat's test
    fermat_test(n, k, rng)                -- k rounds; True unless any round fails
    miller_rabin_round(n, rng)           -- one round of Miller-Rabin's test
    miller_rabin_test(n, k, rng)          -- k rounds; True unless any round fails
    deterministic_is_prime(n)             -- trial division up to sqrt(n), O(sqrt(n))
"""

import math


def fermat_round(n, rng):
    """One round of Fermat's primality test: picks a random witness a in [2, n-2] and
    checks Fermat's little theorem, a^(n-1) = 1 (mod n). Returns False (definitely
    composite) if the theorem fails, True ("probably prime", not proven) if it holds.

    Fermat's test has a known weakness: Carmichael numbers are composite numbers that
    satisfy a^(n-1) = 1 (mod n) for ALMOST EVERY witness a, so Fermat's test can keep
    returning "probably prime" on them no matter how large k gets -- a real gap that
    shows up in this week's empirical false-positive measurements, not just a
    theoretical footnote.
    """
    if n < 4:
        return n in (2, 3)
    a = rng.randint(2, n - 2)
    return pow(a, n - 1, n) == 1


def fermat_test(n, k, rng):
    """Runs k independent rounds of Fermat's test. Returns False the moment any round
    proves compositeness; returns True ("probably prime") only if every round passes."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        if not fermat_round(n, rng):
            return False
    return True


def miller_rabin_round(n, rng):
    """One round of the Miller-Rabin test. Writes n - 1 = 2^r * d with d odd, picks a
    random witness a, and checks whether a^d, a^(2d), a^(4d), ..., a^(2^(r-1) * d)
    ever hits n - 1 (mod n) -- the sequence of successive squarings a^(n-1) = 1 (mod n)
    implies, which is a strictly stronger condition than Fermat's plain a^(n-1) = 1
    check. That extra structure is exactly why Miller-Rabin has NO Carmichael-number
    style blind spot: at most 1/4 of possible witnesses can ever falsely pass a
    composite number, a guarantee that holds for every composite, not just most of
    them."""
    if n < 4:
        return n in (2, 3)
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    a = rng.randint(2, n - 2)
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True

    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
    return False


def miller_rabin_test(n, k, rng):
    """Runs k independent rounds of the Miller-Rabin test. Returns False the moment
    any round proves compositeness; returns True ("probably prime") only if every
    round passes. Worst-case false-positive probability is at most (1/4)^k for ANY
    composite n (see probability_analysis.miller_rabin_error_bound)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        if not miller_rabin_round(n, rng):
            return False
    return True


def deterministic_is_prime(n):
    """Deterministic trial division up to sqrt(n): checks every odd candidate divisor
    from 3 up to floor(sqrt(n)). Always correct, but O(sqrt(n)) -- fine for small n,
    but for n > 10^6 this already means checking on the order of 1000+ candidate
    divisors, and the gap only widens as n grows, which is exactly the point of
    comparing it against Fermat/Miller-Rabin's O(k * log^3 n)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    for divisor in range(3, limit + 1, 2):
        if n % divisor == 0:
            return False
    return True
