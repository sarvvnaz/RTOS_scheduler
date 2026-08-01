"""The two algorithms the project needs to split a total randomly.

Both answer the same question: *split a total into n random pieces*.

  uunifast      - used to split a task's utilization over its nodes
  randfixedsum  - used everywhere an upper bound matters
"""

import numpy as np


def uunifast(n: int, total: float, rng: np.random.Generator,
             limit: float = 1.0) -> np.ndarray:
    """Split ``total`` into ``n`` random values, each below ``limit``.

    This is UUniFast (Bini & Buttazzo). Each step keeps a random share of
    what is left. If a value comes out too large the whole draw is thrown
    away and repeated ("UUniFast-Discard"), because node utilizations must
    stay below 1. When the limit is too tight for that to work, the bounded
    splitter below is used instead.
    """
    if total >= n * limit:
        raise ValueError(f"cannot split {total:.2f} into {n} values below {limit}")

    for _ in range(100):
        values = np.empty(n)
        remaining = total
        for i in range(1, n):
            left = remaining * rng.uniform() ** (1.0 / (n - i))
            values[i - 1] = remaining - left
            remaining = left
        values[n - 1] = remaining
        if values.max() < limit:
            return values

    return randfixedsum(n, total, rng, high=limit)


def randfixedsum(n: int, total: float, rng: np.random.Generator,
                 high: float = None) -> np.ndarray:
    """Split ``total`` into ``n`` random values, none of them above ``high``.

    Two ways of doing it:

    1. The plain random split. ``dirichlet`` with all-ones parameters gives
       a uniformly random way of cutting a total into n pieces. If no piece
       is over the limit we are done, so this is what normally happens.
    2. If the limit is tight (many pieces, small limit), step 1 would be
       rejected again and again. Then the total is handed out one piece at
       a time instead, never taking so much that the remaining pieces
       cannot fit, and never leaving more than they can hold.
    """
    if n == 1:
        if high is not None and total > high + 1e-9:
            raise ValueError(f"{total:.2f} does not fit under {high:.2f}")
        return np.array([float(total)])

    if high is None:
        return rng.dirichlet(np.ones(n)) * total

    if total > n * high + 1e-9:
        raise ValueError(
            f"cannot split {total:.2f} into {n} values below {high:.2f}")

    for _ in range(100):
        pieces = rng.dirichlet(np.ones(n)) * total
        if pieces.max() <= high:
            return pieces

    pieces = np.empty(n)
    left = total
    for i in range(n):
        still_to_come = n - i - 1
        lowest = max(0.0, left - still_to_come * high)      # leave enough room
        highest = min(high, left)                           # do not overfill
        pieces[i] = rng.uniform(lowest, highest) if still_to_come else left
        left -= pieces[i]

    rng.shuffle(pieces)     # the order the pieces were drawn should not matter
    return pieces
