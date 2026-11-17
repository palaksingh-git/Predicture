"""Synthetic next-digit sequence data."""

import numpy as np


PATTERN_TYPES = (
    "arithmetic",
    "repeating_cycle",
    "alternating",
    "constant",
    "fibonacci",
    "countdown",
    "random",
)


def _make_sequence(rng, pattern):
    if pattern == "arithmetic":
        start = int(rng.integers(10))
        step = int(rng.integers(1, 10))
        return [(start + step * i) % 10 for i in range(11)]

    if pattern == "repeating_cycle":
        cycle_length = int(rng.integers(2, 6))
        cycle = rng.integers(0, 10, size=cycle_length).tolist()
        return [cycle[i % cycle_length] for i in range(11)]

    if pattern == "alternating":
        first = int(rng.integers(10))
        second = int(rng.integers(9))
        if second >= first:
            second += 1
        return [first if i % 2 == 0 else second for i in range(11)]

    if pattern == "constant":
        digit = int(rng.integers(10))
        return [digit] * 11

    if pattern == "fibonacci":
        sequence = rng.integers(0, 10, size=2).tolist()
        for _ in range(9):
            sequence.append((sequence[-1] + sequence[-2]) % 10)
        return sequence

    if pattern == "countdown":
        start = int(rng.integers(10))
        return [(start - i) % 10 for i in range(11)]

    if pattern == "random":
        return rng.integers(0, 10, size=11).tolist()

    raise ValueError("Unknown pattern type: {}".format(pattern))


def generate_dataset(size, seed=0, noise_fraction=0.05):
    """Return inputs, labels, and pattern tags for a seeded synthetic split."""
    if size < 1:
        raise ValueError("size must be positive")
    if not 0 <= noise_fraction <= 1:
        raise ValueError("noise_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    sequences = np.empty((size, 10), dtype=np.int64)
    labels = np.empty(size, dtype=np.int64)
    tags = np.empty(size, dtype=object)

    for row in range(size):
        if rng.random() < noise_fraction:
            pattern = "random"
        else:
            pattern = PATTERN_TYPES[int(rng.integers(len(PATTERN_TYPES) - 1))]
        full_sequence = _make_sequence(rng, pattern)
        sequences[row] = full_sequence[:10]
        labels[row] = full_sequence[10]
        tags[row] = pattern

    return sequences, labels, tags
