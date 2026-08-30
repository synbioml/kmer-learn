"""Compare collision-chain strategies for position-bounded shuffling.

This is an isolated prototype; it is not used by ``kmer.background.LocalShuffler``.
Each shuffle inserts items in random order into legal destination intervals.
On collision, the previous occupant is displaced and samples again.

Policies
--------
``uniform-kick``
    Sample uniformly from every legal slot, kicking its occupant if necessary.
``free-first``
    Sample a legal free slot when one exists; otherwise sample an occupied
    legal slot and kick its occupant.

A randomized augmenting-path search completes an insertion if a collision
chain exceeds ``max_turns``.  Since every item can occupy its original index,
a perfect matching always exists.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class ShuffleStats:
    turns: list[int]
    fallbacks: int = 0


def _random_index(rng: np.random.Generator, values: np.ndarray) -> int:
    return int(values[int(rng.integers(len(values)))])


def _augment(
    unmatched: int,
    owner: np.ndarray,
    assigned: np.ndarray,
    max_shift: int,
    rng: np.random.Generator,
) -> None:
    """Find a randomized augmenting path from ``unmatched`` to a free slot."""
    length = len(owner)
    queue = deque([unmatched])
    parent: dict[int, tuple[int, int]] = {}
    seen = {unmatched}

    while queue:
        item = queue.popleft()
        lower = max(0, item - max_shift)
        upper = min(length, item + max_shift + 1)
        slots = rng.permutation(np.arange(lower, upper, dtype=np.int64))

        for slot_value in slots:
            slot = int(slot_value)
            occupant = int(owner[slot])
            if occupant == -1:
                path = [(item, slot)]
                while item != unmatched:
                    previous, previous_slot = parent[item]
                    path.append((previous, previous_slot))
                    item = previous
                for path_item, path_slot in path:
                    owner[path_slot] = path_item
                    assigned[path_item] = path_slot
                return

            if occupant not in seen:
                seen.add(occupant)
                parent[occupant] = (item, slot)
                queue.append(occupant)

    raise RuntimeError("no augmenting path despite identity perfect matching")


def collision_shuffle(
    length: int,
    max_shift: int,
    rng: np.random.Generator,
    policy: str,
    max_turns: int,
) -> tuple[np.ndarray, ShuffleStats]:
    """Generate one bounded permutation with a collision-chain policy."""
    if policy not in {"uniform-kick", "free-first"}:
        raise ValueError(f"unknown policy: {policy}")

    max_shift = min(max_shift, max(0, length - 1))
    owner = np.full(length, -1, dtype=np.int64)
    assigned = np.full(length, -1, dtype=np.int64)
    stats = ShuffleStats(turns=[])

    for first_item_value in rng.permutation(length):
        current = int(first_item_value)
        turns = 0

        while turns < max_turns:
            lower = max(0, current - max_shift)
            upper = min(length, current + max_shift + 1)
            legal = np.arange(lower, upper, dtype=np.int64)

            if policy == "free-first":
                free = legal[owner[legal] == -1]
                destination = _random_index(rng, free if len(free) else legal)
            else:
                destination = _random_index(rng, legal)

            turns += 1
            displaced = int(owner[destination])
            owner[destination] = current
            assigned[current] = destination

            if displaced == -1:
                break

            assigned[displaced] = -1
            current = displaced
        else:
            _augment(current, owner, assigned, max_shift, rng)
            stats.fallbacks += 1

        stats.turns.append(turns)

    if np.any(owner < 0) or np.any(assigned < 0):
        raise RuntimeError("collision shuffle produced an incomplete matching")
    if np.any(np.abs(assigned - np.arange(length)) > max_shift):
        raise RuntimeError("collision shuffle violated max_shift")

    return owner, stats


def benchmark(
    length: int,
    max_shift: int,
    trials: int,
    policy: str,
    seed: int,
    max_turns: int,
) -> dict[str, float]:
    rng = np.random.Generator(np.random.Philox(seed))
    histogram = np.zeros(2 * max_shift + 1, dtype=np.int64)
    all_turns = []
    fallbacks = 0
    interior_items = max(0, length - 2 * max_shift)

    started = perf_counter()
    for _ in range(trials):
        permutation, stats = collision_shuffle(
            length, max_shift, rng, policy, max_turns
        )
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(length)
        shifts = inverse - np.arange(length)
        if interior_items:
            interior = shifts[max_shift : length - max_shift]
            histogram += np.bincount(
                interior + max_shift, minlength=len(histogram)
            )
        all_turns.extend(stats.turns)
        fallbacks += stats.fallbacks
    elapsed = perf_counter() - started

    probabilities = histogram / histogram.sum()
    uniform = np.full(len(histogram), 1 / len(histogram))
    turns = np.asarray(all_turns)
    collided = turns[turns > 1]

    return {
        "shift_tv": float(0.5 * np.abs(probabilities - uniform).sum()),
        "mean_turns": float(turns.mean()),
        "mean_collision_turns": float(collided.mean()) if len(collided) else 1.0,
        "collision_fraction": float((turns > 1).mean()),
        "p95_turns": float(np.quantile(turns, 0.95)),
        "p99_turns": float(np.quantile(turns, 0.99)),
        "max_turns": float(turns.max()),
        "fallbacks": float(fallbacks),
        "seconds": elapsed,
        "items_per_second": length * trials / elapsed,
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--shifts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=10_000)
    args = parser.parse_args()

    heading = (
        "policy       S  shift-TV  E[turns]  collision%  "
        "E[turns|collision]  p95  p99  max  fallbacks  items/s"
    )
    print(heading)
    for max_shift in args.shifts:
        if 2 * max_shift >= args.length:
            raise ValueError("length must exceed 2 * max_shift")
        for policy in ("uniform-kick", "free-first"):
            result = benchmark(
                args.length,
                max_shift,
                args.trials,
                policy,
                args.seed,
                args.max_turns,
            )
            print(
                f"{policy:12s} {max_shift:2d}  {result['shift_tv']:8.4f}  "
                f"{result['mean_turns']:8.3f}  "
                f"{100 * result['collision_fraction']:9.2f}  "
                f"{result['mean_collision_turns']:18.3f}  "
                f"{result['p95_turns']:3.0f}  {result['p99_turns']:3.0f}  "
                f"{result['max_turns']:3.0f}  {result['fallbacks']:9.0f}  "
                f"{result['items_per_second']:,.0f}"
            )


if __name__ == "__main__":
    main()
