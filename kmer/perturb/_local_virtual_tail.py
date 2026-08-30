"""Position-bounded local sequence shuffling.

``LocalShuffler`` builds an output from left to right.  At each position it
considers the unplaced input items that can legally move there and weights each
candidate by a virtual-tail completion count.  The virtual tail places the
remaining active items without introducing later items, for which the nested
deadline count has a closed-form product.

The resulting distribution approximates the uniform distribution over all
permutations whose maximum displacement is bounded.  The displacement bound
itself is strict: an item originally at index ``i`` is always emitted within
``max_shift`` positions of ``i``.

Shared prefix and suffix products evaluate all candidate weights in
``O(max_shift)`` time per output position.  The complete shuffle therefore
takes ``O(n * max_shift)`` time and ``O(max_shift)`` auxiliary space.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Sequence

import numpy as np

from ._base import BaseBackgroundModel, SeqLike


def _virtual_tail_count(
    available: Sequence[int], position: int, max_shift: int
) -> int:
    """Count deadline-compatible orders of the currently available items.

    No later items are introduced in the virtual tail.  Since every available
    item's legal virtual positions form a prefix and ``available`` is sorted,
    the number of orders is the product of the remaining prefix sizes.
    """
    count = 1
    last_slot = len(available) - 1

    for placed, original_index in enumerate(available):
        deadline = min(last_slot, original_index + max_shift - position)
        choices = deadline + 1 - placed
        if choices <= 0:
            return 0
        count *= choices

    return count


def _candidate_weights(
    available: Sequence[int], position: int, max_shift: int, length: int
) -> tuple[int, ...]:
    """Return H=0 virtual-tail weights for candidates at ``position``."""
    if not available:
        return ()

    deadline = position - max_shift
    if available[0] < deadline:
        return tuple(0 for _ in available)

    item_count = len(available)
    newly_available = position + max_shift + 1
    child_size = item_count - 1 + (newly_available < length)
    last_slot = child_size - 1

    # If candidate j is removed, items before it retain their ranks while
    # items after it shift left by one.  Their factors therefore differ by
    # exactly one.  Two passes place the prefix and suffix products directly
    # into the output array, avoiding per-candidate lists and products.
    weights = [1] * item_count
    prefix = 1
    for rank, original_index in enumerate(available):
        weights[rank] = prefix
        item_deadline = min(
            last_slot, original_index + max_shift - (position + 1)
        )
        prefix *= max(0, item_deadline + 1 - rank)

    suffix = 1
    for rank in range(item_count - 1, -1, -1):
        weights[rank] *= suffix
        item_deadline = min(
            last_slot, available[rank] + max_shift - (position + 1)
        )
        suffix *= max(0, item_deadline + 2 - rank)

    if available[0] == deadline:
        weights[1:] = [0] * (item_count - 1)
    return tuple(weights)


class LocalShuffler(BaseBackgroundModel):
    """Shuffle a sequence while strictly bounding positional displacement.

    Parameters
    ----------
    max_shift : int
        Maximum number of positions by which any input item may move left or
        right.  Must be non-negative.
    seed : int or None, default None
        Seed for NumPy's Philox counter-based random generator.  If ``None``,
        entropy is obtained from the operating system.

    Notes
    -----
    Candidate probabilities use the H=0 virtual-tail approximation.  Complete
    valid permutations are therefore not exactly equiprobable, although the
    displacement guarantee is exact.
    """

    def __init__(self, max_shift: int, seed: int | None = None) -> None:
        if isinstance(max_shift, bool) or not isinstance(max_shift, Integral):
            raise TypeError("max_shift must be an integer")
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")

        self.max_shift = int(max_shift)
        self._rng = np.random.Generator(np.random.Philox(seed))

    def _randbelow(self, upper: int) -> int:
        """Draw uniformly from ``range(upper)`` for an arbitrary-size integer."""
        if upper <= 0:
            raise ValueError("upper must be positive")

        bits = upper.bit_length()
        words = (bits + 63) // 64
        mask = (1 << bits) - 1

        while True:
            value = 0
            for _ in range(words):
                value = (value << 64) | int(self._rng.bit_generator.random_raw())
            value &= mask
            if value < upper:
                return value

    def shuffle(self, seq: SeqLike) -> SeqLike:
        """Return a locally shuffled sequence of the same type and length."""
        if not isinstance(seq, (str, bytes)):
            raise TypeError("seq must be str or bytes")

        length = len(seq)
        if length < 2 or self.max_shift == 0:
            return seq

        max_shift = min(self.max_shift, length - 1)
        available = list(range(min(max_shift + 1, length)))
        output = []

        for position in range(length):
            weights = _candidate_weights(
                available, position, max_shift, length
            )
            total = sum(weights)
            if total <= 0:
                raise RuntimeError("local shuffle reached an invalid state")

            draw = self._randbelow(total)
            cumulative = 0
            selected_offset = 0
            for selected_offset, weight in enumerate(weights):
                cumulative += weight
                if draw < cumulative:
                    break

            original_index = available.pop(selected_offset)
            output.append(seq[original_index])

            newly_available = position + max_shift + 1
            if newly_available < length:
                available.append(newly_available)

        if isinstance(seq, bytes):
            return bytes(output)
        return "".join(output)

    def perturb(self, seq: SeqLike) -> SeqLike:
        """Alias for :meth:`shuffle`, implementing ``BaseBackgroundModel``."""
        return self.shuffle(seq)

    def getstate(self) -> dict:
        """Return a copy of the Philox generator state."""
        return deepcopy(self._rng.bit_generator.state)

    def setstate(self, state: dict) -> None:
        """Restore a state previously returned by :meth:`getstate`."""
        self._rng.bit_generator.state = deepcopy(state)
