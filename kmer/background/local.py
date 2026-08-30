"""Position-bounded local sequence shuffling.

``LocalShuffler`` inserts items in random order into legal destination windows.
An item samples uniformly from every position within ``max_shift`` of its
original index.  On collision it takes the position and the previous occupant
continues the chain.  Long chains switch to free-first sampling at ``L`` and
use an augmenting-path fallback at ``H``.

The native implementation has expected ``O(n * E[T])`` running time, where
``T`` is the number of position draws per inserted item.  Measurements show a
small, nearly length-independent ``E[T]`` for practical shift bounds.  The
strict displacement bound is always preserved.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np

from ._base import BaseBackgroundModel, SeqLike, _validate_n_jobs
from ._native._local import _LocalShuffler

LimitSpec: TypeAlias = int | str | None


def _validate_limit_spec(value: LimitSpec, name: str, alias: str) -> LimitSpec:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer, 'auto', {alias!r}, or None")
    if isinstance(value, Integral):
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value
    if isinstance(value, str) and value in {"auto", alias}:
        return value
    raise TypeError(
        f"{name} must be a non-negative integer, 'auto', {alias!r}, or None"
    )


class LocalShuffler(_LocalShuffler, BaseBackgroundModel):
    """Shuffle items with a strict maximum positional displacement.

    Parameters
    ----------
    max_shift : int
        Maximum number of positions by which an item may move left or right.
    seed : int or None, default None
        Seed for the native Philox4x32-10 random generator.
    L : int, {"auto", "H"}, or None, default "auto"
        Soft collision limit.  At this many collisions, the current chain
        chooses a legal free position whenever one exists.  ``"H"`` copies
        the resolved hard limit; ``None`` disables the soft phase.
    H : int, {"auto", "L"}, or None, default "auto"
        Hard collision limit.  At this many collisions, an augmenting-path
        search completes the insertion.  ``"L"`` copies the resolved soft
        limit; ``None`` disables the hard fallback.

    Notes
    -----
    Automatic limits depend on input length ``n`` and effective shift ``S``::

        L = max(16, 2*S, 4*ceil(log2(n + 1)))
        H = max(1024, 4*L_auto)

    If resolved ``L >= H``, the hard fallback runs first and the soft phase is
    effectively disabled.  Both limits may be disabled explicitly.

    Inputs may be ``str``, ``bytes``, ``list``, ``tuple``, or one-dimensional
    NumPy arrays. Output type, length, values, object identities, and NumPy
    dtype are preserved. Batch inputs are sampled independently from the
    advancing RNG stream; coincidental duplicate permutations remain possible.
    """

    def __init__(
        self,
        max_shift: int,
        seed: int | None = None,
        L: LimitSpec = "auto",
        H: LimitSpec = "auto",
    ) -> None:
        if isinstance(max_shift, bool) or not isinstance(max_shift, Integral):
            raise TypeError("max_shift must be an integer")
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")

        self.max_shift = int(max_shift)
        self.L = _validate_limit_spec(L, "L", "H")
        self.H = _validate_limit_spec(H, "H", "L")
        if self.L == "H" and self.H == "L":
            raise ValueError("L='H' and H='L' form a circular alias")
        super().__init__(self.max_shift, seed)

    def limits_for(self, length: int) -> tuple[int | None, int | None]:
        """Resolve effective ``(L, H)`` limits for an input length."""
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise TypeError("length must be an integer")
        length = int(length)
        if length < 0:
            raise ValueError("length must be non-negative")

        effective_shift = min(self.max_shift, max(0, length - 1))
        log_term = length.bit_length() if length else 0
        automatic_l = max(16, 2 * effective_shift, 4 * log_term)
        automatic_h = max(1024, 4 * automatic_l)

        resolved_l = automatic_l if self.L == "auto" else self.L
        resolved_h = automatic_h if self.H == "auto" else self.H
        if resolved_l == "H":
            resolved_l = resolved_h
        if resolved_h == "L":
            resolved_h = resolved_l
        return resolved_l, resolved_h

    def permutation(self, length: int) -> np.ndarray:
        """Return output-slot to original-index mapping for ``length`` items."""
        L, H = self.limits_for(length)
        raw = self._permutation_bytes(int(length), L, H)
        return np.frombuffer(raw, dtype=np.intp).copy()

    def shuffle(self, seq: SeqLike | list | tuple | np.ndarray):
        """Return a bounded shuffle with the same type and length as ``seq``."""
        L, H = self.limits_for(len(seq))
        if isinstance(seq, np.ndarray):
            if seq.ndim != 1:
                raise ValueError("NumPy input must be one-dimensional")
            raw = self._permutation_bytes(len(seq), L, H)
            indices = np.frombuffer(raw, dtype=np.intp)
            return seq[indices]
        if isinstance(seq, list):
            indices = self.permutation(len(seq))
            return [seq[index] for index in indices]
        if isinstance(seq, tuple):
            indices = self.permutation(len(seq))
            return tuple(seq[index] for index in indices)
        if not isinstance(seq, (str, bytes)):
            raise TypeError(
                "seq must be str, bytes, list, tuple, or a one-dimensional "
                "NumPy array"
            )
        return self._shuffle(seq, L, H)

    def perturb(self, seq: SeqLike | list | tuple | np.ndarray):
        """Alias for :meth:`shuffle`, implementing ``BaseBackgroundModel``."""
        return self.shuffle(seq)

    def perturb_many(self, seqs, n_jobs: int = 1) -> list:
        """Independently shuffle every input in ``seqs``.

        Local shuffles currently run serially, so seeded output does not depend
        on the requested worker count.
        """
        _validate_n_jobs(n_jobs)
        return [self.shuffle(seq) for seq in seqs]

    @property
    def last_stats_(self) -> dict[str, int]:
        """Collision statistics for the most recent shuffle."""
        return self._get_last_stats()

    def __repr__(self) -> str:
        return (
            f"LocalShuffler(max_shift={self.max_shift}, seed=<state>, "
            f"L={self.L!r}, H={self.H!r})"
        )
