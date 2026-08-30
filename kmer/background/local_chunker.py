"""Position-bounded permutation of pre-existing chunks.

``LocalChunker`` does not split or concatenate sequences.  It treats each item
in a supplied chunk collection as one indivisible unit and delegates chunk
order generation to :class:`LocalShuffler`.  Consequently, ``max_shift`` is
measured in chunk indices, while chunk contents and object identities are left
unchanged.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ._base import BaseBackgroundModel
from .local import LimitSpec, LocalShuffler

ChunkCollection: TypeAlias = list | tuple | np.ndarray


class LocalChunker(BaseBackgroundModel):
    """Locally permute a collection of pre-existing chunks.

    Parameters
    ----------
    max_shift : int
        Maximum displacement of each chunk, measured in chunk positions.
    seed : int or None, default None
        Seed for the native Philox4x32-10 random generator.  ``None`` uses
        operating-system entropy.
    L : int, {"auto", "H"}, or None, default "auto"
        Soft collision-chain limit delegated to :class:`LocalShuffler`.
    H : int, {"auto", "L"}, or None, default "auto"
        Hard augmenting-path limit delegated to :class:`LocalShuffler`.

    Notes
    -----
    Input must be a ``list``, ``tuple``, or one-dimensional NumPy array.  The
    outer container type is preserved.  Chunks are reordered by reference;
    they are never copied, split, reversed, or concatenated.
    """

    def __init__(
        self,
        max_shift: int,
        seed: int | None = None,
        L: LimitSpec = "auto",
        H: LimitSpec = "auto",
    ) -> None:
        self._shuffler = LocalShuffler(
            max_shift=max_shift,
            seed=seed,
            L=L,
            H=H,
        )

    @property
    def max_shift(self) -> int:
        """Maximum displacement measured in chunk positions."""
        return self._shuffler.max_shift

    @property
    def L(self) -> LimitSpec:
        """Configured soft collision-chain limit."""
        return self._shuffler.L

    @property
    def H(self) -> LimitSpec:
        """Configured hard augmenting-path limit."""
        return self._shuffler.H

    def limits_for(self, length: int) -> tuple[int | None, int | None]:
        """Resolve effective ``(L, H)`` for ``length`` chunks."""
        return self._shuffler.limits_for(length)

    def permutation(self, length: int) -> np.ndarray:
        """Return output-position to original-chunk-index mapping."""
        return self._shuffler.permutation(length)

    def shuffle_chunks(self, chunks: ChunkCollection) -> ChunkCollection:
        """Return a bounded permutation of pre-existing ``chunks``.

        Lists and tuples are rebuilt with the same chunk objects.  NumPy arrays
        retain their dtype and must be one-dimensional.
        """
        if isinstance(chunks, np.ndarray):
            if chunks.ndim != 1:
                raise ValueError("NumPy chunk collection must be one-dimensional")
            return chunks[self.permutation(len(chunks))]

        if isinstance(chunks, list):
            permutation = self.permutation(len(chunks))
            return [chunks[index] for index in permutation]

        if isinstance(chunks, tuple):
            permutation = self.permutation(len(chunks))
            return tuple(chunks[index] for index in permutation)

        raise TypeError(
            "chunks must be a list, tuple, or one-dimensional NumPy array"
        )

    def shuffle(self, chunks: ChunkCollection) -> ChunkCollection:
        """Alias for :meth:`shuffle_chunks`."""
        return self.shuffle_chunks(chunks)

    def perturb(self, chunks: ChunkCollection) -> ChunkCollection:
        """Alias for :meth:`shuffle_chunks` for background-model use."""
        return self.shuffle_chunks(chunks)

    def getstate(self) -> bytes:
        """Return the complete native Philox RNG state."""
        return self._shuffler.getstate()

    def setstate(self, state: bytes) -> None:
        """Restore a state returned by :meth:`getstate`."""
        self._shuffler.setstate(state)

    @property
    def last_stats_(self) -> dict[str, int]:
        """Collision statistics for the most recent chunk permutation."""
        return self._shuffler.last_stats_

    def __repr__(self) -> str:
        return (
            f"LocalChunker(max_shift={self.max_shift}, seed=<state>, "
            f"L={self.L!r}, H={self.H!r})"
        )
