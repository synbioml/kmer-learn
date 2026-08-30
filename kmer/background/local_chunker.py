"""Locally bounded shuffling of variable-size sequence chunks.

``LocalChunker`` composes a :class:`Chunker`, which samples and slices chunks,
with a :class:`LocalShuffler`, which reorders those chunks under a strict
position bound. Chunk sizes and counts are sampled independently for each
input sequence.
"""

from __future__ import annotations

from ._base import BaseBackgroundModel, SeqLike, _validate_n_jobs
from .chunker import Chunker
from .local import LocalShuffler


class LocalChunker(BaseBackgroundModel):
    """Split a sequence and locally reorder its chunks.

    Parameters
    ----------
    chunker : Chunker
        Configured chunk sampler. Its ``slice_chunks`` method determines the
        chunk boundaries and optional strand flips without reordering chunks.
    shuffler : LocalShuffler
        Configured local permutation model. ``max_shift`` is measured in chunk
        positions, not sequence positions.

    Notes
    -----
    The two injected objects retain and advance their own RNG states. Every
    sequence in a batch receives newly sampled chunk boundaries and a newly
    sampled local permutation. Independent draws can coincide by chance.
    """

    def __init__(self, chunker: Chunker, shuffler: LocalShuffler) -> None:
        if not isinstance(chunker, Chunker):
            raise TypeError("chunker must be a Chunker instance")
        if not isinstance(shuffler, LocalShuffler):
            raise TypeError("shuffler must be a LocalShuffler instance")
        self.chunker = chunker
        self.shuffler = shuffler

    @property
    def max_shift(self) -> int:
        """Maximum displacement measured in chunk positions."""
        return self.shuffler.max_shift

    def slice_chunks(self, seq: SeqLike) -> list:
        """Sample chunks without reordering them."""
        return self.chunker.slice_chunks(seq)

    def chunk(self, seq: SeqLike) -> SeqLike:
        """Split and locally reorder one string or byte sequence."""
        chunks = self.slice_chunks(seq)
        shuffled = self.shuffler.shuffle(chunks)
        separator = b"" if isinstance(seq, bytes) else ""
        return separator.join(shuffled)

    def perturb(self, seq: SeqLike) -> SeqLike:
        """Alias for :meth:`chunk`, implementing ``BaseBackgroundModel``."""
        return self.chunk(seq)

    def perturb_many(self, seqs, n_jobs: int = 1) -> list:
        """Independently split and locally reorder every input sequence.

        The composed operation currently runs serially, so seeded output does
        not depend on the requested worker count.
        """
        _validate_n_jobs(n_jobs)
        return [self.chunk(seq) for seq in seqs]

    def getstate(self) -> tuple[bytes, bytes]:
        """Return the RNG states of the chunker and local shuffler."""
        return self.chunker.getstate(), self.shuffler.getstate()

    def setstate(self, state: tuple[bytes, bytes]) -> None:
        """Restore a state returned by :meth:`getstate`."""
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("state must be a (chunker_state, shuffler_state) tuple")
        self.chunker.setstate(state[0])
        self.shuffler.setstate(state[1])

    @property
    def last_stats_(self) -> dict[str, int]:
        """Collision statistics for the most recent chunk permutation."""
        return self.shuffler.last_stats_

    def __repr__(self) -> str:
        return f"LocalChunker(chunker={self.chunker!r}, shuffler={self.shuffler!r})"
