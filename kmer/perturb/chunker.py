"""Sequence chunker for block-level perturbation.

Splits a sequence into chunks whose sizes are drawn from
``[min_size, max_size]``, optionally reverse-complements each chunk
(IUPAC), shuffles the chunk order, and concatenates back.

Output length always equals input length.

Algorithms
----------
``algorithm="random"`` (default)
    Greedy: pick the next chunk size uniformly from
    ``[min_size, min(max_size, remaining)]``. If the remaining tail is
    shorter than ``min_size``, apply ``residual_chunk``.

``algorithm="backtrack"``
    Use dynamic programming to find an exact decomposition of
    ``len(seq)`` into sizes from ``[min_size, max_size]``. If multiple
    decompositions exist, sample one uniformly at random (using the
    DP counts as a distribution). If no exact decomposition exists,
    make the tail as short as possible and apply ``residual_chunk``.

Residual handling
-----------------
When a tail too short for a valid chunk remains, ``residual_chunk``
determines how it's absorbed:

``"end"``      — keep as final, undersized chunk.
``"start"``    — keep as leading, undersized chunk.
``"random"``   — insert at a random position in the size list.
``"extend"``   — pick a random existing chunk, extend by residual.
``"distribute"`` (default) — for each residual unit, pick a random
                  chunk (with replacement) and extend by 1.

Strand flipping
---------------
If ``flip_strand=True``, each chunk is independently reverse-complemented
with probability ``flip_strand_prob`` (default 1.0). The chunker uses
the IUPAC alphabet for RC; non-IUPAC characters raise ``ValueError``.
With ``flip_strand=False``, the chunker accepts any byte sequence
without validation.

RNG
---
Philox4×32-10, same as :class:`KmerShuffler`.

Examples
--------
>>> from kmer.perturb import Chunker
>>> ch = Chunker(min_size=2, max_size=4, seed=42)
>>> ch.perturb("ACGTACGTACGT")
'GTACACGTACGT'  # same length, blocks shuffled
"""

from __future__ import annotations

from ._base import BaseBackgroundModel
from ._native._chunker import Chunker as _Chunker


class Chunker(_Chunker, BaseBackgroundModel):
    """Sequence chunker.

    Parameters
    ----------
    min_size, max_size : int
        Chunk size range. Require ``1 <= min_size <= max_size``.
    algorithm : {"random", "backtrack"}, default "random"
    residual_chunk : {"end", "start", "random", "extend", "distribute"},
                     default "distribute"
    flip_strand : bool, default False
        If True, each chunk is independently reverse-complemented with
        probability ``flip_strand_prob``. Requires IUPAC alphabet.
    flip_strand_prob : float, default 0.5
        Probability that any given chunk is reverse-complemented.
    seed : int or None, default None
        Master seed for Philox4×32-10.

    Notes
    -----
    Implements :class:`BaseBackgroundModel`. The C extension provides
    ``chunk(seq)`` and ``chunk_many(seqs)``; this class exposes
    ``perturb`` / ``perturb_many`` (the BaseBackgroundModel interface)
    which delegate to the C methods.

    See Also
    --------
    kmer.perturb.KmerShuffler : k-mer-preserving shuffling.
    kmer.perturb.BaseBackgroundModel : abstract interface.
    """

    # The C extension implements chunk() and chunk_many().
    # perturb / perturb_many are the BaseBackgroundModel interface.
    def perturb(self, seq):
        return self.chunk(seq)

    def perturb_many(self, seqs):
        return self.chunk_many(seqs)
