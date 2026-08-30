"""kmer.perturb — sequence disruption for negative-set generation.

Provides:
- :class:`BaseBackgroundModel` — abstract interface for background models.
- :class:`KmerShuffler` — preserves exact k-mer composition via random
  Eulerian paths in the De Bruijn graph.
- :class:`Chunker` — splits a sequence into chunks of a given size range,
  shuffles them, and concatenates back.
- :class:`LocalShuffler` — approximately samples local permutations while
  strictly bounding each item's positional displacement.
- :class:`LocalChunker` — applies the same bounded permutation to collections
  of pre-existing chunks without changing chunk contents.

All background models implement :class:`BaseBackgroundModel`, which
exposes ``perturb(seq)`` and ``perturb_many(seqs)``.

The permutation engines are C-backed and use Philox4×32-10 RNG streams for
seeded reproducibility. ``LocalChunker`` composes the native ``LocalShuffler``
engine without duplicating it.
"""

from ._base import BaseBackgroundModel
from .shuffler import KmerShuffler
from .chunker import Chunker
from .local import LocalShuffler
from .local_chunker import LocalChunker

__all__ = [
    "BaseBackgroundModel",
    "KmerShuffler",
    "Chunker",
    "LocalShuffler",
    "LocalChunker",
]
