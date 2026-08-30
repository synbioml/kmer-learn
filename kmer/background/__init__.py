"""kmer.background — sequence disruption for negative-set generation.

Provides:
- :class:`BaseBackgroundModel` — abstract interface for background models.
- :class:`KmerShuffler` — preserves exact k-mer composition via random
  Eulerian paths in the De Bruijn graph.
- :class:`Chunker` — splits a sequence into chunks of a given size range,
  shuffles them, and concatenates back.
- :class:`LocalShuffler` — samples local permutations of strings, arrays, or
  Python collections while strictly bounding item displacement.
- :class:`LocalChunker` — composes injected ``Chunker`` and ``LocalShuffler``
  objects to sample and locally reorder variable-size sequence chunks.

All background models implement :class:`BaseBackgroundModel`, which
exposes ``perturb(seq)`` and ``perturb_many(seqs)``.

The permutation engines are C-backed and use Philox4×32-10 RNG streams for
seeded reproducibility. Batch items receive independent draws from advancing
streams. ``LocalChunker`` composes the existing chunking and local-shuffling
APIs without duplicating either algorithm.
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
