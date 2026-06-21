"""kmer.perturb — sequence disruption for negative-set generation.

Provides:
- :class:`BaseBackgroundModel` — abstract interface for background models.
- :class:`KmerShuffler` — preserves exact k-mer composition via random
  Eulerian paths in the De Bruijn graph.
- :class:`Chunker` — splits a sequence into chunks of a given size range,
  shuffles them, and concatenates back.

All background models implement :class:`BaseBackgroundModel`, which
exposes ``perturb(seq)`` and ``perturb_many(seqs)``.

C-backed (Philox4×32-10 RNG); reproducible when a seed is provided,
regardless of n_jobs.
"""

from ._base import BaseBackgroundModel
from .shuffler import KmerShuffler
from .chunker import Chunker

__all__ = ["BaseBackgroundModel", "KmerShuffler", "Chunker"]
