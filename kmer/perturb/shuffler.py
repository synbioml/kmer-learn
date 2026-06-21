"""K-mer-preserving sequence shuffler.

Builds the De Bruijn multigraph of the input sequence (nodes = (k-1)-mers,
edges = k-mers, each occurrence a distinct edge) and finds a random
Eulerian path through it via Hierholzer's algorithm with random edge
selection.

Endpoint modes
--------------
``endpoints="preserve"``
    Start the Eulerian path from the original odd-degree vertex. Both
    endpoints match the input. Exact k-mer composition preserved.

``endpoints="free"`` (default)
    Close the graph into a circuit by adding a virtual wrap-around
    edge, find an Eulerian circuit, and break it at a random edge.
    Both endpoints can change. k-mer composition is exact plus one
    wrap-around k-mer and minus the would-be-wrap k-mer.

``endpoints="crop"``
    Drop the first and last (k-1)-mer, shuffle the interior as a
    circuit. Output length = len(seq) - 2(k-1). Exact k-mer composition
    of the interior preserved.

RNG
---
Philox4×32-10 (counter-based). State lives on the shuffler instance and
advances on every ``perturb()`` call, so successive calls produce
independent outputs. For ``perturb_many()``, each task gets an
independent key derived from the master seed + task index — output is
bit-for-bit identical regardless of ``n_jobs``.

Examples
--------
>>> from kmer.perturb import KmerShuffler
>>> sh = KmerShuffler(k=3, seed=42)
>>> sh.perturb("ACGTACGTACGT")
'GACGTACGTACG'  # exact trinucleotide composition preserved

>>> sh.perturb_many(["ACGTACGT", "TTTTAAAA"], n_jobs=2)
['...', '...']  # reproducible regardless of n_jobs
"""

from __future__ import annotations

from ._base import BaseBackgroundModel
from ._native._shuffler import KmerShuffler as _KmerShuffler


class KmerShuffler(_KmerShuffler, BaseBackgroundModel):
    """K-mer-preserving sequence shuffler.

    Parameters
    ----------
    k : int
        K-mer size whose composition is preserved. Must be in [1, 12].
        k=1 gives mononucleotide shuffle; k=2 matches Clote's
        dinucleotide-preserving shuffle.
    seed : int or None, default None
        Master seed for the Philox4×32-10 RNG. If None, OS entropy is
        used and output is non-reproducible.
    endpoints : {"preserve", "free", "crop"}, default "free"
        Endpoint handling mode. See module docstring.

    Notes
    -----
    Implements :class:`BaseBackgroundModel`. The C extension provides
    ``shuffle(seq)`` and ``shuffle_many(seqs)``; this class exposes
    ``perturb`` / ``perturb_many`` (the BaseBackgroundModel interface)
    which delegate to the C methods.

    The internal De Bruijn graph uses (k-1)-mer nodes (the natural
    representation); the user-facing parameter is ``k`` (the size whose
    composition is preserved), matching the convention in the k-mer
    literature.

    Non-ACGT characters raise ``ValueError`` at shuffle time. For
    IUPAC-aware handling, use the chunker instead.

    See Also
    --------
    kmer.perturb.Chunker : block-level sequence perturbation.
    kmer.perturb.BaseBackgroundModel : abstract interface.
    """

    # The C extension implements shuffle() and shuffle_many().
    # perturb / perturb_many are the BaseBackgroundModel interface.
    def perturb(self, seq):
        return self.shuffle(seq)

    def perturb_many(self, seqs):
        return self.shuffle_many(seqs)
