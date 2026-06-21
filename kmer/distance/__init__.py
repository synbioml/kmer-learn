"""kmer.distance — pairwise sequence distances and distance-based kernels.

Classes
-------
- :class:`BaseDistance`    — abstract interface for pairwise metrics.
- :class:`DistanceKernel`  — turns any distance into a kernel via a
                             post-transform (rbf, laplacian, etc.).
- :class:`Hamming`         — equal-length position-mismatch count.
- :class:`Levenshtein`     — edit distance (rapidfuzz backend).
- :class:`NeedlemanWunsch` — global alignment score (parasail backend).
- :class:`SmithWaterman`   — local alignment score (parasail backend).
"""

from ._base import BaseDistance, DistanceKernel
from .edit import Hamming, Levenshtein
from .alignment import NeedlemanWunsch, SmithWaterman

__all__ = [
    "BaseDistance",
    "DistanceKernel",
    "Hamming",
    "Levenshtein",
    "NeedlemanWunsch",
    "SmithWaterman",
]
