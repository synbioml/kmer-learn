"""kmer — classical machine learning primitives for nucleotide sequences.

Subpackages
-----------
- :mod:`kmer.kernels`  — sequence kernels (gkm family).
- :mod:`kmer.encoders` — k-mer feature extractors (spectrum, gappy, mismatch).
- :mod:`kmer.distance` — pairwise distance metrics and distance-based kernels.
- :mod:`kmer.models`   — sklearn-compatible estimators (SVM, NB, KNN).
- :mod:`kmer.perturb`  — sequence disruption (shuffler, chunker).
- :mod:`kmer.utils`    — bit-packed k-mer helpers.

Quick start
-----------
>>> from kmer.kernels import GKMKernel
>>> from kmer.models import KernelSVM
>>> clf = KernelSVM(GKMKernel(L=10, k=6, d=3), C=1.0)
"""

__version__ = "0.1.0"
