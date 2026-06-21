"""kmer.models — sklearn-compatible estimators for sequence classification.

Classes
-------
- :class:`DifferentialKmerScorer` — Multinomial NB on k-mer features,
  with auto-generated negatives via Shuffler/Chunker.
- :class:`KernelSVM`  — SVM with a precomputed sequence kernel.
- :class:`LinearSVM`  — Linear SVM on encoder features.
- :class:`KNNClassifier` — k-Nearest Neighbors with a sequence distance.
"""

from .differential import DifferentialKmerScorer
from .svm import KernelSVM, LinearSVM
from .knn import KNNClassifier

__all__ = [
    "DifferentialKmerScorer",
    "KernelSVM",
    "LinearSVM",
    "KNNClassifier",
]
