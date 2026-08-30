"""kmer.models — sklearn-compatible estimators for sequence classification.

Classes
-------
- :class:`KmerMultinomialNB` — Multinomial NB on k-mer features,
  with auto-generated negatives via Shuffler/Chunker.
- :class:`KernelSVM`  — SVM with a precomputed sequence kernel.
- :class:`LinearSVM`  — Linear SVM on encoder features.
- :class:`KNNClassifier` — k-Nearest Neighbors with a sequence distance.
- :class:`KmerLinearRegression` — linear regression on encoded k-mer features.
"""

from .differential import KmerMultinomialNB
from .svm import KernelSVM, LinearSVM
from .knn import KNNClassifier
from .regression import KmerLinearRegression

__all__ = [
    "KmerMultinomialNB",
    "KernelSVM",
    "LinearSVM",
    "KNNClassifier",
    "KmerLinearRegression",
]
