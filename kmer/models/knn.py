"""kmer.models.knn — k-Nearest Neighbors with sequence distances."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """k-Nearest Neighbors classifier with a sequence distance metric.

    Wraps :class:`sklearn.neighbors.KNeighborsClassifier` with
    ``metric='precomputed'``. The distance object must provide a
    ``cdist(X, Y)`` method returning a 2D numpy array of distances.

    Parameters
    ----------
    distance : object
        Distance with a ``cdist(X, Y=None) -> np.ndarray`` method.
        Must be a proper metric (satisfy triangle inequality) for
        meaningful KNN. Examples: :class:`kmer.distance.Hamming`,
        :class:`kmer.distance.Levenshtein`.
    n_neighbors : int, default 5
        Number of neighbors to use.
    weights : {"uniform", "distance"}, default "uniform"
        Weight function used in prediction.

    Examples
    --------
    >>> from kmer.distance import Levenshtein
    >>> from kmer.models import KNNClassifier
    >>> clf = KNNClassifier(Levenshtein(), n_neighbors=3)
    >>> clf.fit(train_seqs, y_train)
    >>> preds = clf.predict(test_seqs)
    """

    def __init__(self, distance, n_neighbors: int = 5,
                 weights: str = "uniform"):
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, seqs, y):
        self.classes_ = np.unique(y)
        self._train_seqs_ = list(seqs)
        self._knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric="precomputed",
        )
        # KNN.fit with precomputed metric requires a distance matrix
        # from training samples to themselves.
        D = self.distance.cdist(self._train_seqs_)
        self._knn_.fit(D, y)
        return self

    def predict(self, seqs):
        D = self.distance.cdist(seqs, self._train_seqs_)
        return self._knn_.predict(D)

    def predict_proba(self, seqs):
        D = self.distance.cdist(seqs, self._train_seqs_)
        return self._knn_.predict_proba(D)
