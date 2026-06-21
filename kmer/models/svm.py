"""kmer.models.svm — SVM wrappers for kernels and encoders."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler


class KernelSVM(BaseEstimator, ClassifierMixin):
    """SVM with a precomputed sequence kernel.

    Wraps :class:`sklearn.svm.SVC` with ``kernel='precomputed'``. The
    kernel object must provide a ``kernel_matrix(X, Y=None) -> np.ndarray``
    method (e.g., :class:`kmer.distance.DistanceKernel`), OR follow the
    gkm convention of ``set_references(X)`` + ``kernel(X_query=Y)``
    (e.g., :class:`kmer.kernels.GKMKernel`).

    Parameters
    ----------
    kernel : object
        Kernel with either a ``kernel_matrix(X, Y=None)`` method or the
        gkm-style ``set_references``/``kernel`` API.
    C : float, default 1.0
        SVM regularization parameter.
    probability : bool, default False
        Whether to enable probability estimates (slower training).

    Examples
    --------
    >>> from kmer.kernels import GKMKernel
    >>> from kmer.models import KernelSVM
    >>> clf = KernelSVM(GKMKernel(L=10, k=6, d=3), C=1.0)
    >>> clf.fit(train_seqs, y_train)
    >>> preds = clf.predict(test_seqs)
    """

    def __init__(self, kernel, C: float = 1.0, probability: bool = False):
        self.kernel = kernel
        self.C = C
        self.probability = probability

    def _kernel_matrix(self, X, Y=None):
        """Compute the kernel matrix, supporting both kernel APIs."""
        if hasattr(self.kernel, "kernel_matrix"):
            return self.kernel.kernel_matrix(X, Y)
        # gkm-style API: set_references + kernel(X_query=...)
        if Y is None:
            self.kernel.set_references(list(X))
            return np.asarray(self.kernel.kernel())
        else:
            self.kernel.set_references(list(Y))
            return np.asarray(self.kernel.kernel(X_query=list(X)))

    def fit(self, seqs, y):
        self.classes_ = np.unique(y)
        self._train_seqs_ = list(seqs)
        K = self._kernel_matrix(self._train_seqs_)
        import warnings
        with warnings.catch_warnings():
            # SVC(probability=...) is deprecated in newer sklearn but the
            # alternative (CalibratedClassifierCV) doesn't support precomputed
            # kernels. Suppress the warning until sklearn resolves this.
            warnings.simplefilter("ignore", FutureWarning)
            self._svc_ = SVC(C=self.C, kernel="precomputed",
                             probability=self.probability)
            self._svc_.fit(K, y)
        return self

    def predict(self, seqs):
        K = self._kernel_matrix(seqs, self._train_seqs_)
        return self._svc_.predict(K)

    def predict_proba(self, seqs):
        if not self.probability:
            raise RuntimeError("probability=True required at construction")
        K = self._kernel_matrix(seqs, self._train_seqs_)
        return self._svc_.predict_proba(K)

    def decision_function(self, seqs):
        K = self._kernel_matrix(seqs, self._train_seqs_)
        return self._svc_.decision_function(K)


class LinearSVM(BaseEstimator, ClassifierMixin):
    """Linear SVM on encoder features.

    Wraps :class:`sklearn.svm.LinearSVC`. Works with any encoder that
    produces a sparse CSR matrix (e.g., :class:`SpectrumEncoder`,
    :class:`GappyEncoder`, :class:`MismatchEncoder`).

    Parameters
    ----------
    encoder : object
        Encoder with ``fit_transform(seqs)`` and ``transform(seqs)``
        methods returning a scipy.sparse matrix.
    C : float, default 1.0
        SVM regularization parameter.
    standardize : bool, default False
        If True, standardize features (zero mean, unit variance) before
        fitting the SVM. Recommended for encoders with very different
        k-mer count scales.

    Examples
    --------
    >>> from kmer.encoders import SpectrumEncoder
    >>> from kmer.models import LinearSVM
    >>> clf = LinearSVM(SpectrumEncoder(k=6), C=1.0)
    >>> clf.fit(train_seqs, y_train)
    >>> preds = clf.predict(test_seqs)
    """

    def __init__(self, encoder, C: float = 1.0, standardize: bool = False):
        self.encoder = encoder
        self.C = C
        self.standardize = standardize

    def fit(self, seqs, y):
        self.classes_ = np.unique(y)
        X = self.encoder.fit_transform(seqs)
        if self.standardize:
            self._scaler_ = StandardScaler(with_mean=False)
            X = self._scaler_.fit_transform(X)
        self._clf_ = LinearSVC(C=self.C, dual="auto")
        self._clf_.fit(X, y)
        return self

    def predict(self, seqs):
        X = self.encoder.transform(seqs)
        if self.standardize:
            X = self._scaler_.transform(X)
        return self._clf_.predict(X)

    def decision_function(self, seqs):
        X = self.encoder.transform(seqs)
        if self.standardize:
            X = self._scaler_.transform(X)
        return self._clf_.decision_function(X)

    @property
    def coef_(self):
        """Linear coefficients (one per feature)."""
        return self._clf_.coef_
