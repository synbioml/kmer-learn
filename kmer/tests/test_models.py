"""Tests for the new models: KernelSVM, LinearSVM, KNNClassifier."""

import pytest
import numpy as np

from kmer.encoders import SpectrumEncoder, MismatchEncoder
from kmer.kernels import GKMKernel
from kmer.distance import Levenshtein, DistanceKernel
from kmer.models import KernelSVM, LinearSVM, KNNClassifier


@pytest.fixture
def toy_dataset():
    """Binary classification: ACGT-repeat (positive) vs AT-repeat (negative)."""
    np.random.seed(0)
    positives = ["ACGTACGT" + "ACGT" * np.random.randint(0, 3) for _ in range(20)]
    negatives = ["ATATATAT" + "ATAT" * np.random.randint(0, 3) for _ in range(20)]
    seqs = positives + negatives
    y = np.array([1] * 20 + [0] * 20)
    return seqs, y


# ======================================================================
# KernelSVM
# ======================================================================

class TestKernelSVM:
    def test_with_gkm_kernel(self, toy_dataset):
        seqs, y = toy_dataset
        clf = KernelSVM(
            GKMKernel(L=6, k=4, d=1, kernel_type="truncated", use_rc=True),
            C=1.0,
        )
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_with_distance_kernel(self, toy_dataset):
        seqs, y = toy_dataset
        kern = DistanceKernel(Levenshtein(), transform="rbf", gamma=0.1)
        clf = KernelSVM(kern, C=1.0)
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)

    def test_decision_function(self, toy_dataset):
        seqs, y = toy_dataset
        clf = KernelSVM(
            GKMKernel(L=6, k=4, d=1, kernel_type="truncated", use_rc=True),
            C=1.0,
        )
        clf.fit(seqs, y)
        scores = clf.decision_function(seqs)
        assert scores.shape == (40,)

    def test_achieves_perfect_train_accuracy(self, toy_dataset):
        """On the easy toy dataset, all models should achieve 100% train acc."""
        seqs, y = toy_dataset
        clf = KernelSVM(
            GKMKernel(L=6, k=4, d=1, kernel_type="truncated", use_rc=True),
            C=1.0,
        )
        clf.fit(seqs, y)
        acc = (clf.predict(seqs) == y).mean()
        assert acc == 1.0


# ======================================================================
# LinearSVM
# ======================================================================

class TestLinearSVM:
    def test_basic_fit_predict(self, toy_dataset):
        seqs, y = toy_dataset
        clf = LinearSVM(SpectrumEncoder(k=4), C=1.0)
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)

    def test_with_mismatch_encoder(self, toy_dataset):
        seqs, y = toy_dataset
        clf = LinearSVM(MismatchEncoder(k=4, m=1), C=1.0)
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)

    def test_standardize(self, toy_dataset):
        seqs, y = toy_dataset
        clf = LinearSVM(SpectrumEncoder(k=4), C=1.0, standardize=True)
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)

    def test_coef_attribute(self, toy_dataset):
        seqs, y = toy_dataset
        clf = LinearSVM(SpectrumEncoder(k=4), C=1.0)
        clf.fit(seqs, y)
        assert clf.coef_.shape == (1, 256)  # 4^4 features

    def test_achieves_perfect_train_accuracy(self, toy_dataset):
        seqs, y = toy_dataset
        clf = LinearSVM(SpectrumEncoder(k=4), C=1.0)
        clf.fit(seqs, y)
        assert (clf.predict(seqs) == y).mean() == 1.0


# ======================================================================
# KNNClassifier
# ======================================================================

class TestKNNClassifier:
    def test_basic_fit_predict(self, toy_dataset):
        seqs, y = toy_dataset
        clf = KNNClassifier(Levenshtein(), n_neighbors=3)
        clf.fit(seqs, y)
        preds = clf.predict(seqs)
        assert preds.shape == (40,)

    def test_with_hamming(self, toy_dataset):
        from kmer.distance import Hamming
        # Need equal-length sequences for Hamming; pad shorter ones.
        seqs, y = toy_dataset
        max_len = max(len(s) for s in seqs)
        seqs_pad = [s.ljust(max_len, "A") for s in seqs]
        clf = KNNClassifier(Hamming(), n_neighbors=3)
        clf.fit(seqs_pad, y)
        preds = clf.predict(seqs_pad)
        assert preds.shape == (40,)

    def test_predict_proba(self, toy_dataset):
        seqs, y = toy_dataset
        clf = KNNClassifier(Levenshtein(), n_neighbors=5, weights="distance")
        clf.fit(seqs, y)
        proba = clf.predict_proba(seqs)
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
