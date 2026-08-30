"""Tests for KmerLinearRegression."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from kmer.encoders import SpectrumEncoder
from kmer.models import KmerLinearRegression


@pytest.fixture
def regression_dataset():
    seqs = [
        "AAAA",
        "AAAACC",
        "AACCCCCC",
        "CCCCGG",
        "GGGGGGGG",
        "TTTT",
        "ACGTAC",
        "CGCGCGCG",
    ]
    y = np.array([
        2.0 * seq.count("A")
        + 0.5 * seq.count("C")
        - 1.0 * seq.count("G")
        + 1.5
        for seq in seqs
    ])
    return seqs, y


class TestKmerLinearRegression:
    def test_basic_fit_predict(self, regression_dataset):
        seqs, y = regression_dataset
        model = KmerLinearRegression(SpectrumEncoder(k=1))
        returned = model.fit(seqs, y)

        assert returned is model
        assert np.allclose(model.predict(seqs), y)
        assert model.coef_.shape == (4,)
        assert np.ndim(model.intercept_) == 0

    def test_standardize_without_centering(self, regression_dataset):
        seqs, y = regression_dataset
        model = KmerLinearRegression(
            SpectrumEncoder(k=1),
            standardize=True,
        ).fit(seqs, y)

        assert model.scaler_.with_mean is False
        scaled = model.scaler_.transform(model.encoder.transform(seqs))
        assert sp.issparse(scaled)
        assert np.allclose(model.predict(seqs), y)

    def test_standardize_with_centering_densifies(self, regression_dataset):
        seqs, y = regression_dataset
        model = KmerLinearRegression(
            SpectrumEncoder(k=1),
            standardize=True,
            with_mean=True,
        ).fit(seqs, y)

        assert model.scaler_.with_mean is True
        assert np.allclose(model.predict(seqs), y)

    def test_sample_weight(self, regression_dataset):
        seqs, y = regression_dataset
        weights = np.linspace(1.0, 2.0, len(seqs))
        model = KmerLinearRegression(SpectrumEncoder(k=1))
        model.fit(seqs, y, sample_weight=weights)

        assert model.predict(seqs).shape == y.shape

    def test_kwargs_forwarded_to_linear_regression(self, regression_dataset):
        seqs, y = regression_dataset
        model = KmerLinearRegression(
            SpectrumEncoder(k=1),
            fit_intercept=False,
            copy_X=False,
            n_jobs=1,
        ).fit(seqs, y)

        params = model.regressor_.get_params()
        assert params["fit_intercept"] is False
        assert params["copy_X"] is False
        assert params["n_jobs"] == 1

    def test_clone_preserves_forwarded_kwargs(self):
        model = KmerLinearRegression(
            SpectrumEncoder(k=2),
            standardize=True,
            with_mean=True,
            fit_intercept=False,
            n_jobs=1,
        )
        cloned = clone(model)

        params = cloned.get_params(deep=False)
        assert params["standardize"] is True
        assert params["with_mean"] is True
        assert params["fit_intercept"] is False
        assert params["n_jobs"] == 1
        assert cloned.encoder is not model.encoder

    def test_set_params_updates_wrapper_and_regressor_kwargs(self):
        model = KmerLinearRegression(
            SpectrumEncoder(k=2),
            fit_intercept=True,
        )
        returned = model.set_params(
            standardize=True,
            with_mean=True,
            fit_intercept=False,
        )

        assert returned is model
        assert model.standardize is True
        assert model.with_mean is True
        assert model.regressor_kwargs["fit_intercept"] is False

    def test_nested_encoder_parameter_raises_clear_error(self):
        model = KmerLinearRegression(SpectrumEncoder(k=2))
        with pytest.raises(ValueError, match="does not support nested parameters"):
            model.set_params(encoder__canonical_rc=True)

    def test_invalid_parameter_raises(self):
        model = KmerLinearRegression(SpectrumEncoder(k=2))
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(not_a_parameter=True)

    def test_multioutput_targets(self, regression_dataset):
        seqs, y = regression_dataset
        targets = np.column_stack([y, 2.0 * y])
        model = KmerLinearRegression(SpectrumEncoder(k=1)).fit(seqs, targets)

        predictions = model.predict(seqs)
        assert predictions.shape == targets.shape
        assert np.allclose(predictions, targets)
