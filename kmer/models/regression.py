"""Linear regression on encoded k-mer features."""

from __future__ import annotations

import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class KmerLinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary least-squares regression on k-mer features.

    The estimator fits an encoder on training sequences, optionally scales and
    centers the resulting features, and delegates regression to
    :class:`sklearn.linear_model.LinearRegression`.

    Parameters
    ----------
    encoder : object
        Encoder with ``fit_transform(seqs)`` and ``transform(seqs)`` methods.
        Sparse encoder output is supported.
    standardize : bool, default False
        Scale features before regression.
    with_mean : bool, default False
        Center features when ``standardize=True``. The default keeps sparse
        matrices sparse. Setting this to True converts sparse encoder output
        to a dense array, which may require substantial memory.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`sklearn.linear_model.LinearRegression`. Supported arguments
        depend on the installed scikit-learn version.

    Attributes
    ----------
    regressor_ : sklearn.linear_model.LinearRegression
        Fitted underlying estimator.
    scaler_ : sklearn.preprocessing.StandardScaler
        Fitted feature scaler. Present only when ``standardize=True``.

    Examples
    --------
    >>> from kmer.encoders import SpectrumEncoder
    >>> from kmer.models import KmerLinearRegression
    >>> model = KmerLinearRegression(
    ...     encoder=SpectrumEncoder(k=3),
    ...     standardize=True,
    ...     fit_intercept=True,
    ... )
    >>> model.fit(["ACGTACGT", "AAAACCCC", "GGGGTTTT"], [0.2, 0.7, 0.9])
    KmerLinearRegression(...)
    >>> model.predict(["ACGTACGA"]).shape
    (1,)
    """

    def __init__(
        self,
        encoder,
        standardize: bool = False,
        with_mean: bool = False,
        **kwargs,
    ):
        self.encoder = encoder
        self.standardize = standardize
        self.with_mean = with_mean
        self.regressor_kwargs = kwargs

    def get_params(self, deep: bool = True):
        """Return wrapper and forwarded LinearRegression parameters."""
        params = {
            "encoder": self.encoder,
            "standardize": self.standardize,
            "with_mean": self.with_mean,
            **self.regressor_kwargs,
        }
        if deep:
            nested = {}
            for name, value in params.items():
                if hasattr(value, "get_params"):
                    for key, nested_value in value.get_params().items():
                        nested[f"{name}__{key}"] = nested_value
            params.update(nested)
        return params

    def set_params(self, **params):
        """Set wrapper, encoder, or forwarded LinearRegression parameters."""
        if not params:
            return self

        valid_params = self.get_params(deep=False)
        nested_params = {}
        for key, value in params.items():
            name, delimiter, nested_key = key.partition("__")
            if name not in valid_params:
                valid = ", ".join(sorted(valid_params))
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator "
                    f"{type(self).__name__}. Valid parameters are: {valid}."
                )
            if delimiter:
                nested_params.setdefault(name, {})[nested_key] = value
            elif name == "encoder":
                self.encoder = value
            elif name == "standardize":
                self.standardize = value
            elif name == "with_mean":
                self.with_mean = value
            else:
                self.regressor_kwargs[name] = value

        for name, sub_params in nested_params.items():
            target = self.encoder if name == "encoder" else self.regressor_kwargs[name]
            if not hasattr(target, "set_params"):
                raise ValueError(
                    f"Parameter {name!r} does not support nested parameters"
                )
            target.set_params(**sub_params)
        return self

    def fit(self, seqs, y, sample_weight=None):
        """Fit the encoder, optional scaler, and linear regressor.

        Parameters
        ----------
        seqs : iterable of str or bytes
            Training sequences.
        y : array-like
            Continuous target values. Multi-output targets are supported by
            the underlying estimator.
        sample_weight : array-like or None, default None
            Per-sample weights forwarded to ``LinearRegression.fit``.

        Returns
        -------
        self
        """
        X = self.encoder.fit_transform(seqs)
        if self.standardize:
            if self.with_mean and sp.issparse(X):
                X = X.toarray()
            self.scaler_ = StandardScaler(with_mean=self.with_mean)
            X = self.scaler_.fit_transform(X)

        self.regressor_ = LinearRegression(**self.regressor_kwargs)
        self.regressor_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, seqs):
        """Predict continuous targets for sequences."""
        X = self.encoder.transform(seqs)
        if self.standardize:
            if self.with_mean and sp.issparse(X):
                X = X.toarray()
            X = self.scaler_.transform(X)
        return self.regressor_.predict(X)

    @property
    def coef_(self):
        """Fitted regression coefficients."""
        return self.regressor_.coef_

    @property
    def intercept_(self):
        """Fitted regression intercept."""
        return self.regressor_.intercept_
