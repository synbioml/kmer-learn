"""Dense aggregation of callable sequence statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..stats import BaseSequenceStat
from ..stats._base import _as_sequence_list


class SequenceStatsEncoder(TransformerMixin, BaseEstimator):
    """Aggregate callable sequence statistics into a dense feature matrix.

    Parameters
    ----------
    stats : BaseSequenceStat or iterable of BaseSequenceStat
        Raw decorated statistics using their default arguments, statistics
        configured with ``stat.partial(**kwargs)``, or a mixture of both.
        Every statistic must provide exact, unique output names.

    Attributes
    ----------
    n_features_out_ : int
        Number of columns produced by the aggregated statistics.

    Examples
    --------
    Concrete statistics are not built into this infrastructure module. Given a
    decorated statistic named ``stat``, it can be configured and aggregated as
    follows::

        encoder = SequenceStatsEncoder(
            [stat.partial(k=2), stat.partial(k=3)]
        )
        X = encoder.fit_transform(sequences)
    """

    def __init__(self, stats):
        self.stats = stats

    @staticmethod
    def _validate_stats(stats) -> tuple[BaseSequenceStat, ...]:
        if isinstance(stats, BaseSequenceStat):
            validated = (stats,)
        else:
            if isinstance(stats, (str, bytes)) or not isinstance(stats, Iterable):
                raise TypeError(
                    "stats must be a BaseSequenceStat or an iterable of them"
                )
            validated = tuple(stats)

        if not validated:
            raise ValueError("stats must contain at least one sequence statistic")
        for statistic in validated:
            if not isinstance(statistic, BaseSequenceStat):
                raise TypeError(
                    "every item in stats must implement BaseSequenceStat; "
                    f"got {type(statistic).__name__}"
                )
        return validated

    def fit(self, sequences, y=None):
        """Validate statistics and establish the output-column contract."""
        del sequences, y
        self.stats_ = self._validate_stats(self.stats)
        name_blocks = [stat.get_feature_names_out() for stat in self.stats_]
        self.feature_names_out_ = np.concatenate(name_blocks).astype(
            object, copy=False
        )
        if len(set(self.feature_names_out_.tolist())) != len(
            self.feature_names_out_
        ):
            raise ValueError(
                "statistics supplied to SequenceStatsEncoder produce duplicate "
                "feature names"
            )
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, sequences) -> np.ndarray:
        """Calculate and horizontally stack all configured statistics."""
        check_is_fitted(self, ("stats_", "feature_names_out_"))
        normalized = _as_sequence_list(sequences)
        blocks = []
        for statistic in self.stats_:
            block = statistic(normalized)
            expected_columns = len(statistic.get_feature_names_out())
            if block.shape[1] != expected_columns:
                raise ValueError(
                    f"{statistic.name} changed its output-column count after fit"
                )
            blocks.append(block)
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the exact names supplied by the configured statistics."""
        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
