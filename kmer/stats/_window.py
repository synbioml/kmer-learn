"""Shared abstraction and aggregation for windowed sequence statistics."""

from __future__ import annotations

from abc import abstractmethod
from functools import update_wrapper
import inspect
import pickle
from typing import Callable, Sequence

import numpy as np

from ._base import (
    BaseSequenceStat,
    FeatureNameResolver,
    SequenceInput,
    _as_sequence_list,
    _restore_sequence_stat,
    _validate_feature_names,
    _validate_result,
)

_SUPPORTED_AGGREGATIONS = {
    "mean", "std", "median", "min", "max", "quantile", "count"
}


def _as_tuple(value, *, name: str) -> tuple:
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a str or an iterable") from exc


def _levels(level) -> tuple[float, ...]:
    if level is None:
        return ()
    if np.isscalar(level):
        values = (float(level),)
    else:
        values = tuple(float(item) for item in level)
    if not values or any(
        not np.isfinite(item) or item < 0.0 or item > 1.0
        for item in values
    ):
        raise ValueError("level must contain finite quantiles in [0, 1]")
    return values


def _validate_window_options(
    *,
    window_size: int,
    step: int,
    agg,
    level,
    count_thresholds,
):
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")

    aggregations = _as_tuple(agg, name="agg")
    if not aggregations:
        raise ValueError("agg must contain at least one aggregation")
    unknown = set(aggregations) - _SUPPORTED_AGGREGATIONS
    if unknown:
        raise ValueError(
            "unsupported aggregation(s): " + ", ".join(sorted(unknown))
        )
    if len(set(aggregations)) != len(aggregations):
        raise ValueError("agg values must be unique")

    quantiles = _levels(level)
    if "quantile" in aggregations and not quantiles:
        raise ValueError("level is required when agg includes 'quantile'")

    thresholds = None
    if "count" in aggregations:
        if count_thresholds is None:
            raise ValueError(
                "count_thresholds is required when agg includes 'count'"
            )
        try:
            thresholds = tuple(count_thresholds)
        except TypeError as exc:
            raise TypeError(
                "count_thresholds must be a (lower, upper) pair"
            ) from exc
        if len(thresholds) != 2 or thresholds == (None, None):
            raise ValueError(
                "count_thresholds must contain at least one finite bound"
            )
        lower, upper = thresholds
        if lower is not None:
            lower = float(lower)
            if not np.isfinite(lower):
                raise ValueError("count_thresholds bounds must be finite")
        if upper is not None:
            upper = float(upper)
            if not np.isfinite(upper):
                raise ValueError("count_thresholds bounds must be finite")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("count_thresholds lower bound exceeds upper bound")
        thresholds = (lower, upper)

    return aggregations, quantiles, thresholds


def _expanded_names(base_names, aggregations, quantiles) -> np.ndarray:
    names = []
    for base_name in base_names:
        for aggregation in aggregations:
            if aggregation == "quantile":
                names.extend(
                    f"{base_name}_quantile_{quantile:g}"
                    for quantile in quantiles
                )
            else:
                names.append(f"{base_name}_{aggregation}")
    return _validate_feature_names(names)


def _aggregate_windows(values, aggregations, quantiles, thresholds):
    n_channels = values.shape[1]
    result = []
    for channel in range(n_channels):
        column = values[:, channel]
        finite = column[np.isfinite(column)]
        for aggregation in aggregations:
            if aggregation == "count":
                lower, upper = thresholds
                selected = finite
                if lower is not None:
                    selected = selected[selected >= lower]
                if upper is not None:
                    selected = selected[selected <= upper]
                result.append(float(len(selected)))
            elif finite.size == 0:
                if aggregation == "quantile":
                    result.extend([np.nan] * len(quantiles))
                else:
                    result.append(np.nan)
            elif aggregation == "mean":
                result.append(float(np.mean(finite)))
            elif aggregation == "std":
                result.append(float(np.std(finite)))
            elif aggregation == "median":
                result.append(float(np.median(finite)))
            elif aggregation == "min":
                result.append(float(np.min(finite)))
            elif aggregation == "max":
                result.append(float(np.max(finite)))
            elif aggregation == "quantile":
                result.extend(float(item) for item in np.quantile(finite, quantiles))
    return result


class BaseWindowSequenceStat(BaseSequenceStat):
    """Abstract contract for statistics computed over sequence windows.

    Subclasses produce one two-dimensional array of per-window channel values
    for a single sequence. This base class handles batch input, common window
    aggregation, threshold counting, output naming, and validation.
    """

    @abstractmethod
    def _window_values(
        self,
        sequence: str,
        *,
        window_size: int,
        step: int,
        **kwargs,
    ) -> np.ndarray:
        """Return ``(n_windows, n_channels)`` values for one sequence."""

    @abstractmethod
    def _window_feature_names(self, **kwargs) -> np.ndarray:
        """Return one name per unaggregated window channel."""

    def __call__(
        self,
        sequences: SequenceInput,
        *,
        window_size: int,
        step: int = 1,
        agg="mean",
        level=None,
        count_thresholds=None,
        **kwargs,
    ) -> np.ndarray:
        normalized = _as_sequence_list(sequences)
        aggregations, quantiles, thresholds = _validate_window_options(
            window_size=window_size,
            step=step,
            agg=agg,
            level=level,
            count_thresholds=count_thresholds,
        )
        base_names = self._window_feature_names(**kwargs)
        output_names = _expanded_names(base_names, aggregations, quantiles)
        rows = []
        for sequence in normalized:
            values = self._window_values(
                sequence,
                window_size=window_size,
                step=step,
                **kwargs,
            )
            values = _validate_result(
                values,
                n_sequences=values.shape[0] if isinstance(values, np.ndarray) and values.ndim == 2 else -1,
                feature_names=base_names,
                statistic_name=self.name,
            )
            rows.append(
                _aggregate_windows(values, aggregations, quantiles, thresholds)
            )
        result = np.asarray(rows, dtype=float).reshape(
            len(normalized), len(output_names)
        )
        return _validate_result(
            result,
            n_sequences=len(normalized),
            feature_names=output_names,
            statistic_name=self.name,
        )

    def get_feature_names_out(
        self,
        *,
        window_size: int,
        step: int = 1,
        agg="mean",
        level=None,
        count_thresholds=None,
        **kwargs,
    ) -> np.ndarray:
        aggregations, quantiles, _ = _validate_window_options(
            window_size=window_size,
            step=step,
            agg=agg,
            level=level,
            count_thresholds=count_thresholds,
        )
        del step
        return _expanded_names(
            self._window_feature_names(**kwargs), aggregations, quantiles
        )


class _FunctionWindowSequenceStat(BaseWindowSequenceStat):
    def __init__(
        self,
        function: Callable[..., np.ndarray],
        feature_names: FeatureNameResolver,
    ):
        if not callable(function):
            raise TypeError("decorated window statistic must be callable")
        if not callable(feature_names):
            raise TypeError("feature_names must be a callable resolver")
        self._function = function
        self._feature_name_resolver = feature_names
        self._global_module = function.__module__
        self._global_qualname = function.__qualname__
        update_wrapper(self, function)
        parameters = list(inspect.signature(function).parameters.values())
        if parameters:
            parameters[0] = parameters[0].replace(name="sequences")
        insertion = next(
            (index + 1 for index, parameter in enumerate(parameters)
             if parameter.name == "step"),
            1,
        )
        common = [
            inspect.Parameter(
                "agg", inspect.Parameter.KEYWORD_ONLY, default="mean"
            ),
            inspect.Parameter(
                "level", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
            inspect.Parameter(
                "count_thresholds",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
            ),
        ]
        parameters[insertion:insertion] = common
        self.__signature__ = inspect.Signature(
            parameters, return_annotation=np.ndarray
        )

    @property
    def name(self) -> str:
        return self.__name__

    def _window_values(self, sequence, *, window_size, step, **kwargs):
        result = self._function(
            sequence, window_size=window_size, step=step, **kwargs
        )
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"{self.name} must return a numpy.ndarray, "
                f"got {type(result).__name__}"
            )
        if result.ndim != 2:
            raise ValueError(
                f"{self.name} must return a two-dimensional per-window array"
            )
        return result

    def _window_feature_names(self, **kwargs):
        return _validate_feature_names(self._feature_name_resolver(**kwargs))

    def __reduce__(self):
        if "<locals>" in self._global_qualname:
            raise pickle.PicklingError(
                "decorated window statistics must be defined at module scope "
                "to support serialization"
            )
        return (_restore_sequence_stat, (self._global_module, self._global_qualname))


def windowed_sequence_stat(
    *,
    feature_names: FeatureNameResolver,
) -> Callable[[Callable[..., np.ndarray]], BaseWindowSequenceStat]:
    """Decorate a per-sequence window calculator as a public statistic.

    The decorated implementation receives one sequence plus ``window_size``
    and ``step`` and returns ``(n_windows, n_channels)`` values. The wrapper
    applies common aggregations and returns the standard batch-level 2D array.
    """
    if not callable(feature_names):
        raise TypeError("feature_names must be a callable resolver")

    def decorate(function):
        return _FunctionWindowSequenceStat(function, feature_names)

    return decorate
