"""Core contracts for callable sequence statistics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import update_wrapper
import importlib
import inspect
import pickle
from typing import Callable, Iterable, Mapping, Sequence, Union

import numpy as np


FeatureNameResolver = Callable[..., Sequence[str]]
SequenceInput = Union[str, Iterable[str]]


def _as_sequence_list(sequences: SequenceInput) -> list[str]:
    """Normalize one sequence or an iterable of sequences to a list."""
    if isinstance(sequences, str):
        normalized = [sequences]
    else:
        try:
            normalized = list(sequences)
        except TypeError as exc:
            raise TypeError(
                "sequences must be a str or an iterable of str"
            ) from exc

    if any(not isinstance(sequence, str) for sequence in normalized):
        raise TypeError("every sequence must be a str")
    return normalized


def _validate_feature_names(names) -> np.ndarray:
    """Return a validated one-dimensional object array of unique names."""
    names = np.asarray(list(names), dtype=object)
    if names.ndim != 1:
        raise ValueError("feature-name resolver must return a one-dimensional sequence")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("feature names must be str instances")
    if len(set(names.tolist())) != len(names):
        raise ValueError("feature names must be unique within a statistic")
    return names


def _validate_result(
    result,
    *,
    n_sequences: int,
    feature_names: np.ndarray,
    statistic_name: str,
) -> np.ndarray:
    """Validate the common two-dimensional output contract."""
    if not isinstance(result, np.ndarray):
        raise TypeError(
            f"{statistic_name} must return a numpy.ndarray, "
            f"got {type(result).__name__}"
        )
    if result.ndim != 2:
        raise ValueError(
            f"{statistic_name} must return a two-dimensional array, "
            f"got shape {result.shape}"
        )
    if result.shape[0] != n_sequences:
        raise ValueError(
            f"{statistic_name} returned {result.shape[0]} rows for "
            f"{n_sequences} sequences"
        )
    if result.shape[1] != len(feature_names):
        raise ValueError(
            f"{statistic_name} returned {result.shape[1]} columns but its "
            f"feature-name resolver returned {len(feature_names)} names"
        )
    return result


class BaseSequenceStat(ABC):
    """Abstract callable contract for one or more sequence statistics.

    Public statistic callables accept either one sequence or an iterable of
    sequences and always return a two-dimensional NumPy array. Concrete
    statistics are normally created by decorating a function with
    :func:`sequence_stat`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable name of the statistic."""

    @abstractmethod
    def __call__(self, sequences: SequenceInput, **kwargs) -> np.ndarray:
        """Calculate the statistic for one or more sequences."""

    @abstractmethod
    def get_feature_names_out(self, **kwargs) -> np.ndarray:
        """Return exact names for columns produced with ``kwargs``."""

    def partial(self, **kwargs) -> "BaseSequenceStat":
        """Bind keyword arguments and return another callable statistic.

        Call-time keyword arguments override saved values, matching
        :func:`functools.partial` semantics.
        """
        if isinstance(self, _PartialSequenceStat):
            merged = {**self._keywords, **kwargs}
            return _PartialSequenceStat(self._statistic, merged)
        return _PartialSequenceStat(self, kwargs)


class _FunctionSequenceStat(BaseSequenceStat):
    """Validated callable wrapper created by :func:`sequence_stat`."""

    def __init__(
        self,
        function: Callable[..., np.ndarray],
        feature_names: FeatureNameResolver,
    ):
        if not callable(function):
            raise TypeError("decorated statistic must be callable")
        if not callable(feature_names):
            raise TypeError("feature_names must be a callable resolver")

        self._function = function
        self._feature_name_resolver = feature_names
        self._global_module = function.__module__
        self._global_qualname = function.__qualname__
        update_wrapper(self, function)

    @property
    def name(self) -> str:
        return self.__name__

    def __call__(self, sequences: SequenceInput, **kwargs) -> np.ndarray:
        normalized = _as_sequence_list(sequences)
        names = self.get_feature_names_out(**kwargs)
        result = self._function(normalized, **kwargs)
        return _validate_result(
            result,
            n_sequences=len(normalized),
            feature_names=names,
            statistic_name=self.name,
        )

    def get_feature_names_out(self, **kwargs) -> np.ndarray:
        return _validate_feature_names(self._feature_name_resolver(**kwargs))

    def __reduce__(self):
        if "<locals>" in self._global_qualname:
            raise pickle.PicklingError(
                "decorated sequence statistics must be defined at module scope "
                "to support serialization"
            )
        return (_restore_sequence_stat, (self._global_module, self._global_qualname))


class _PartialSequenceStat(BaseSequenceStat):
    """A sequence statistic with saved keyword arguments."""

    def __init__(
        self,
        statistic: BaseSequenceStat,
        keywords: Mapping[str, object],
    ):
        if not isinstance(statistic, BaseSequenceStat):
            raise TypeError("statistic must implement BaseSequenceStat")
        self._statistic = statistic
        self._keywords = dict(keywords)
        self.__name__ = statistic.name
        self.__doc__ = getattr(statistic, "__doc__", None)
        self.__module__ = getattr(statistic, "__module__", type(statistic).__module__)
        try:
            signature = inspect.signature(statistic)
            parameters = [
                parameter.replace(default=self._keywords[parameter.name])
                if parameter.name in self._keywords
                else parameter
                for parameter in signature.parameters.values()
            ]
            self.__signature__ = signature.replace(parameters=parameters)
        except (TypeError, ValueError):
            pass

    @property
    def name(self) -> str:
        return self._statistic.name

    @property
    def keywords(self) -> dict[str, object]:
        """A copy of the saved keyword arguments."""
        return dict(self._keywords)

    def __call__(self, sequences: SequenceInput, **kwargs) -> np.ndarray:
        merged = {**self._keywords, **kwargs}
        return self._statistic(sequences, **merged)

    def get_feature_names_out(self, **kwargs) -> np.ndarray:
        merged = {**self._keywords, **kwargs}
        return self._statistic.get_feature_names_out(**merged)

    def __repr__(self) -> str:
        arguments = ", ".join(
            f"{key}={value!r}" for key, value in self._keywords.items()
        )
        return f"{self.name}.partial({arguments})"

    def __reduce__(self):
        return (_restore_partial_sequence_stat, (self._statistic, self._keywords))


def _restore_sequence_stat(module_name: str, qualname: str) -> BaseSequenceStat:
    """Restore a decorated statistic by its module-level binding."""
    value = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not isinstance(value, BaseSequenceStat):
        raise pickle.UnpicklingError(
            f"{module_name}.{qualname} is no longer a sequence statistic"
        )
    return value


def _restore_partial_sequence_stat(
    statistic: BaseSequenceStat,
    keywords: Mapping[str, object],
) -> BaseSequenceStat:
    return statistic.partial(**keywords)


def sequence_stat(
    *,
    feature_names: FeatureNameResolver,
) -> Callable[[Callable[..., np.ndarray]], BaseSequenceStat]:
    """Decorate a function as a validated, partial-bindable statistic.

    The decorated implementation receives a ``list[str]`` even when its caller
    supplies one string. It must return a two-dimensional NumPy array with one
    row per sequence. ``feature_names`` receives the same keyword arguments and
    must return one exact name per output column.

    Decorated functions should be defined at module scope so that statistics,
    bound statistics, and estimators containing them can be serialized.
    """
    if not callable(feature_names):
        raise TypeError("feature_names must be a callable resolver")

    def decorate(function: Callable[..., np.ndarray]) -> BaseSequenceStat:
        return _FunctionSequenceStat(function, feature_names)

    return decorate
