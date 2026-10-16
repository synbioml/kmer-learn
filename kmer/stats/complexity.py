"""Substring diversity, runs, and compressibility statistics."""

from __future__ import annotations

import math

import numpy as np

from ..utils import ALPHABET
from ._base import sequence_stat
from .content import _validate_sequence
from ._native._complexity import _count_distinct_substrings


def _complexity_names(
    L_min=1,
    L_max=None,
    normalize=False,
    alphabet=ALPHABET,
):
    del alphabet
    upper = "end" if L_max is None else str(L_max)
    suffix = "normalized" if normalize else "count"
    return [f"substring_{L_min}_{upper}_{suffix}"]


def _maximum_distinct(length, L_min, L_max, alphabet_size):
    upper = length if L_max is None else min(length, L_max)
    if upper < L_min:
        return 0
    total = 0
    possibilities = min(alphabet_size ** L_min, length)
    for word_length in range(L_min, upper + 1):
        if word_length > L_min:
            possibilities = min(possibilities * alphabet_size, length)
        total += min(possibilities, length - word_length + 1)
    return total


@sequence_stat(feature_names=_complexity_names)
def substring_complexity(
    sequences,
    L_min=1,
    L_max=None,
    normalize=False,
    alphabet=ALPHABET,
):
    """Count distinct substrings in a length range using a native suffix automaton."""
    if not isinstance(L_min, int) or L_min < 1:
        raise ValueError("L_min must be a positive integer")
    if L_max is not None and (
        not isinstance(L_max, int) or L_max < L_min
    ):
        raise ValueError("L_max must be None or an integer >= L_min")
    if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must be a non-empty string of unique symbols")

    counts = _count_distinct_substrings(
        sequences,
        L_min=L_min,
        L_max=-1 if L_max is None else L_max,
        alphabet=alphabet,
    )
    result = np.asarray(counts, dtype=float).reshape(-1, 1)
    if normalize:
        for row, sequence in enumerate(sequences):
            maximum = _maximum_distinct(
                len(sequence), L_min, L_max, len(alphabet)
            )
            result[row, 0] = result[row, 0] / maximum if maximum else np.nan
    return result


kmer_complexity = substring_complexity


def _homopolymer_stats(value):
    values = (value,) if isinstance(value, str) else tuple(value)
    supported = {"count", "fraction", "max", "mean"}
    if not values or any(item not in supported for item in values):
        raise ValueError(
            "stat must contain one or more of: count, fraction, max, mean"
        )
    if len(set(values)) != len(values):
        raise ValueError("stat values must be unique")
    return values


def _homopolymer_names(L_min=5, stat=("count", "fraction", "max"), alphabet=ALPHABET):
    del alphabet
    return [f"homopolymer_L{L_min}_{item}" for item in _homopolymer_stats(stat)]


@sequence_stat(feature_names=_homopolymer_names)
def homopolymer_stats(
    sequences,
    L_min=5,
    stat=("count", "fraction", "max"),
    alphabet=ALPHABET,
):
    """Summarize homopolymer runs whose lengths are at least ``L_min``."""
    if not isinstance(L_min, int) or L_min < 1:
        raise ValueError("L_min must be a positive integer")
    statistics = _homopolymer_stats(stat)
    result = np.empty((len(sequences), len(statistics)), dtype=float)
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        runs = []
        start = 0
        while start < len(sequence):
            end = start + 1
            while end < len(sequence) and sequence[end] == sequence[start]:
                end += 1
            if end - start >= L_min:
                runs.append(end - start)
            start = end
        for column, statistic in enumerate(statistics):
            if statistic == "count":
                value = len(runs)
            elif statistic == "fraction":
                value = sum(runs) / len(sequence) if sequence else np.nan
            elif statistic == "max":
                value = max(runs, default=0)
            else:
                value = float(np.mean(runs)) if runs else 0.0
            result[row, column] = value
    return result


def _lz_names(normalize=True, alphabet=ALPHABET):
    del alphabet
    return ["lempel_ziv_normalized" if normalize else "lempel_ziv_factors"]


def _lz76_factor_count(sequence):
    n = len(sequence)
    if n == 0:
        return 0
    if n == 1:
        return 1
    complexity = 1
    prefix = 0
    match = 1
    boundary = 1
    longest = 1
    while True:
        if sequence[prefix + match - 1] == sequence[boundary + match - 1]:
            match += 1
            if boundary + match > n:
                complexity += 1
                break
        else:
            if match > longest:
                longest = match
            prefix += 1
            if prefix == boundary:
                complexity += 1
                boundary += longest
                if boundary + 1 > n:
                    break
                prefix = 0
                match = 1
                longest = 1
            else:
                match = 1
    return complexity


@sequence_stat(feature_names=_lz_names)
def lempel_ziv_complexity(sequences, normalize=True, alphabet=ALPHABET):
    """Calculate LZ76 factor count, optionally with asymptotic normalization."""
    result = np.empty((len(sequences), 1), dtype=float)
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        factors = _lz76_factor_count(sequence)
        if normalize:
            if len(sequence) <= 1:
                value = float(factors)
            else:
                value = (
                    factors
                    * (math.log(len(sequence)) / math.log(len(alphabet)))
                    / len(sequence)
                )
        else:
            value = float(factors)
        result[row, 0] = value
    return result
