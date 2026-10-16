"""Composition and k-mer-distribution sequence statistics."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from ..utils import ALPHABET
from ._base import sequence_stat
from ._window import windowed_sequence_stat


def _groups(value, *, alphabet=ALPHABET, pair=False):
    groups = (value,) if isinstance(value, str) else tuple(value)
    if not groups:
        raise ValueError("letters must contain at least one group")
    seen = set()
    for group in groups:
        if not isinstance(group, str) or not group:
            raise ValueError("every letter group must be a non-empty str")
        if pair and (len(group) != 2 or group[0] == group[1]):
            raise ValueError("skew letter groups must contain two distinct letters")
        if any(letter not in alphabet for letter in group):
            raise ValueError(f"letter group {group!r} is outside alphabet {alphabet!r}")
        key = frozenset(group) if not pair else group
        if key in seen:
            raise ValueError("letter groups must be unique")
        seen.add(key)
    return groups


def _validate_sequence(sequence, alphabet):
    if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must be a non-empty string of unique symbols")
    if any(letter not in alphabet for letter in sequence):
        raise ValueError(f"input contains characters outside alphabet {alphabet!r}")


def _content_names(letters="GC", alphabet=ALPHABET):
    return [f"{group}_content" for group in _groups(letters, alphabet=alphabet)]


@sequence_stat(feature_names=_content_names)
def monomer_content(sequences, letters="GC", alphabet=ALPHABET):
    """Calculate one or more combined monomer fractions per sequence."""
    groups = _groups(letters, alphabet=alphabet)
    result = np.empty((len(sequences), len(groups)), dtype=float)
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        denominator = len(sequence)
        for column, group in enumerate(groups):
            result[row, column] = (
                sum(sequence.count(letter) for letter in set(group)) / denominator
                if denominator else np.nan
            )
    return result


@windowed_sequence_stat(feature_names=_content_names)
def windowed_monomer_content(
    sequence,
    *,
    window_size,
    step=1,
    letters="GC",
    alphabet=ALPHABET,
):
    """Calculate monomer fractions in complete sliding windows."""
    groups = _groups(letters, alphabet=alphabet)
    _validate_sequence(sequence, alphabet)
    if len(sequence) < window_size:
        return np.empty((0, len(groups)), dtype=float)

    prefixes = np.zeros((len(groups), len(sequence) + 1), dtype=np.int64)
    for index, letter in enumerate(sequence):
        prefixes[:, index + 1] = prefixes[:, index]
        for group_index, group in enumerate(groups):
            if letter in group:
                prefixes[group_index, index + 1] += 1

    starts = range(0, len(sequence) - window_size + 1, step)
    values = [
        (prefixes[:, start + window_size] - prefixes[:, start]) / window_size
        for start in starts
    ]
    return np.asarray(values, dtype=float).reshape(-1, len(groups))


def _stats(value):
    values = (value,) if isinstance(value, str) else tuple(value)
    supported = {"entropy", "gini", "duplicates"}
    if not values or any(item not in supported for item in values):
        raise ValueError(
            "stat must contain one or more of: entropy, gini, duplicates"
        )
    if len(set(values)) != len(values):
        raise ValueError("stat values must be unique")
    return values


def _validate_L(L):
    if not isinstance(L, int) or not 1 <= L <= 12:
        raise ValueError("L must be an integer in [1, 12]")


def _kmer_names(L=3, stat="entropy", alphabet=ALPHABET):
    _validate_L(L)
    _validate_sequence("", alphabet)
    return [f"{L}mer_{item}" for item in _stats(stat)]


def _histogram_values_from_moments(
    unique,
    sum_count_log_count,
    sum_squares,
    n_observations,
    max_categories,
    stats,
):
    maximum_unique = min(n_observations, max_categories)
    output = []
    for statistic in stats:
        if n_observations == 0:
            output.append(np.nan if statistic != "duplicates" else 0.0)
        elif statistic == "duplicates":
            output.append(float(n_observations - unique))
        elif maximum_unique <= 1:
            output.append(0.0)
        elif statistic == "entropy":
            entropy = (
                math.log(n_observations)
                - sum_count_log_count / n_observations
            )
            output.append(entropy / math.log(maximum_unique))
        elif statistic == "gini":
            raw = 1.0 - sum_squares / (n_observations * n_observations)
            maximum = 1.0 - 1.0 / maximum_unique
            output.append(raw / maximum)
    return output


def _histogram_values(counts, n_observations, max_categories, stats):
    return _histogram_values_from_moments(
        len(counts),
        sum(count * math.log(count) for count in counts.values()),
        sum(count * count for count in counts.values()),
        n_observations,
        max_categories,
        stats,
    )


@sequence_stat(feature_names=_kmer_names)
def kmer_content(sequences, L=3, stat="entropy", alphabet=ALPHABET):
    """Summarize a sequence's k-mer histogram."""
    _validate_L(L)
    statistics = _stats(stat)
    max_categories = len(alphabet) ** L
    result = np.empty((len(sequences), len(statistics)), dtype=float)
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        n_observations = max(0, len(sequence) - L + 1)
        counts = Counter(
            sequence[index:index + L] for index in range(n_observations)
        )
        result[row] = _histogram_values(
            counts, n_observations, max_categories, statistics
        )
    return result


@windowed_sequence_stat(feature_names=_kmer_names)
def windowed_kmer_content(
    sequence,
    *,
    window_size,
    step=1,
    L=3,
    stat="entropy",
    alphabet=ALPHABET,
):
    """Summarize rolling k-mer histograms in complete sequence windows."""
    _validate_L(L)
    statistics = _stats(stat)
    _validate_sequence(sequence, alphabet)
    if window_size < L:
        raise ValueError("window_size must be at least L")
    if len(sequence) < window_size:
        return np.empty((0, len(statistics)), dtype=float)

    n_observations = window_size - L + 1
    max_categories = len(alphabet) ** L
    counts = Counter(
        sequence[index:index + L] for index in range(n_observations)
    )
    unique = len(counts)
    sum_count_log_count = sum(
        count * math.log(count) for count in counts.values()
    )
    sum_squares = sum(count * count for count in counts.values())

    def update(word, change):
        nonlocal unique, sum_count_log_count, sum_squares
        previous = counts.get(word, 0)
        if previous:
            sum_count_log_count -= previous * math.log(previous)
            sum_squares -= previous * previous
        current = previous + change
        if current < 0:
            raise RuntimeError("rolling k-mer count became negative")
        if current:
            counts[word] = current
            sum_count_log_count += current * math.log(current)
            sum_squares += current * current
            if previous == 0:
                unique += 1
        else:
            del counts[word]
            unique -= 1

    rows = []
    last_start = len(sequence) - window_size
    for start in range(last_start + 1):
        if start % step == 0:
            rows.append(
                _histogram_values_from_moments(
                    unique,
                    sum_count_log_count,
                    sum_squares,
                    n_observations,
                    max_categories,
                    statistics,
                )
            )
        if start == last_start:
            break
        update(sequence[start:start + L], -1)
        incoming_start = start + window_size - L + 1
        update(sequence[incoming_start:incoming_start + L], 1)
    return np.asarray(rows, dtype=float).reshape(-1, len(statistics))


def _skew_names(letters="GC", alphabet=ALPHABET):
    return [f"{group}_skew" for group in _groups(letters, alphabet=alphabet, pair=True)]


def _skew(positive, negative):
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else np.nan


@sequence_stat(feature_names=_skew_names)
def monomer_skew(sequences, letters="GC", alphabet=ALPHABET):
    """Calculate directional monomer skews such as GC or AT skew."""
    groups = _groups(letters, alphabet=alphabet, pair=True)
    result = np.empty((len(sequences), len(groups)), dtype=float)
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        for column, group in enumerate(groups):
            result[row, column] = _skew(
                sequence.count(group[0]), sequence.count(group[1])
            )
    return result


@windowed_sequence_stat(feature_names=_skew_names)
def windowed_monomer_skew(
    sequence,
    *,
    window_size,
    step=1,
    letters="GC",
    alphabet=ALPHABET,
):
    """Calculate monomer skew in complete sliding windows."""
    groups = _groups(letters, alphabet=alphabet, pair=True)
    _validate_sequence(sequence, alphabet)
    if len(sequence) < window_size:
        return np.empty((0, len(groups)), dtype=float)
    prefixes = {
        letter: np.concatenate(([0], np.cumsum([char == letter for char in sequence])))
        for letter in set("".join(groups))
    }
    rows = []
    for start in range(0, len(sequence) - window_size + 1, step):
        row = []
        for group in groups:
            positive = prefixes[group[0]][start + window_size] - prefixes[group[0]][start]
            negative = prefixes[group[1]][start + window_size] - prefixes[group[1]][start]
            row.append(_skew(positive, negative))
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(-1, len(groups))
