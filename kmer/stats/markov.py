"""Per-sequence Markov-chain statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from ..utils import ALPHABET, reverse_complement
from ._base import sequence_stat
from .content import _validate_sequence


def _validate_options(order, smoothing, alphabet, use_rc):
    if not isinstance(order, int) or not 0 <= order <= 3:
        raise ValueError("order must be an integer in [0, 3]")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    if use_rc and alphabet != ALPHABET:
        raise ValueError("use_rc requires the standard DNA alphabet")


def _strand_sequences(sequence, use_rc):
    return (("fwd", sequence), ("rc", reverse_complement(sequence))) if use_rc else (("fwd", sequence),)


def _transition_counts(sequence, order):
    counts = defaultdict(Counter)
    for position in range(order, len(sequence)):
        context = sequence[position - order:position] if order else ""
        counts[context][sequence[position]] += 1
    return counts


def _conditional_entropy(sequence, order, smoothing, alphabet):
    counts = _transition_counts(sequence, order)
    n_transitions = sum(sum(row.values()) for row in counts.values())
    if n_transitions == 0:
        return np.nan
    entropy = 0.0
    for row in counts.values():
        total = sum(row.values())
        denominator = total + smoothing * len(alphabet)
        row_entropy = 0.0
        for symbol in alphabet:
            probability = (row[symbol] + smoothing) / denominator
            if probability > 0:
                row_entropy -= probability * math.log2(probability)
        entropy += total / n_transitions * row_entropy
    return entropy


def _markov_entropy_names(
    order=1,
    normalize=True,
    smoothing=0.0,
    use_rc=False,
    alphabet=ALPHABET,
):
    del normalize, smoothing, alphabet
    strands = ("fwd", "rc") if use_rc else ("fwd",)
    return [f"markov{order}_entropy_{strand}" for strand in strands]


@sequence_stat(feature_names=_markov_entropy_names)
def markov_entropy(
    sequences,
    order=1,
    normalize=True,
    smoothing=0.0,
    use_rc=False,
    alphabet=ALPHABET,
):
    """Calculate empirical conditional entropy for an order-k chain."""
    _validate_options(order, smoothing, alphabet, use_rc)
    result = np.empty((len(sequences), 2 if use_rc else 1), dtype=float)
    normalizer = math.log2(len(alphabet))
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        for column, (_, oriented) in enumerate(_strand_sequences(sequence, use_rc)):
            value = _conditional_entropy(oriented, order, smoothing, alphabet)
            result[row, column] = value / normalizer if normalize else value
    return result


_TRANSITION_FEATURES = (
    "max_transition",
    "deterministic_context_fraction",
    "observed_context_fraction",
    "normalized_log_likelihood",
    "perplexity",
    "spectral_gap",
    "mixing_time",
)


def _transition_names(
    order=1,
    smoothing=0.0,
    use_rc=False,
    alphabet=ALPHABET,
):
    del smoothing, alphabet
    strands = ("fwd", "rc") if use_rc else ("fwd",)
    return [
        f"markov{order}_{feature}_{strand}"
        for strand in strands
        for feature in _TRANSITION_FEATURES
    ]


def _transition_summary(sequence, order, smoothing, alphabet):
    counts = _transition_counts(sequence, order)
    total_transitions = sum(sum(row.values()) for row in counts.values())
    if total_transitions == 0:
        return [np.nan] * len(_TRANSITION_FEATURES)

    weighted_max = 0.0
    deterministic = 0
    log_likelihood = 0.0
    entropy = 0.0
    for row in counts.values():
        total = sum(row.values())
        denominator = total + smoothing * len(alphabet)
        probabilities = {
            symbol: (row[symbol] + smoothing) / denominator
            for symbol in alphabet
        }
        weighted_max += total * max(probabilities.values())
        deterministic += int(max(row.values()) == total)
        for symbol, count in row.items():
            probability = probabilities[symbol]
            log_likelihood += count * math.log(probability)
        entropy -= total / total_transitions * sum(
            probability * math.log(probability)
            for probability in probabilities.values()
            if probability > 0
        )

    contexts = tuple(counts)
    context_index = {context: index for index, context in enumerate(contexts)}
    matrix = np.zeros((len(contexts), len(contexts)), dtype=float)
    for context, row in counts.items():
        source = context_index[context]
        for symbol in alphabet:
            target_context = (context + symbol)[-order:] if order else ""
            if target_context in context_index:
                matrix[source, context_index[target_context]] += row[symbol] + smoothing
        row_sum = matrix[source].sum()
        if row_sum:
            matrix[source] /= row_sum
        else:
            matrix[source, source] = 1.0

    if len(contexts) == 1:
        spectral_gap = 1.0
        mixing_time = 0.0
    else:
        eigenvalues = np.linalg.eigvals(matrix)
        moduli = np.sort(np.abs(eigenvalues))[::-1]
        slem = float(min(1.0, max(0.0, moduli[1])))
        spectral_gap = max(0.0, 1.0 - slem)
        mixing_time = math.inf if spectral_gap <= 1e-12 else 1.0 / spectral_gap

    return [
        weighted_max / total_transitions,
        deterministic / len(counts),
        len(counts) / (len(alphabet) ** order),
        log_likelihood / (total_transitions * math.log(len(alphabet))),
        math.exp(entropy),
        spectral_gap,
        mixing_time,
    ]


@sequence_stat(feature_names=_transition_names)
def markov_transition_stats(
    sequences,
    order=1,
    smoothing=0.0,
    use_rc=False,
    alphabet=ALPHABET,
):
    """Summarize transition predictability and convergence of a Markov chain."""
    _validate_options(order, smoothing, alphabet, use_rc)
    n_strands = 2 if use_rc else 1
    result = np.empty(
        (len(sequences), n_strands * len(_TRANSITION_FEATURES)), dtype=float
    )
    for row, sequence in enumerate(sequences):
        _validate_sequence(sequence, alphabet)
        values = []
        for _, oriented in _strand_sequences(sequence, use_rc):
            values.extend(
                _transition_summary(oriented, order, smoothing, alphabet)
            )
        result[row] = values
    return result
