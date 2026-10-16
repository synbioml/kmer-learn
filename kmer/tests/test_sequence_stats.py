"""Tests for callable sequence statistics and their aggregating encoder."""

from abc import ABC
import io
import pickle

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from kmer.encoders import SequenceStatsEncoder
from kmer.stats import BaseSequenceStat, sequence_stat


def _length_names(scale=1.0, label="length"):
    del scale
    return [label]


@sequence_stat(feature_names=_length_names)
def length_stat(sequences, scale=1.0, label="length"):
    del label
    return np.asarray(
        [[len(sequence) * scale] for sequence in sequences], dtype=float
    )


def _composition_names():
    return ["a_fraction", "t_fraction"]


@sequence_stat(feature_names=_composition_names)
def composition_stat(sequences):
    result = np.empty((len(sequences), 2), dtype=float)
    for row, sequence in enumerate(sequences):
        denominator = len(sequence) or 1
        result[row] = (
            sequence.count("A") / denominator,
            sequence.count("T") / denominator,
        )
    return result


def _bad_names():
    return ["bad"]


@sequence_stat(feature_names=_bad_names)
def non_array_stat(sequences):
    return [[len(sequence)] for sequence in sequences]


@sequence_stat(feature_names=_bad_names)
def one_dimensional_stat(sequences):
    return np.asarray([len(sequence) for sequence in sequences])


@sequence_stat(feature_names=_bad_names)
def wrong_row_count_stat(sequences):
    return np.empty((len(sequences) + 1, 1))


@sequence_stat(feature_names=lambda: ["first", "second"])
def wrong_column_count_stat(sequences):
    return np.empty((len(sequences), 1))


def test_base_sequence_stat_is_abstract():
    assert issubclass(BaseSequenceStat, ABC)
    with pytest.raises(TypeError):
        BaseSequenceStat()


def test_sequence_stat_accepts_one_string_and_returns_2d_array():
    result = length_stat("ACGT")
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1)
    np.testing.assert_array_equal(result, [[4.0]])


def test_sequence_stat_accepts_list_like_and_generator_inputs():
    expected = np.asarray([[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(length_stat(["A", "AC", "ACG"]), expected)
    np.testing.assert_array_equal(
        length_stat(sequence for sequence in ["A", "AC", "ACG"]), expected
    )


def test_sequence_stat_rejects_non_string_inputs():
    with pytest.raises(TypeError, match="every sequence"):
        length_stat(["ACGT", b"ACGT"])
    with pytest.raises(TypeError, match="str or an iterable"):
        length_stat(42)


def test_sequence_stat_requires_callable_feature_name_resolver():
    with pytest.raises(TypeError, match="feature_names"):
        sequence_stat(feature_names=["value"])


def test_sequence_stat_validates_result_contract():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        non_array_stat(["ACGT"])
    with pytest.raises(ValueError, match="two-dimensional"):
        one_dimensional_stat(["ACGT"])
    with pytest.raises(ValueError, match="rows"):
        wrong_row_count_stat(["ACGT"])
    with pytest.raises(ValueError, match="columns"):
        wrong_column_count_stat(["ACGT"])


def test_partial_saves_kwargs_and_call_time_kwargs_override_them():
    doubled = length_stat.partial(scale=2.0, label="double_length")
    assert isinstance(doubled, BaseSequenceStat)
    assert doubled.keywords == {"scale": 2.0, "label": "double_length"}
    np.testing.assert_array_equal(doubled(["AC", "ACGT"]), [[4.0], [8.0]])
    np.testing.assert_array_equal(doubled("AC", scale=3.0), [[6.0]])
    assert doubled.get_feature_names_out().tolist() == ["double_length"]


def test_repeated_partial_merges_kwargs():
    configured = length_stat.partial(scale=2.0).partial(label="scaled_length")
    assert configured.keywords == {"scale": 2.0, "label": "scaled_length"}
    assert configured.get_feature_names_out().tolist() == ["scaled_length"]


def test_decorated_and_partial_statistics_are_pickleable():
    assert pickle.loads(pickle.dumps(length_stat)) is length_stat

    configured = length_stat.partial(scale=2.0, label="double_length")
    restored = pickle.loads(pickle.dumps(configured))
    np.testing.assert_array_equal(restored(["A", "AC"]), [[2.0], [4.0]])
    assert restored.get_feature_names_out().tolist() == ["double_length"]


def test_sequence_stats_encoder_stacks_raw_and_partial_statistics():
    encoder = SequenceStatsEncoder(
        [
            length_stat,
            length_stat.partial(scale=0.5, label="half_length"),
            composition_stat,
        ]
    )
    result = encoder.fit_transform(["AATT", "AAAA"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)
    np.testing.assert_allclose(
        result,
        [[4.0, 2.0, 0.5, 0.5], [4.0, 2.0, 1.0, 0.0]],
    )
    assert encoder.get_feature_names_out().tolist() == [
        "length",
        "half_length",
        "a_fraction",
        "t_fraction",
    ]


def test_sequence_stats_encoder_accepts_one_statistic_and_one_string():
    encoder = SequenceStatsEncoder(length_stat)
    result = encoder.fit_transform("ACGT")
    assert result.shape == (1, 1)
    np.testing.assert_array_equal(result, [[4.0]])


def test_sequence_stats_encoder_rejects_invalid_or_empty_stats():
    with pytest.raises(ValueError, match="at least one"):
        SequenceStatsEncoder([]).fit(["ACGT"])
    with pytest.raises(TypeError, match="every item"):
        SequenceStatsEncoder([length_stat, len]).fit(["ACGT"])


def test_sequence_stats_encoder_rejects_duplicate_feature_names():
    duplicate = length_stat.partial(scale=2.0)
    with pytest.raises(ValueError, match="duplicate"):
        SequenceStatsEncoder([length_stat, duplicate]).fit(["ACGT"])


def test_sequence_stats_encoder_is_cloneable_and_joblib_serializable():
    encoder = SequenceStatsEncoder(
        [length_stat.partial(scale=2.0, label="double_length")]
    )
    cloned = clone(encoder)
    np.testing.assert_array_equal(cloned.fit_transform(["A", "AC"]), [[2.0], [4.0]])

    encoder.fit(["A"])
    buffer = io.BytesIO()
    joblib.dump(encoder, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(restored.transform(["ACG"]), [[6.0]])
    assert restored.get_feature_names_out().tolist() == ["double_length"]
