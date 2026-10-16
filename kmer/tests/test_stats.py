"""Tests for concrete sequence statistics and window aggregation."""

from itertools import product

import numpy as np
import pytest

from kmer.encoders import SequenceStatsEncoder
from kmer.stats import (
    BaseWindowSequenceStat,
    homopolymer_stats,
    kmer_complexity,
    kmer_content,
    lempel_ziv_complexity,
    markov_entropy,
    markov_transition_stats,
    monomer_content,
    monomer_skew,
    substring_complexity,
    tandem_repeat_abundance,
    windowed_kmer_content,
    windowed_monomer_content,
    windowed_monomer_skew,
    windowed_sequence_stat,
)


def _identity_names():
    return ["identity"]


@windowed_sequence_stat(feature_names=_identity_names)
def identity_windows(sequence, *, window_size, step=1):
    if len(sequence) < window_size:
        return np.empty((0, 1), dtype=float)
    return np.asarray(
        [[float(sequence[start])] for start in range(0, len(sequence) - window_size + 1, step)]
    )


def _brute_distinct(sequence, L_min=1, L_max=None):
    upper = len(sequence) if L_max is None else min(L_max, len(sequence))
    return sum(
        len({sequence[start:start + L] for start in range(len(sequence) - L + 1)})
        for L in range(L_min, upper + 1)
    )


def test_window_stat_abstract_contract_and_aggregations():
    assert isinstance(identity_windows, BaseWindowSequenceStat)
    result = identity_windows(
        "12345",
        window_size=2,
        agg=("mean", "std", "quantile", "count"),
        level=(0.25, 0.75),
        count_thresholds=(2, 3),
    )
    assert result.shape == (1, 5)
    np.testing.assert_allclose(
        result[0], [2.5, np.std([1, 2, 3, 4]), 1.75, 3.25, 2]
    )
    assert identity_windows.get_feature_names_out(
        window_size=2,
        agg=("mean", "quantile", "count"),
        level=(0.25, 0.75),
        count_thresholds=(2, None),
    ).tolist() == [
        "identity_mean",
        "identity_quantile_0.25",
        "identity_quantile_0.75",
        "identity_count",
    ]


def test_window_stat_validates_quantiles_and_count_thresholds():
    with pytest.raises(ValueError, match="level is required"):
        identity_windows("123", window_size=2, agg="quantile")
    with pytest.raises(ValueError, match="count_thresholds is required"):
        identity_windows("123", window_size=2, agg="count")
    with pytest.raises(ValueError, match="at least one finite"):
        identity_windows(
            "123", window_size=2, agg="count", count_thresholds=(None, None)
        )


def test_window_stat_short_sequences_return_nan_or_zero():
    result = identity_windows(
        ["1", ""],
        window_size=2,
        agg=("mean", "count"),
        count_thresholds=(1, None),
    )
    assert result.shape == (2, 2)
    assert np.isnan(result[:, 0]).all()
    np.testing.assert_array_equal(result[:, 1], [0, 0])


def test_monomer_content_default_and_explicit_groups():
    np.testing.assert_allclose(monomer_content("ACGT"), [[0.5]])
    result = monomer_content("GGCA", letters=("G", "C", "GC"))
    np.testing.assert_allclose(result, [[0.5, 0.25, 0.75]])
    assert monomer_content.get_feature_names_out(
        letters=("G", "C", "GC")
    ).tolist() == ["G_content", "C_content", "GC_content"]


def test_windowed_monomer_content_cartesian_output():
    result = windowed_monomer_content(
        "GGCCAATT",
        window_size=4,
        step=4,
        letters=("G", "C", "GC"),
        agg=("min", "max"),
    )
    np.testing.assert_allclose(result, [[0.0, 0.5, 0.0, 0.5, 0.0, 1.0]])


def test_kmer_content_modes():
    repeated = kmer_content("AAAA", L=2, stat=("entropy", "gini", "duplicates"))
    np.testing.assert_allclose(repeated, [[0.0, 0.0, 2.0]])
    diverse = kmer_content("ACGT", L=2, stat=("entropy", "gini", "duplicates"))
    np.testing.assert_allclose(diverse, [[1.0, 1.0, 0.0]])


def test_windowed_kmer_duplicates_and_tandem_sugar():
    direct = windowed_kmer_content(
        "ATATATATA",
        L=3,
        window_size=9,
        stat="duplicates",
        agg=("mean", "count"),
        count_thresholds=(1, None),
    )
    np.testing.assert_array_equal(direct, [[5.0, 1.0]])
    np.testing.assert_array_equal(tandem_repeat_abundance("ATATATATA"), [[1.0]])


def test_substring_complexity_matches_brute_force_and_alias():
    sequences = ["", "A", "AAAA", "ACGT", "ACACGT"]
    expected = np.asarray([[_brute_distinct(sequence, 2, 4)] for sequence in sequences])
    np.testing.assert_array_equal(
        substring_complexity(sequences, L_min=2, L_max=4), expected
    )
    assert kmer_complexity is substring_complexity


def test_substring_complexity_normalization_orders_low_and_high_diversity():
    low = substring_complexity("AAAAAAAA", normalize=True)[0, 0]
    high = substring_complexity("ACGTAGCT", normalize=True)[0, 0]
    assert 0 < low < high <= 1


def test_native_substring_complexity_exhaustive_small_sequences():
    sequences = [
        "".join(chars)
        for length in range(1, 7)
        for chars in product("AC", repeat=length)
    ]
    expected = np.asarray([[_brute_distinct(sequence)] for sequence in sequences])
    np.testing.assert_array_equal(substring_complexity(sequences), expected)


def test_homopolymer_stats():
    result = homopolymer_stats(
        ["AAAAACCCCG", "ACGT"],
        L_min=4,
        stat=("count", "fraction", "max", "mean"),
    )
    np.testing.assert_allclose(result[0], [2, 0.9, 5, 4.5])
    np.testing.assert_allclose(result[1], [0, 0, 0, 0])


def test_lempel_ziv_complexity_distinguishes_repetition():
    repeated = lempel_ziv_complexity("AAAAAAAAAAAAAAAA", normalize=False)[0, 0]
    diverse = lempel_ziv_complexity("ACGTGCACTGATCGTA", normalize=False)[0, 0]
    assert repeated < diverse


def test_monomer_skew_and_windowed_skew():
    np.testing.assert_allclose(
        monomer_skew("GGGC", letters=("GC", "AT")), [[0.5, np.nan]], equal_nan=True
    )
    result = windowed_monomer_skew(
        "GGGCCCAA",
        window_size=4,
        step=4,
        letters="GC",
        agg=("min", "max"),
    )
    np.testing.assert_allclose(result, [[-1.0, 0.5]])


def test_markov_entropy_detects_deterministic_transitions_and_rc_names():
    deterministic = markov_entropy("ATATATAT", order=1)[0, 0]
    less_predictable = markov_entropy("AACAGATAC", order=1)[0, 0]
    assert deterministic == pytest.approx(0.0)
    assert less_predictable > deterministic
    result = markov_entropy("AACGTT", order=1, use_rc=True)
    assert result.shape == (1, 2)
    assert markov_entropy.get_feature_names_out(order=1, use_rc=True).tolist() == [
        "markov1_entropy_fwd",
        "markov1_entropy_rc",
    ]


def test_markov_transition_spectral_gap_detects_periodicity():
    names = markov_transition_stats.get_feature_names_out(order=1).tolist()
    gap = names.index("markov1_spectral_gap_fwd")
    mixing = names.index("markov1_mixing_time_fwd")
    periodic = markov_transition_stats("ATATATAT", order=1)[0]
    fixed = markov_transition_stats("AAAAAAAA", order=1)[0]
    assert periodic[gap] == pytest.approx(0.0, abs=1e-12)
    assert np.isinf(periodic[mixing])
    assert fixed[gap] == pytest.approx(1.0)
    assert fixed[mixing] == pytest.approx(0.0)


def test_sequence_stats_encoder_aggregates_concrete_partials():
    encoder = SequenceStatsEncoder(
        [
            monomer_content,
            kmer_content.partial(L=2, stat="duplicates"),
            substring_complexity.partial(L_min=2, L_max=3),
        ]
    )
    result = encoder.fit_transform(["ACGT", "AAAA"])
    assert result.shape == (2, 3)
    assert encoder.get_feature_names_out().tolist() == [
        "GC_content",
        "2mer_duplicates",
        "substring_2_3_count",
    ]
