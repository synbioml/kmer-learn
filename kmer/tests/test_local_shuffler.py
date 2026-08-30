"""Tests for the native position-bounded LocalShuffler."""

from collections import Counter

import numpy as np
import pytest

from kmer.background import LocalShuffler


def _assert_bounded_permutation(source, shuffled, max_shift):
    positions = {value: index for index, value in enumerate(shuffled)}
    for original_index, value in enumerate(source):
        assert abs(positions[value] - original_index) <= max_shift


@pytest.mark.parametrize("length", [2, 3, 7, 16, 32, 100])
@pytest.mark.parametrize("max_shift", [0, 1, 2, 3, 8, 32])
def test_displacement_bound(length, max_shift):
    source = "".join(chr(0x100 + index) for index in range(length))
    shuffler = LocalShuffler(max_shift=max_shift, seed=42)

    for _ in range(20):
        shuffled = shuffler.shuffle(source)
        _assert_bounded_permutation(source, shuffled, max_shift)


def test_permutation_mapping_matches_shuffle_semantics():
    shuffler = LocalShuffler(max_shift=4, seed=42)
    permutation = shuffler.permutation(50)
    assert permutation.dtype == np.intp
    assert sorted(permutation.tolist()) == list(range(50))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(50)
    assert np.all(np.abs(inverse - np.arange(50)) <= 4)


def test_str_length_type_and_composition_are_preserved():
    source = "AAACCCGGGTTTAAACCC"
    shuffled = LocalShuffler(max_shift=4, seed=42).shuffle(source)
    assert isinstance(shuffled, str)
    assert len(shuffled) == len(source)
    assert Counter(shuffled) == Counter(source)


def test_unicode_codepoints_are_preserved():
    source = "αβγδεζηθικλμνξοπρστυφχψω"
    shuffled = LocalShuffler(max_shift=5, seed=42).shuffle(source)
    assert isinstance(shuffled, str)
    assert Counter(shuffled) == Counter(source)
    _assert_bounded_permutation(source, shuffled, 5)


def test_bytes_input_preserves_type_and_composition():
    source = b"AAACCCGGGTTTAAACCC"
    shuffled = LocalShuffler(max_shift=4, seed=42).shuffle(source)
    assert isinstance(shuffled, bytes)
    assert len(shuffled) == len(source)
    assert Counter(shuffled) == Counter(source)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, object])
def test_numpy_input_preserves_shape_dtype_and_values(dtype):
    source = np.arange(100).astype(dtype)
    shuffled = LocalShuffler(max_shift=8, seed=42).shuffle(source)
    assert isinstance(shuffled, np.ndarray)
    assert shuffled.shape == source.shape
    assert shuffled.dtype == source.dtype
    assert sorted(shuffled.tolist()) == sorted(source.tolist())
    _assert_bounded_permutation(source.tolist(), shuffled.tolist(), 8)


def test_numpy_view_is_supported():
    source = np.arange(40, dtype=np.int32)[::2]
    shuffled = LocalShuffler(max_shift=3, seed=42).shuffle(source)
    assert shuffled.dtype == source.dtype
    assert sorted(shuffled.tolist()) == sorted(source.tolist())


def test_multidimensional_numpy_input_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        LocalShuffler(max_shift=2).shuffle(np.arange(12).reshape(3, 4))


def test_same_seed_produces_same_sequence():
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    first = LocalShuffler(max_shift=4, seed=123).shuffle(source)
    second = LocalShuffler(max_shift=4, seed=123).shuffle(source)
    assert first == second


def test_successive_calls_advance_rng_state():
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    shuffler = LocalShuffler(max_shift=4, seed=42)
    assert len({shuffler.shuffle(source) for _ in range(5)}) > 1


def test_small_uniform_kick_distribution_is_close_to_theory_and_symmetric():
    shuffler = LocalShuffler(max_shift=1, seed=42, L=None, H=None)
    trials = 50_000
    counts = Counter(shuffler.shuffle("012") for _ in range(trials))
    valid = ("012", "021", "102")
    tv = 0.5 * sum(abs(counts[value] / trials - 1 / 3) for value in valid)
    assert tv < 0.03
    assert abs(counts["021"] - counts["102"]) / trials < 0.01


def test_getstate_setstate_roundtrip_preserves_buffered_rng():
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    first = LocalShuffler(max_shift=4, seed=42)
    first.shuffle(source)
    state = first.getstate()
    assert isinstance(state, bytes)
    assert len(state) == 44
    expected = first.shuffle(source)

    second = LocalShuffler(max_shift=4, seed=999)
    second.setstate(state)
    assert second.shuffle(source) == expected


def test_invalid_rng_state_rejected():
    shuffler = LocalShuffler(max_shift=2, seed=42)
    with pytest.raises(ValueError):
        shuffler.setstate(b"short")


def test_perturb_and_perturb_many_match_serial_calls():
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    serial = LocalShuffler(max_shift=3, seed=42)
    expected = [serial.perturb(source), serial.perturb(source)]

    batch = LocalShuffler(max_shift=3, seed=42)
    assert batch.perturb_many([source, source]) == expected


def test_numpy_perturb_many():
    sources = [np.arange(20), np.arange(30)]
    outputs = LocalShuffler(max_shift=3, seed=42).perturb_many(sources)
    assert [len(output) for output in outputs] == [20, 30]
    assert all(isinstance(output, np.ndarray) for output in outputs)


def test_zero_shift_and_short_inputs_are_unchanged_in_value():
    shuffler = LocalShuffler(max_shift=0, seed=42)
    assert shuffler.shuffle("ACGT") == "ACGT"
    assert shuffler.shuffle("") == ""
    assert shuffler.shuffle(b"A") == b"A"
    np.testing.assert_array_equal(shuffler.shuffle(np.arange(5)), np.arange(5))


def test_shift_larger_than_sequence_is_supported():
    source = "ABCDEFG"
    shuffled = LocalShuffler(max_shift=100, seed=42).shuffle(source)
    assert sorted(shuffled) == sorted(source)


def test_automatic_limits():
    shuffler = LocalShuffler(max_shift=32, seed=42)
    assert shuffler.limits_for(200) == (64, 1024)
    assert shuffler.limits_for(0) == (16, 1024)


def test_custom_and_disabled_limits():
    assert LocalShuffler(4, L=None, H=None).limits_for(100) == (None, None)
    assert LocalShuffler(4, L=7, H=99).limits_for(100) == (7, 99)


def test_limit_aliases():
    assert LocalShuffler(4, L="H", H=37).limits_for(100) == (37, 37)
    assert LocalShuffler(4, L=37, H="L").limits_for(100) == (37, 37)
    assert LocalShuffler(4, L="H", H="auto").limits_for(100) == (1024, 1024)


def test_custom_l_above_automatic_h_is_not_an_error():
    shuffler = LocalShuffler(4, L=2000, H="auto", seed=42)
    assert shuffler.limits_for(100) == (2000, 1024)
    shuffled = shuffler.shuffle("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert len(shuffled) == 26


def test_circular_limit_alias_rejected():
    with pytest.raises(ValueError, match="circular"):
        LocalShuffler(4, L="H", H="L")


@pytest.mark.parametrize("name,kwargs", [
    ("L", {"L": -1}),
    ("H", {"H": -1}),
    ("L", {"L": 1.5}),
    ("H", {"H": "invalid"}),
    ("L", {"L": True}),
])
def test_invalid_limit_rejected(name, kwargs):
    with pytest.raises((TypeError, ValueError), match=name):
        LocalShuffler(4, **kwargs)


def test_soft_limit_zero_activates_free_first():
    shuffler = LocalShuffler(4, seed=42, L=0, H=None)
    shuffler.shuffle("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert shuffler.last_stats_["soft_chains"] > 0
    assert shuffler.last_stats_["fallbacks"] == 0


def test_hard_limit_zero_forces_and_tests_augmenting_fallback():
    source = "".join(chr(0x100 + index) for index in range(100))
    shuffler = LocalShuffler(8, seed=42, L=None, H=0)
    shuffled = shuffler.shuffle(source)
    _assert_bounded_permutation(source, shuffled, 8)
    assert shuffler.last_stats_["fallbacks"] == len(source)
    assert shuffler.last_stats_["turns"] == 0


def test_hard_limit_wins_when_l_is_greater():
    shuffler = LocalShuffler(4, seed=42, L=100, H=0)
    shuffler.shuffle("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert shuffler.last_stats_["fallbacks"] == 26
    assert shuffler.last_stats_["soft_chains"] == 0


def test_automatic_h_handles_only_rare_long_chains():
    source = np.arange(10_000)
    shuffler = LocalShuffler(32, seed=42)
    shuffler.shuffle(source)
    assert 0 < shuffler.last_stats_["fallbacks"] < 10
    assert shuffler.last_stats_["max_collisions"] >= 1024


@pytest.mark.parametrize("max_shift", [-1, -10])
def test_negative_shift_rejected(max_shift):
    with pytest.raises(ValueError):
        LocalShuffler(max_shift=max_shift)


@pytest.mark.parametrize("max_shift", [1.5, "3", True, None])
def test_non_integer_shift_rejected(max_shift):
    with pytest.raises(TypeError):
        LocalShuffler(max_shift=max_shift)


def test_non_sequence_input_rejected():
    with pytest.raises(TypeError):
        LocalShuffler(max_shift=2).shuffle([1, 2, 3])
