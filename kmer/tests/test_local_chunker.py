"""Tests for compositional position-bounded chunk permutation."""

import numpy as np
import pytest

from kmer.perturb import LocalChunker, LocalShuffler
from kmer.perturb._base import BaseBackgroundModel


def _assert_chunk_displacement(chunks, shuffled, max_shift):
    positions = {id(chunk): index for index, chunk in enumerate(shuffled)}
    for original_index, chunk in enumerate(chunks):
        assert abs(positions[id(chunk)] - original_index) <= max_shift


def test_is_background_model():
    assert isinstance(LocalChunker(2, seed=42), BaseBackgroundModel)


@pytest.mark.parametrize("length", [0, 1, 2, 7, 32, 100])
@pytest.mark.parametrize("max_shift", [0, 1, 2, 8, 32])
def test_list_chunk_displacement_is_strictly_bounded(length, max_shift):
    chunks = [object() for _ in range(length)]
    chunker = LocalChunker(max_shift, seed=42)
    for _ in range(20):
        shuffled = chunker.shuffle_chunks(chunks)
        _assert_chunk_displacement(chunks, shuffled, max_shift)


def test_list_preserves_outer_type_and_chunk_identity():
    chunks = [[index] for index in range(30)]
    shuffled = LocalChunker(4, seed=42).shuffle_chunks(chunks)
    assert isinstance(shuffled, list)
    assert {id(chunk) for chunk in shuffled} == {id(chunk) for chunk in chunks}
    assert all(shuffled.count(chunk) == 1 for chunk in chunks)


def test_tuple_preserves_outer_type_and_chunk_identity():
    chunks = tuple([index] for index in range(30))
    shuffled = LocalChunker(4, seed=42).shuffle_chunks(chunks)
    assert isinstance(shuffled, tuple)
    assert {id(chunk) for chunk in shuffled} == {id(chunk) for chunk in chunks}


@pytest.mark.parametrize("dtype", [np.int16, np.float64, object])
def test_numpy_preserves_shape_dtype_and_values(dtype):
    chunks = np.arange(50).astype(dtype)
    shuffled = LocalChunker(5, seed=42).shuffle_chunks(chunks)
    assert isinstance(shuffled, np.ndarray)
    assert shuffled.shape == chunks.shape
    assert shuffled.dtype == chunks.dtype
    assert sorted(shuffled.tolist()) == sorted(chunks.tolist())


def test_numpy_object_array_preserves_chunk_identity():
    chunks = np.empty(20, dtype=object)
    chunks[:] = [[index] for index in range(20)]
    shuffled = LocalChunker(3, seed=42).shuffle_chunks(chunks)
    assert {id(chunk) for chunk in shuffled} == {id(chunk) for chunk in chunks}
    _assert_chunk_displacement(chunks, shuffled, 3)


def test_same_seed_matches_local_shuffler_permutation():
    chunks = [f"chunk-{index}" for index in range(50)]
    permutation = LocalShuffler(6, seed=123).permutation(len(chunks))
    expected = [chunks[index] for index in permutation]
    assert LocalChunker(6, seed=123).shuffle_chunks(chunks) == expected


def test_same_seed_produces_same_call_sequence():
    chunks = list(range(100))
    first = LocalChunker(8, seed=42)
    second = LocalChunker(8, seed=42)
    assert [first.shuffle_chunks(chunks) for _ in range(4)] == [
        second.shuffle_chunks(chunks) for _ in range(4)
    ]


def test_getstate_setstate_roundtrip():
    chunks = list(range(100))
    first = LocalChunker(8, seed=42)
    first.shuffle_chunks(chunks)
    state = first.getstate()
    expected = first.shuffle_chunks(chunks)

    second = LocalChunker(8, seed=999)
    second.setstate(state)
    assert second.shuffle_chunks(chunks) == expected


def test_shuffle_and_perturb_are_aliases():
    chunks = list(range(50))
    expected = LocalChunker(4, seed=42).shuffle_chunks(chunks)
    assert LocalChunker(4, seed=42).shuffle(chunks) == expected
    assert LocalChunker(4, seed=42).perturb(chunks) == expected


def test_perturb_many_matches_serial_calls():
    collections = [list(range(20)), tuple(range(30))]
    serial = LocalChunker(3, seed=42)
    expected = [serial.perturb(chunks) for chunks in collections]
    batch = LocalChunker(3, seed=42)
    assert batch.perturb_many(collections) == expected


def test_permutation_and_limits_delegate():
    chunker = LocalChunker(32, seed=42, L="H", H=4096)
    assert chunker.limits_for(200) == (4096, 4096)
    permutation = chunker.permutation(200)
    assert sorted(permutation.tolist()) == list(range(200))


def test_stats_delegate_and_forced_fallback_works():
    chunks = list(range(100))
    chunker = LocalChunker(8, seed=42, L=None, H=0)
    shuffled = chunker.shuffle_chunks(chunks)
    assert sorted(shuffled) == chunks
    assert chunker.last_stats_["fallbacks"] == len(chunks)


def test_input_collection_is_not_mutated():
    chunks = [[index] for index in range(30)]
    snapshot = list(chunks)
    LocalChunker(4, seed=42).shuffle_chunks(chunks)
    assert chunks == snapshot
    assert all(left is right for left, right in zip(chunks, snapshot))


@pytest.mark.parametrize("chunks", ["abc", b"abc", range(10), iter(range(10))])
def test_unsupported_collection_rejected(chunks):
    with pytest.raises(TypeError, match="list, tuple"):
        LocalChunker(2, seed=42).shuffle_chunks(chunks)


def test_multidimensional_numpy_collection_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        LocalChunker(2, seed=42).shuffle_chunks(np.arange(12).reshape(3, 4))


def test_repr_shows_configuration():
    representation = repr(LocalChunker(3, seed=42, L=10, H=100))
    assert "LocalChunker" in representation
    assert "max_shift=3" in representation
    assert "L=10" in representation
    assert "H=100" in representation
