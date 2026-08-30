"""Tests for locally bounded shuffling of sampled sequence chunks."""

from collections import Counter

import pytest

from kmer.background import Chunker, LocalChunker, LocalShuffler
from kmer.background._base import BaseBackgroundModel


def _make_local_chunker(
    *,
    min_size=2,
    max_size=5,
    max_shift=3,
    chunk_seed=42,
    shuffle_seed=123,
    **chunker_kwargs,
):
    return LocalChunker(
        chunker=Chunker(
            min_size=min_size,
            max_size=max_size,
            seed=chunk_seed,
            **chunker_kwargs,
        ),
        shuffler=LocalShuffler(max_shift=max_shift, seed=shuffle_seed),
    )


def _assert_bounded_chunk_permutation(source, shuffled, max_shift):
    positions = {chunk: index for index, chunk in enumerate(shuffled)}
    for original_index, chunk in enumerate(source):
        assert abs(positions[chunk] - original_index) <= max_shift


def test_is_background_model():
    assert isinstance(_make_local_chunker(), BaseBackgroundModel)


def test_constructor_requires_component_instances():
    with pytest.raises(TypeError, match="chunker"):
        LocalChunker(Chunker, LocalShuffler(2, seed=42))
    with pytest.raises(TypeError, match="shuffler"):
        LocalChunker(Chunker(2, 4, seed=42), LocalShuffler)


def test_chunk_matches_explicit_component_composition():
    sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expected_chunks = Chunker(2, 5, seed=42).slice_chunks(sequence)
    shuffled_chunks = LocalShuffler(3, seed=123).shuffle(expected_chunks)

    observed = _make_local_chunker().chunk(sequence)
    assert observed == "".join(shuffled_chunks)


def test_fixed_size_chunks_obey_strict_displacement_bound():
    sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    original_chunks = [sequence[index:index + 2] for index in range(0, 26, 2)]
    output = _make_local_chunker(
        min_size=2,
        max_size=2,
        max_shift=3,
    ).chunk(sequence)
    output_chunks = [output[index:index + 2] for index in range(0, 26, 2)]
    _assert_bounded_chunk_permutation(original_chunks, output_chunks, 3)


def test_length_and_composition_are_preserved():
    sequence = "AAACCCGGGTTTAAACCCGGGTTT"
    output = _make_local_chunker().chunk(sequence)
    assert len(output) == len(sequence)
    assert Counter(output) == Counter(sequence)


def test_bytes_input_preserves_type_length_and_composition():
    sequence = b"AAACCCGGGTTTAAACCCGGGTTT"
    output = _make_local_chunker().chunk(sequence)
    assert isinstance(output, bytes)
    assert len(output) == len(sequence)
    assert Counter(output) == Counter(sequence)


def test_empty_and_short_sequences_are_preserved():
    model = _make_local_chunker(min_size=5, max_size=8)
    assert model.chunk("") == ""
    assert model.chunk("AC") == "AC"


def test_optional_strand_flipping_comes_from_chunker():
    model = _make_local_chunker(
        min_size=3,
        max_size=3,
        max_shift=0,
        flip_strand=True,
        flip_strand_prob=1.0,
    )
    assert model.chunk("ACGGTT") == "CGTAAC"


def test_same_seeds_produce_same_call_sequence():
    sequence = "ACGT" * 30
    first = _make_local_chunker()
    second = _make_local_chunker()
    assert [first.chunk(sequence) for _ in range(4)] == [
        second.chunk(sequence) for _ in range(4)
    ]


def test_perturb_many_matches_independent_serial_calls():
    sequences = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"] * 8
    serial = _make_local_chunker(min_size=2, max_size=5, max_shift=4)
    expected = [serial.chunk(sequence) for sequence in sequences]

    batch = _make_local_chunker(min_size=2, max_size=5, max_shift=4)
    observed = batch.perturb_many(sequences)
    assert observed == expected
    assert len(set(observed)) > 1


def test_perturb_many_is_reproducible_across_n_jobs():
    sequences = ["ACGT" * 30] * 8
    serial = _make_local_chunker().perturb_many(sequences, n_jobs=1)
    nominal_parallel = _make_local_chunker().perturb_many(sequences, n_jobs=4)
    assert nominal_parallel == serial


def test_perturb_aliases_chunk():
    sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expected = _make_local_chunker().chunk(sequence)
    assert _make_local_chunker().perturb(sequence) == expected


@pytest.mark.parametrize("n_jobs", [0, -1, 1.5, True])
def test_invalid_perturb_many_n_jobs_rejected(n_jobs):
    with pytest.raises((TypeError, ValueError), match="n_jobs"):
        _make_local_chunker().perturb_many(["ACGT"], n_jobs=n_jobs)


def test_chunk_many_is_not_public():
    assert not hasattr(_make_local_chunker(), "chunk_many")


def test_raw_sequence_interface_rejects_preselected_chunks():
    with pytest.raises(TypeError, match="sequence"):
        _make_local_chunker().chunk(["promoter", "exon", "intron"])


def test_getstate_setstate_roundtrip():
    sequence = "ACGT" * 50
    first = _make_local_chunker()
    first.chunk(sequence)
    state = first.getstate()
    expected = first.chunk(sequence)

    second = _make_local_chunker(chunk_seed=999, shuffle_seed=999)
    second.setstate(state)
    assert second.chunk(sequence) == expected


def test_invalid_state_rejected():
    with pytest.raises(TypeError, match="state"):
        _make_local_chunker().setstate(b"not-a-pair")


def test_component_properties_and_stats_are_exposed():
    model = _make_local_chunker(max_shift=4)
    model.chunk("ACGT" * 20)
    assert isinstance(model.chunker, Chunker)
    assert isinstance(model.shuffler, LocalShuffler)
    assert model.max_shift == 4
    assert model.last_stats_["turns"] >= 0


def test_repr_shows_both_components():
    representation = repr(_make_local_chunker(max_shift=3))
    assert "LocalChunker" in representation
    assert "Chunker" in representation
    assert "LocalShuffler" in representation
    assert "max_shift=3" in representation
