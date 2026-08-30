---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Background models

`kmer.background` generates control sequences or bounded permutations while
preserving selected properties of the input. The correct null model depends on
which sequence structure should remain unchanged.

<!-- Illustration placeholder; replace with a user-supplied asset later.
```{image} ../assets/guide/background.svg
:alt: An original colored sequence is transformed by k-mer-preserving shuffling, chunk rearrangement, and locally bounded movement.
:width: 560px
:align: center
:class: guide-illustration
```
-->

## Learning goals

By the end of this guide, you will be able to:

- choose composition-preserving, block-based, or locally bounded perturbation;
- understand endpoint handling in k-mer-preserving shuffles;
- verify preservation and displacement invariants;
- use the common background-model interface in downstream estimators.

## Choose a background model

| Object | Preserves | Changes |
|---|---|---|
| {class}`~kmer.background.KmerShuffler` | Selected k-mer composition | Global sequence order |
| {class}`~kmer.background.Chunker` | Sequence length and block contents | Block boundaries/order; optionally strand |
| {class}`~kmer.background.LocalShuffler` | Items and a strict displacement bound | Local item order |
| {class}`~kmer.background.LocalChunker` | Sampled chunk contents and a chunk-position displacement bound | Local chunk order |

All implement {class}`~kmer.background.BaseBackgroundModel`, whose common
methods are `perturb` and `perturb_many`.

## Preserve k-mer composition

`KmerShuffler(k)` constructs a De Bruijn multigraph and samples an Eulerian
path. With `endpoints="preserve"`, the output has exactly the same multiset of
length-`k` words as the input.

```{code-cell} ipython3
from collections import Counter

from kmer.background import KmerShuffler

sequence = "AAACCCGGGTTTAAACCCGGGTTT"
shuffler = KmerShuffler(k=2, seed=42, endpoints="preserve")
shuffled = shuffler.shuffle(sequence)

count_kmers = lambda s, k: Counter(
    s[i : i + k] for i in range(len(s) - k + 1)
)
print("input: ", sequence)
print("output:", shuffled)
print("dinucleotides preserved:",
      count_kmers(sequence, 2) == count_kmers(shuffled, 2))
```

Endpoint policy is part of the null model:

- `preserve` keeps the original Eulerian path endpoints and exact k-mer counts;
- `free` allows endpoints to change by closing and breaking a circuit;
- `crop` removes the terminal `(k-1)`-mers and shuffles the interior circuit.

The default is `free`. Choose `preserve` when exact composition is the primary
invariant, and account for the shorter output if using `crop`.

## Rearrange variable-size blocks

{class}`~kmer.background.Chunker` splits a sequence into blocks, permutes the
blocks, and concatenates them.

```{code-cell} ipython3
from kmer.background import Chunker

chunker = Chunker(min_size=3, max_size=5, seed=42)
chunked = chunker.chunk(sequence)
print("output:", chunked)
print("length preserved:", len(chunked) == len(sequence))
print("base composition preserved:", Counter(chunked) == Counter(sequence))
```

`algorithm="random"` samples sizes greedily. `algorithm="backtrack"` uses
dynamic programming to seek an exact size decomposition. If a short residual
remains, `residual_chunk` controls whether it is retained at an endpoint,
inserted randomly, used to extend one block, or distributed among blocks.

Set `flip_strand=True` to reverse-complement selected chunks. That option
changes direct base ordering and requires valid IUPAC characters.

## Strictly bound local displacement

`LocalShuffler(max_shift=S)` treats a string, byte sequence, list, tuple, or
one-dimensional NumPy array as an item sequence. Every item moves by at most
`S` positions. The `permutation` method makes that invariant directly
inspectable.

For nucleotide sequences, bounded movement can destroy complex,
order-dependent signals while preserving local nucleotide composition at
scales broader than `max_shift`. This is a local rather than an exact
fixed-window invariant: nucleotides near a window boundary can cross it, but
none can move farther than the configured bound.

```{code-cell} ipython3
import numpy as np

from kmer.background import LocalShuffler

local = LocalShuffler(max_shift=3, seed=42)
permutation = local.permutation(16)
positions = np.arange(len(permutation))
print("permutation:", permutation)
print("maximum displacement:", np.abs(permutation - positions).max())
print("collision statistics:", local.last_stats_)
```

`L` and `H` control collision-chain fallback behavior. Their `"auto"` defaults
adapt to input length and effective shift; change them only when profiling or
algorithm experiments justify it.

```{code-cell} ipython3
text = "ABCDEFGHIJKLMNOP"
locally_shuffled = LocalShuffler(max_shift=2, seed=7).shuffle(text)
print(text)
print(locally_shuffled)
```

## Rearrange sampled chunks locally

`LocalChunker` composes a regular `Chunker` with a `LocalShuffler`. The first
component samples variable-size chunks from each raw sequence; the second moves
each whole chunk by at most `max_shift` chunk positions. The chunks are then
concatenated into the same sequence type.

```{code-cell} ipython3
from kmer.background import Chunker, LocalChunker, LocalShuffler

local_chunks = LocalChunker(
    chunker=Chunker(min_size=3, max_size=5, seed=42),
    shuffler=LocalShuffler(max_shift=2, seed=43),
)
locally_chunked = local_chunks.chunk(sequence)
print("input: ", sequence)
print("output:", locally_chunked)
print("length preserved:", len(locally_chunked) == len(sequence))
```

The two injected components retain separate RNG states. Chunk boundaries are
sampled independently for every input, so even equal-length sequences may have
different numbers of chunks. `max_shift` is measured in chunk positions, not
nucleotide positions. If chunks are already available, pass their list or
tuple directly to `LocalShuffler` instead.

## Apply models to a set of sequences

Every model uses the single common `perturb_many(sequences, n_jobs=1)` batch
interface. There are no model-specific `shuffle_many` or `chunk_many` aliases.
`LocalShuffler.perturb_many` accepts strings, arrays, lists, and tuples.

```{code-cell} ipython3
sequences = [
    "ACGTACGTACGT",
    "TTTTAAAACCCC",
    "GCGCGCGCGCGC",
]
models = {
    "k-mer": KmerShuffler(k=1, seed=11, endpoints="preserve"),
    "chunks": Chunker(min_size=2, max_size=4, seed=12),
    "local": LocalShuffler(max_shift=2, seed=13),
    "local chunks": LocalChunker(
        chunker=Chunker(min_size=2, max_size=4, seed=14),
        shuffler=LocalShuffler(max_shift=1, seed=15),
    ),
}

background_sets = {
    name: model.perturb_many(sequences)
    for name, model in models.items()
}
for name, backgrounds in background_sets.items():
    print(name, [len(background) for background in backgrounds])
```

Each batch item receives a newly sampled permutation; one permutation is not
reused across the set. Independent random draws can nevertheless coincide by
chance, especially for short or highly constrained sequences. State advances
between items and calls, while fresh instances with the same seeds reproduce
the same output sequence.

Any custom subclass implementing `perturb` can be supplied to
{class}`kmer.models.KmerMultinomialNB`; the base class supplies a serial
`perturb_many` fallback.

## Practical guidance

- State explicitly which properties the null model preserves.
- Verify invariants on representative data before large-scale generation.
- Use distinct seeds for independent experimental replicates.
- Remember that a highly constrained shuffle may have few distinct outputs.
- Keep positive sequences and their generated backgrounds in the same
  train/test partition to prevent related controls from leaking across folds.

## Next steps

- Use backgrounds with differential scoring in {doc}`models`.
- Compare resulting feature distributions with {doc}`encoders`.
- Consult {doc}`../api/background` for endpoint, residual, strand, and local
  collision-control parameters.
