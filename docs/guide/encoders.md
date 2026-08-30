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

# K-mer encoders

The classes in `kmer.encoders` turn nucleotide sequences into sparse count
matrices. Rows represent sequences; columns represent exact, gappy, or
mismatch-tolerant k-mer features. The result can be passed directly to
scikit-learn estimators and sparse-matrix tools.

## Learning goals

By the end of this guide, you will be able to:

- choose among spectrum, gappy, and mismatch encoders;
- interpret matrix dimensions, counts, and feature names;
- collapse reverse-complement pairs when strand should not matter;
- anticipate the principal memory and runtime trade-offs.

## Choose an encoder

| Encoder | A feature represents | Use it when |
|---|---|---|
| {class}`~kmer.encoders.SpectrumEncoder` | One exact k-mer | Exact local composition is informative and speed matters most. |
| {class}`~kmer.encoders.GappyEncoder` | Bases selected by a positional mask | Motifs may contain unconstrained internal positions. |
| {class}`~kmer.encoders.MismatchEncoder` | A k-mer neighborhood within Hamming distance `m` | Related motifs should share features despite substitutions. |

All three return {class}`scipy.sparse.csr_matrix` objects. Keep these matrices
sparse: converting a large feature space to a dense array can consume much more
memory.

## Setup

```{code-cell} ipython3
import numpy as np

from kmer.encoders import (
    GappyEncoder,
    MismatchEncoder,
    SpectrumEncoder,
    generate_masks,
)

sequences = [
    "ACGTACGTACGT",
    "ACGTACGAACGT",
    "TTTTAAAACCCC",
]
```

## Exact k-mer counts

`SpectrumEncoder(k=4)` has one column for every four-base word. A sequence of
length $n$ contributes $n-k+1$ overlapping k-mers when all characters are
valid.

```{code-cell} ipython3
spectrum = SpectrumEncoder(k=4)
X_spectrum = spectrum.fit_transform(sequences)

print("shape:", X_spectrum.shape)
print("storage:", X_spectrum.format)
print("counts per sequence:", X_spectrum.sum(axis=1).A1)
```

The feature matrix has $4^4=256$ columns, but only observed k-mers occupy
storage. Feature names map column indices back to nucleotide words.

```{code-cell} ipython3
feature_names = spectrum.get_feature_names_out()
first_row = X_spectrum.getrow(0)
observed = [
    (feature_names[index], int(count))
    for index, count in zip(first_row.indices, first_row.data)
]
print(observed)
```

### Make strand orientation irrelevant

Set `canonical_rc=True` when a k-mer and its reverse complement should be the
same feature. This is common for double-stranded DNA assays, but it is not
appropriate when strand direction carries biological meaning.

```{code-cell} ipython3
forward = "AACCGTTA"
reverse_complement = "TAACGGTT"

canonical = SpectrumEncoder(k=3, canonical_rc=True)
X_rc = canonical.fit_transform([forward, reverse_complement])
print("identical features:", (X_rc[0] != X_rc[1]).nnz == 0)
```

## Gappy k-mers

A gappy mask uses `*` for a retained base and `-` for a don't-care position.
For example, `*--*` compares the first and fourth bases of each four-base
window. Masks must start and end with `*`.

```{code-cell} ipython3
gappy = GappyEncoder(masks=["*--*", "*-*-*"])
X_gappy = gappy.fit_transform(sequences)

print("masks:", gappy.masks_)
print("shape:", X_gappy.shape)
print("nonzero entries:", X_gappy.nnz)
```

Use {func}`kmer.encoders.generate_masks` or the `L`, `g_min`, and `g_max`
constructor arguments to enumerate masks instead of specifying each one.

```{code-cell} ipython3
masks = generate_masks(L_min=5, L_max=5, g_min=1, g_max=2)
automatic_gappy = GappyEncoder(L=5, g_min=1, g_max=2)

print("generated masks:", masks)
print("encoder uses the same masks:", automatic_gappy.masks_ == masks)
```

The feature space grows with both the number of masks and the number of
concrete positions in each mask. Prefer masks motivated by the expected motif
structure rather than enumerating a broad range without checking its size.

## Mismatch neighborhoods

`MismatchEncoder(k, m)` increments every feature within Hamming distance `m` of
an observed k-mer. Setting `m=0` recovers the exact spectrum representation.

```{code-cell} ipython3
exact = SpectrumEncoder(k=4).fit_transform(sequences)
mismatch_zero = MismatchEncoder(k=4, m=0).fit_transform(sequences)

print("m=0 equals spectrum:", np.array_equal(
    exact.toarray(), mismatch_zero.toarray()
))
```

Increasing `m` lets related words share signal, but each observed k-mer updates
many more features.

```{code-cell} ipython3
for m in (0, 1, 2):
    encoded = MismatchEncoder(k=5, m=m).fit_transform([sequences[0]])
    print(f"m={m}: nonzero={encoded.nnz:4d}, total count={int(encoded.sum()):4d}")
```

For one observed k-mer, the size of the radius-`m` neighborhood is

$$
\sum_{i=0}^{m} \binom{k}{i} 3^i.
$$

This combinatorial growth is the main cost of mismatch encoding.

## Practical guidance

- Start with `SpectrumEncoder`; use a more flexible encoder only when the
  biological hypothesis justifies it.
- Keep matrices sparse and use estimators that accept sparse input.
- Choose `canonical_rc=True` deliberately rather than automatically.
- Fit the encoder once and reuse it for training and test sequences so feature
  interpretation remains consistent.
- Feature spaces grow exponentially with `k`; all current encoders restrict
  `k` or concrete mask positions to keep indexing practical.
- Inputs are nucleotide strings or bytes. Validate ambiguous symbols against
  the specific encoder behavior before processing a large dataset.

## Next steps

- Use encoded matrices with {class}`kmer.models.LinearSVM` for supervised
  classification.
- Use an encoder with {class}`kmer.models.DifferentialKmerScorer` to rank
  enriched k-mers.
- Compare feature-based methods with {class}`kmer.kernels.GKMKernel` when a
  kernel representation better matches the task.
- Consult {doc}`../api/encoders` for complete signatures and attributes.
