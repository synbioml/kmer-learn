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

# K-mer kernels

`kmer.kernels` implements the gkmSVM/LS-GKM family introduced by
[Ghandi et al. (2014), *Enhanced Regulatory Sequence Prediction Using Gapped
k-mer Features*](https://doi.org/10.1371/journal.pcbi.1003711). These kernels
compare sequences through shared gapped words without materializing an
explicit $4^k$ feature matrix.

## Learning goals

By the end of this guide, you will be able to:

- interpret the core `L`, `k`, and `d` parameters;
- build self- and cross-kernel matrices;
- choose reverse-complement handling and mismatch weighting;
- distinguish full-sequence, weighted, sliding-query, and windowed kernels.

## Kernel family

| Object | Use it when | Output |
|---|---|---|
| {class}`~kmer.kernels.GKMKernel` | Motif position within a sequence is not fixed | 2D matrix |
| {class}`~kmer.kernels.WGKMKernel` | Motifs near a chosen position should matter more | 2D matrix; multiple offsets may add an axis |
| {class}`~kmer.kernels.WindowedGKMKernel` | Similarity should be resolved along aligned windows | 3D tensor |
| {class}`~kmer.kernels.WindowedWGKMKernel` | Windowed comparison also needs positional weighting | 3D self-kernel tensor |

## Understand `L`, `k`, and `d`

- `L` is the length of each extracted word, between 2 and 12.
- `k` is the number of informative positions within that word.
- `L-k` positions may be gaps.
- `d` is the maximum mismatch level and must satisfy `0 <= d <= L-k`.

The GKM-family constructors use the literature-style starting point `L=10`,
`k=6`, `d=3`, `kernel_type="full"`, and strand-specific `use_rc=False`. The
examples below retain those defaults unless they demonstrate another behavior.
Treat them as a defensible baseline rather than universally optimal settings:
evaluate sensitivity on held-out data for each biological application.

## Build a normalized kernel matrix

```{code-cell} ipython3
import numpy as np

from kmer.kernels import GKMKernel

sequences = [
    "ACGTACGTACGTACGTACGT",
    "ACGTACGTACGTACGTACGA",
    "TTTTAAAAGGGGCCCCAAAA",
    "GCGCGCGCATATATATGCGC",
]

kernel = GKMKernel()
kernel.set_references(sequences)
K = np.asarray(kernel.kernel())

print("shape:", K.shape)
print("symmetric:", np.allclose(K, K.T))
print("diagonal:", np.round(np.diag(K), 6))
```

`set_references` builds the native index. Calling `kernel()` without queries
returns the reference self-kernel.

## Compare queries with references

```{code-cell} ipython3
queries = [
    "ACGTACGTACGTACGTTCGT",
    "CCCCGGGGTTTTAAAACCCC",
]
K_cross = np.asarray(kernel.kernel(X_query=queries))
print("query-by-reference shape:", K_cross.shape)
print(np.round(K_cross, 3))
```

Passing `X_ref=` to `kernel` replaces the stored reference collection.
Repeated-query workflows should call `set_references` once and reuse the built
index.

## Weight schemes and reverse complements

`kernel_type` controls how mismatch contributions are weighted:

- `full`: exact full weighting;
- `estimated_full`: an approximation intended to reduce work;
- `truncated`: ignores contributions beyond the configured mismatch range.

Set `use_rc=True` when forward and reverse-complement motif occurrences should
contribute equivalently. Leave it false for strand-specific questions.

## Post-transforms

A normalized GKM value can be transformed inside the kernel object.

```{code-cell} ipython3
rbf_kernel = GKMKernel(
    transform="rbf",
    transform_gamma=2.0,
)
rbf_kernel.set_references(sequences)
K_rbf = np.asarray(rbf_kernel.kernel())
print(np.round(K_rbf, 3))
```

Supported transforms are `linear`, `rbf`, `poly`, `sigmoid`, `exponential`,
and `exponential_clipped`. Keep the untransformed kernel as the baseline and
validate that a transform improves held-out behavior.

## Positional weighting

{class}`~kmer.kernels.WGKMKernel` emphasizes words near a center. Reference and
query sequences must have equal lengths.

```{code-cell} ipython3
from kmer.kernels import WGKMKernel

weighted = WGKMKernel()
weighted.set_references(sequences)
K_weighted = np.asarray(weighted.kernel())
print("weighted shape:", K_weighted.shape)
print(np.round(K_weighted, 3))
```

The default positional profile is `triangular`. Alternatives are
`epanechnikov`, `gaussian`, `laplacian`, and `cauchy`; their scale parameters
should reflect the expected positional spread. `center_offset` moves the
default center relative to the sequence midpoint.

## Scan a long query

A non-windowed `GKMKernel` can slide a query window against fixed references.
Different long queries may have different lengths, so the method returns one
matrix per query.

```{code-cell} ipython3
long_query = "TTTTAAAA" + sequences[0] + "CCCCGGGG"
scan = kernel.sliding_window_query_kernel(
    [long_query],
    window=20,
    shift=4,
)
print("number of query results:", len(scan))
print("windows-by-references:", scan[0].shape)
```

## Compare aligned windows

Windowed kernels compare the same window positions across equal-length
sequences and return `(n_windows, n_sequences, n_sequences)`.

```{code-cell} ipython3
from kmer.kernels import WindowedGKMKernel

windowed = WindowedGKMKernel(
    window=12,
    shift=4,
)
windowed.set_references(sequences)
T = np.asarray(windowed.kernel())
print("window tensor shape:", T.shape)
```

Current limitations matter when choosing a weighted variant:

- weighted sliding queries are not supported;
- `WindowedWGKMKernel` supports self-window evaluation but not cross-window
  evaluation;
- all sequences in a windowed matrix must have equal lengths.

## Practical guidance

- Ensure every sequence is at least `L` bases long.
- Reuse reference indexes instead of rebuilding for each query batch.
- Use reverse complements only when strand invariance is scientifically valid.
- Use `WGKMKernel` only when sequence coordinates are meaningfully aligned.
- Keep train/test splitting outside the kernel to avoid evaluating parameter
  choices on the same observations used for fitting.

## Next steps

- Fit a precomputed-kernel classifier in {doc}`models`.
- Use {doc}`ann` for approximate search with GKM-derived distance.
- Consult {doc}`../api/kernels` for all weighting, offset, padding, and
  transform parameters.
