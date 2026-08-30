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

# Sequence distances and alignment scores

`kmer.distance` compares sequences directly. It provides edit metrics,
alignment scores, pairwise matrix helpers, and transforms that turn proper
distances into similarities suitable for kernel models.

<!-- Illustration placeholder; replace with a user-supplied asset later.
```{image} ../assets/guide/distances.svg
:alt: Four sequence comparisons illustrate a positional mismatch, an edit operation, a full global alignment, and a highlighted local alignment.
:width: 560px
:align: center
:class: guide-illustration
```
-->

## Learning goals

By the end of this guide, you will be able to:

- choose Hamming, Levenshtein, global alignment, or local alignment;
- distinguish a distance from a similarity score;
- construct square, cross, and condensed pairwise matrices;
- transform a proper distance into a kernel matrix.

## Choose a comparison

| Object | Measures | Length requirement | Direction |
|---|---|---|---|
| {class}`~kmer.distance.Hamming` | Position-wise substitutions | Equal lengths | Smaller is more similar |
| {class}`~kmer.distance.Levenshtein` | Insertions, deletions, substitutions | Any lengths | Smaller is more similar |
| {class}`~kmer.distance.NeedlemanWunsch` | Best global alignment | Any lengths | Raw score is larger for similarity |
| {class}`~kmer.distance.SmithWaterman` | Best local alignment | Any lengths | Raw score is larger for similarity |

RapidFuzz accelerates Hamming and Levenshtein, while parasail accelerates
alignment. Both are optional; the public behavior is also available through
Python fallbacks.

## Edit distances

```{code-cell} ipython3
:tags: [remove-stderr]

from kmer.distance import Hamming, Levenshtein

hamming = Hamming()
levenshtein = Levenshtein()

print("Hamming substitution:", hamming("ACGT", "ACGA"))
print("Levenshtein insertion:", levenshtein("ACGT", "ACGTA"))
print("Levenshtein deletion:", levenshtein("ACGT", "ACG"))
```

Hamming rejects unequal lengths. Levenshtein treats each insertion, deletion,
or substitution as one operation and is therefore appropriate when indels are
part of the expected variation.

## Pairwise matrices

Every distance inherits `cdist` and `pdist` from
{class}`~kmer.distance.BaseDistance`.

```{code-cell} ipython3
import numpy as np

sequences = ["ACGT", "ACGA", "ACGTA", "TTTT"]
D = levenshtein.cdist(sequences)
print(D)
print("shape:", D.shape)
print("symmetric:", np.allclose(D, D.T))
```

Pass a second collection for a rectangular query-by-reference matrix.

```{code-cell} ipython3
queries = ["ACGT", "TTTA"]
references = ["ACGA", "TTTT", "ACGTA"]
print(levenshtein.cdist(queries, references))
```

`pdist` returns SciPy's condensed upper-triangle representation.

```{code-cell} ipython3
from scipy.spatial.distance import squareform

condensed = levenshtein.pdist(sequences)
full_matrix = squareform(condensed)
print("condensed:", condensed)
print("full matrix:\n", full_matrix)
```

## Global and local alignment

Needleman-Wunsch scores an end-to-end alignment. Smith-Waterman finds the best
matching subsequence.

```{code-cell} ipython3
from kmer.distance import NeedlemanWunsch, SmithWaterman

nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
sw = SmithWaterman(match=1, mismatch=-1, gap=-1)

print("global score:", nw("ACGT", "TTACGTTT"))
print("local score:", sw("ACGT", "TTACGTTT"))
```

These defaults return **similarity scores**, not distances. Use
`as_distance=True` before passing an alignment object to code that expects
smaller values for closer sequences.

```{code-cell} ipython3
nw_distance = NeedlemanWunsch(
    match=1,
    mismatch=-1,
    gap=-1,
    as_distance=True,
)
print("identical:", nw_distance("ACGT", "ACGT"))
print("one deletion:", nw_distance("ACGT", "ACG"))
```

`NUC4.4` is built in. BLOSUM and PAM matrices require parasail.

```{code-cell} ipython3
nuc44 = NeedlemanWunsch(substitution_matrix="NUC4.4", gap=-4)
print("NUC4.4 identical score:", nuc44("ACGT", "ACGT"))
```

## Turn a distance into a kernel

{class}`~kmer.distance.DistanceKernel` expects a wrapped measure that returns
**distances**: smaller values mean more similar sequences. It transforms those
distances into **similarity** values. For the RBF transform,

$$
K(a,b) = \exp\left(-\gamma d(a,b)^2\right).
$$

```{important}
Input and output use opposite conventions. `DistanceKernel` does not
automatically reinterpret a raw alignment similarity score. If the wrapped
measure has `is_distance=False`, construction emits a `UserWarning` but remains
backward compatible and continues. Hamming and Levenshtein can be wrapped
directly; construct Needleman-Wunsch or Smith-Waterman with `as_distance=True`
before wrapping it.
```

```{code-cell} ipython3
from kmer.distance import DistanceKernel

kernel = DistanceKernel(Levenshtein(), transform="rbf", gamma=0.2)
K = kernel.kernel_matrix(sequences)
print(np.round(K, 3))
print("input is a distance:", kernel.distance.is_distance)
print("output is a similarity:", not kernel.is_distance)
print("unit diagonal:", np.allclose(np.diag(K), 1.0))
```

Available transforms also include `linear`, `poly`, `sigmoid`, `laplacian`,
`exponential`, `exponential_clipped`, and `cauchy`. A transformed symmetric
matrix is not automatically positive semidefinite for every distance and
transform combination; check eigenvalues when that property is required.

## Practical guidance

- Use Hamming only when coordinates are already aligned and lengths match.
- Use Levenshtein for simple edit variation without a biological scoring
  matrix.
- Use Needleman-Wunsch for end-to-end homology and Smith-Waterman for embedded
  local matches.
- Treat raw alignment output as a score. Set `as_distance=True` for KNN or a
  distance transform.
- Pairwise methods scale quadratically in collection size; use
  {class}`kmer.ann.ANNIndex` when exhaustive comparison becomes too costly.

## Next steps

- Fit {class}`kmer.models.KNNClassifier` or
  {class}`kmer.models.KernelSVM` in {doc}`models`.
- Scale nearest-neighbor search with {doc}`ann`.
- Consult {doc}`../api/distance` for all parameters and transforms.
