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

# Approximate nearest-neighbor search

`kmer.ann` uses a native HNSW graph to search a sequence collection without
computing every query-to-reference comparison. It trades exactness for speed
and memory-efficient repeated retrieval.

## Learning goals

By the end of this guide, you will be able to:

- build and query an {class}`~kmer.ann.ANNIndex`;
- interpret returned reference indices and distances;
- choose among edit, alignment, and GKM metrics;
- tune construction and search effort while measuring recall.

## When to use ANN

Use exhaustive `cdist` when the collection is small or exact neighbors are
required. Use ANN when:

- the reference collection is reused for many queries;
- exhaustive pairwise comparison is too expensive;
- a small loss in neighbor recall is acceptable and measured.

HNSW results are approximate. Always benchmark recall and query time on data
representative of the intended deployment.

## Build an index

```{code-cell} ipython3
from kmer.ann import ANNIndex

references = [
    "ACGTACGTACGT",
    "ACGTACGTACGA",
    "AACGTACGTACG",
    "TTTTAAAACCCC",
    "GGGGCCCCAAAA",
    "TATATATATATA",
]

index = ANNIndex(
    references,
    metric="levenshtein",
    M=16,
    ef_construction=200,
    ef_search=50,
)
print("indexed sequences:", index.get_size())
```

The index stores integer positions, not application labels. Keep the original
reference collection or a parallel metadata table so returned positions can be
mapped back to records.

## Query one or many sequences

`query` always returns one result per query. Each result is a pair containing a
list of reference indices and a list of distances, ordered nearest first.

```{code-cell} ipython3
result = index.query("ACGTACGTACGG", k=3)
indices, distances = result[0]
print("indices:", indices)
print("distances:", distances)
print("sequences:", [references[i] for i in indices])
```

A list input performs a batch query.

```{code-cell} ipython3
batch_results = index.query(
    ["ACGTACGTACGG", "TTTTAAAACCCG"],
    k=2,
)
for query_result in batch_results:
    print(query_result)
```

If `k` exceeds the collection size, at most the available number of references
is returned.

## Choose a metric

| `metric` | Internal distance | Important constraint |
|---|---|---|
| `levenshtein` | Insertions, deletions, substitutions | General-purpose variable-length search |
| `hamming` | Position mismatches | Use equal-length, aligned sequences |
| `nw` | Maximum possible score minus global-alignment score | Configure `match`, `mismatch`, and `gap` |
| `sw` | Maximum possible score minus local-alignment score | Behaves as a search dissimilarity |
| `gkm` | `1 -` normalized GKM similarity | Configure GKM parameters; sequences must support `gkm_L` |

The ANN implementations are native and configured directly on `ANNIndex`; they
are related to, but do not receive, Python distance or kernel objects.

## Search with GKM similarity

```{code-cell} ipython3
gkm_index = ANNIndex(
    references,
    metric="gkm",
    M=16,
    ef_construction=200,
    ef_search=50,
    gkm_L=6,
    gkm_k=4,
    gkm_d=2,
    gkm_use_rc=True,
)
gkm_indices, gkm_distances = gkm_index.query(
    "ACGTACGTACGG",
    k=3,
)[0]
print("indices:", gkm_indices)
print("1 - GKM similarity:", [round(d, 4) for d in gkm_distances])
```

GKM ANN currently uses the truncated mismatch-weight scheme. Use the full
kernel API when other schemes, post-transforms, weighted positions, or complete
kernel matrices are required.

## Tune construction and search

- `M` controls graph connectivity. Larger values generally improve recall but
  increase memory and construction work.
- `ef_construction` controls candidate exploration while building the graph.
  Larger values generally produce a better graph more slowly.
- `ef_search` controls candidate exploration at query time. Increase it for
  better recall at the cost of latency.

Search effort can be changed after construction.

```{code-cell} ipython3
index.set_ef_search(100)
print(index.query("ACGTACGTACGG", k=3)[0])
```

## Measure recall

For a manageable validation subset:

1. Compute exact distances to every reference.
2. Select the exact top `k`.
3. Query the ANN index with the same sequences.
4. Measure overlap between exact and approximate neighbor sets.
5. Tune `M`, `ef_construction`, and `ef_search` against recall, build time,
   query latency, and memory.

Do not infer recall from self-queries alone: retrieving the indexed sequence at
distance zero is necessary but does not test the ordering of nearby records.

## Practical guidance

- Preserve reference order and metadata for interpreting integer results.
- Use deterministic benchmark query sets when comparing configurations.
- Validate metric assumptions, especially equal lengths for Hamming and score
  parameters for alignment.
- Treat duplicate sequences as distinct indexed records; multiple zero-distance
  results are valid.
- Rebuild the index when the reference collection or metric configuration
  changes.

## Next steps

- Review exact comparison semantics in {doc}`distances` and {doc}`kernels`.
- Use retrieved neighbors in {doc}`models` or downstream exploratory analysis.
- Consult {doc}`../api/ann` for the constructor and query methods.
