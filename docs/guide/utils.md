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

# Sequence and k-mer utilities

`kmer.utils` contains the small, deterministic operations used to represent
DNA words as integers, recover their string representation, handle reverse
complements, and iterate through a sequence.

## Learning goals

By the end of this guide, you will be able to:

- convert ACGT k-mers to compact two-bit integer codes and back;
- compute reverse complements for unambiguous and IUPAC sequences;
- canonicalize a k-mer with respect to its reverse complement;
- iterate over overlapping k-mers without constructing an encoder matrix.

## Alphabets

```{code-cell} ipython3
from kmer.utils import ALPHABET, IUPAC_ALPHABET

print("strict alphabet:", ALPHABET)
print("IUPAC alphabet:", IUPAC_ALPHABET)
```

The packed representation is defined only for `A`, `C`, `G`, and `T`.
Reverse-complement operations additionally understand uppercase IUPAC
ambiguity symbols. Lowercase letters and `U` are not normalized automatically.

## Pack and unpack k-mers

{func}`~kmer.utils.kmer_to_code` stores each base in two bits using
`A=0`, `C=1`, `G=2`, and `T=3`, with the leftmost base in the most significant
position.

```{code-cell} ipython3
from kmer.utils import code_to_kmer, kmer_to_code

for kmer in ["AAA", "ACG", "TGA", "TTT"]:
    code = kmer_to_code(kmer)
    print(f"{kmer} -> {code:2d} -> {code_to_kmer(code, len(kmer))}")
```

For a fixed length `k`, codes range from `0` to $4^k-1$. This is why encoder
columns can be indexed directly by packed k-mer codes.

## Reverse complements

```{code-cell} ipython3
from kmer.utils import reverse_complement

for sequence in ["ACGTTGCA", "ARYN"]:
    print(sequence, "->", reverse_complement(sequence))

print(reverse_complement(b"ACGT"), type(reverse_complement(b"ACGT")))
```

Input type is preserved: `str` produces `str`, and `bytes` produces `bytes`.
Invalid characters raise `ValueError` rather than being silently skipped.

## Canonical k-mers

A strand-independent representation chooses the smaller packed code from a
k-mer and its reverse complement. {func}`~kmer.utils.canonical_code` performs
that operation without converting the code back to a string.

```{code-cell} ipython3
from kmer.utils import canonical_code

kmer = "GTT"
code = kmer_to_code(kmer)
canonical = canonical_code(code, len(kmer))
print("input:", kmer)
print("canonical k-mer:", code_to_kmer(canonical, len(kmer)))
```

Use {func}`~kmer.utils.kmer_is_canonical` when you only need to test whether a
string is already the selected orientation.

```{code-cell} ipython3
from kmer.utils import kmer_is_canonical

for kmer in ["AAC", "GTT", "ACGT"]:
    print(kmer, kmer_is_canonical(kmer))
```

This convention is the basis of `canonical_rc=True` in the public encoders.
Do not canonicalize when transcriptional direction or strand assignment is
part of the signal.

## Iterate over overlapping k-mers

```{code-cell} ipython3
from kmer.utils import iterate_kmers

sequence = "ACGTAC"
print(list(iterate_kmers(sequence, k=3)))
```

Iteration yields $n-k+1$ words for a sequence of length $n$ when `k <= n`.
It is useful for inspection and lightweight algorithms; use
{class}`~kmer.encoders.SpectrumEncoder` for sparse feature matrices and larger
batches.

## Practical guidance

- Validate or normalize input before calling these strict utilities.
- Keep `k` alongside every packed code; the integer alone does not record word
  length.
- Prefer packed operations inside tight loops and decode only for reporting.
- Use canonical representations only for analyses where strand orientation is
  irrelevant.

## Next steps

- Continue to {doc}`encoders` to build sparse sequence representations.
- Use {doc}`distances` when whole-sequence substitutions or indels matter.
- Consult {doc}`../api/utils` for complete signatures.
