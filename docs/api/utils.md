# Utilities API

```{eval-rst}
.. currentmodule:: kmer.utils

.. py:data:: ALPHABET
   :type: str
   :value: "ACGT"

   Unambiguous uppercase DNA alphabet used by strict k-mer operations.

.. py:data:: IUPAC_ALPHABET
   :type: str
   :value: "ACGTRYSWKMBDHVN"

   Uppercase IUPAC nucleotide alphabet accepted by sequence utilities.

.. autofunction:: kmer_to_code

.. autofunction:: code_to_kmer

.. autofunction:: reverse_complement

.. autofunction:: canonical_code

.. autofunction:: iterate_kmers

.. autofunction:: kmer_is_canonical
```
