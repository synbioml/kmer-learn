"""kmer.background._base — abstract base class for background models.

A "background model" generates null-model sequences from real ones:
shuffles, chunk-shuffles, or any other disruption that preserves some
composition property. Background models are used by
:class:`kmer.models.KmerMultinomialNB` and similar estimators
to generate negatives when none are provided.

All background models implement two methods:
  - ``perturb(seq)``       — perturb one sequence
  - ``perturb_many(seqs)`` — perturb many (may be parallelized)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Iterable, Union

SeqLike = Union[str, bytes]


def _validate_n_jobs(n_jobs: int) -> int:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise TypeError("n_jobs must be an integer")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    return int(n_jobs)


class BaseBackgroundModel(ABC):
    """Abstract base class for sequence background models.

    Subclasses must implement :meth:`perturb` (single sequence).
    :meth:`perturb_many` has a default Python-loop implementation that
    subclasses may override for performance (e.g. the C-backed
    :class:`KmerShuffler` and :class:`Chunker` do this).
    """

    @abstractmethod
    def perturb(self, seq: SeqLike) -> SeqLike:
        """Perturb a single sequence.

        Parameters
        ----------
        seq : str or bytes
            Input sequence. Return type matches input type.

        Returns
        -------
        str or bytes
            Perturbed sequence of the same type and (usually) length.
        """
        ...

    def perturb_many(
        self,
        seqs: Iterable[SeqLike],
        n_jobs: int = 1,
    ) -> list:
        """Perturb many sequences.

        Default implementation loops in Python. Subclasses with a native
        batch engine may override this for performance.

        Parameters
        ----------
        seqs : iterable of (str or bytes)
            Input sequences.
        n_jobs : int, default 1
            Requested worker count. The default implementation is serial;
            native implementations may use multiple workers.

        Returns
        -------
        list
            Independently perturbed sequences, one per input.
        """
        _validate_n_jobs(n_jobs)
        return [self.perturb(s) for s in seqs]
