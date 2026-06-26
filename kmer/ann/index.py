"""HNSW ANN index for sequences (vendored hnswlib + custom metrics)."""
from __future__ import annotations
from ._native._hnsw import HNSWIndex as _HNSW

class ANNIndex:
    """HNSW approximate nearest neighbor index for sequences."""
    def __init__(self, seqs, metric="levenshtein", M=16, ef_construction=200,
                 ef_search=50, gkm_L=10, gkm_k=6, gkm_d=3, gkm_use_rc=False,
                 match=1, mismatch=-1, gap=-1):
        self._index = _HNSW(seqs, metric=metric, M=M, ef_construction=ef_construction,
                            ef_search=ef_search, gkm_L=gkm_L, gkm_k=gkm_k, gkm_d=gkm_d,
                            gkm_use_rc=gkm_use_rc, match=match, mismatch=mismatch, gap=gap)
    def query(self, query_seqs, k=10):
        return self._index.query(query_seqs, k=k)
    def set_ef_search(self, ef):
        self._index.set_ef_search(ef)
    def get_size(self):
        return self._index.get_size()
