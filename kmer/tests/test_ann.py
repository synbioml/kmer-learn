"""Tests for the HNSW ANN index."""
import pytest, random, numpy as np
from kmer.ann import ANNIndex

def rseq(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))

def lev_py(a, b):
    if len(a) < len(b): a, b = b, a
    if len(b) == 0: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1): curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = curr
    return prev[-1]

def ham_py(a, b):
    if len(a)!=len(b): return max(len(a),len(b))
    return sum(1 for x,y in zip(a,b) if x!=y)

def brute_knn(q, seqs, k, f):
    d = [f(q,s) for s in seqs]; o = np.argsort(d)[:k]
    return o.tolist(), [d[i] for i in o]

class TestBasic:
    def test_lev(self):
        idx = ANNIndex(["ACGTACGT","TTTTAAAA","ACGTACGA"], metric="levenshtein")
        r = idx.query("ACGTACGT", k=2); assert 0 in r[0][0] and r[0][1][r[0][0].index(0)]==0.0
    def test_ham(self):
        idx = ANNIndex(["ACGTACGT","ACGTACGA","TTTTAAAA"], metric="hamming")
        r = idx.query("ACGTACGT", k=2); assert 0 in r[0][0]
    def test_nw(self):
        idx = ANNIndex(["ACGTACGT","TTTTAAAA","ACGTACGA"], metric="nw")
        r = idx.query("ACGTACGT", k=2); assert 0 in r[0][0]
    def test_sw(self):
        idx = ANNIndex(["ACGTACGT","TTTTAAAA","ACGTACGA"], metric="sw")
        r = idx.query("ACGTACGT", k=2); assert 0 in r[0][0]

class TestRecall:
    @pytest.mark.parametrize("m,f", [("levenshtein",lev_py),("hamming",ham_py)])
    def test_100(self, m, f):
        seqs = [rseq(20, i) for i in range(100)]
        idx = ANNIndex(seqs, metric=m, M=16, ef_construction=200, ef_search=200)
        tot = sum(len(set(idx.query(seqs[i*5], k=5)[0][0]) & set(brute_knn(seqs[i*5],seqs,5,f)[0]))/5 for i in range(20))
        assert tot/20 >= 0.7, f"Recall {tot/20:.2f}"
    @pytest.mark.parametrize("m,f", [("levenshtein",lev_py),("hamming",ham_py)])
    def test_1000(self, m, f):
        seqs = [rseq(20, i+1000) for i in range(1000)]
        idx = ANNIndex(seqs, metric=m, M=16, ef_construction=200, ef_search=200)
        tot = sum(len(set(idx.query(seqs[i*100], k=10)[0][0]) & set(brute_knn(seqs[i*100],seqs,10,f)[0]))/10 for i in range(10))
        assert tot/10 >= 0.35, f"Recall {tot/10:.2f}"

class TestLargeScale:
    def test_1000(self):
        seqs = [rseq(30, i) for i in range(1000)]
        idx = ANNIndex(seqs, metric="levenshtein", M=16, ef_construction=200, ef_search=50)
        assert idx.get_size()==1000
        for i in range(10):
            r = idx.query(rseq(30, 9999+i), k=5)
            assert len(r[0][0])==5 and all(d>=0 for d in r[0][1])
    def test_5000(self):
        seqs = [rseq(20, i+5000) for i in range(5000)]
        idx = ANNIndex(seqs, metric="levenshtein", M=16, ef_construction=200, ef_search=50)
        assert idx.get_size()==5000
        r = idx.query(seqs[0], k=10)
        assert len(r[0][0])==10 and 0 in r[0][0]

class TestDuplicates:
    def test_dups(self):
        idx = ANNIndex(["ACGTACGTACGT","ACGTACGTACGT","TTTTAAAACCCC"], metric="levenshtein")
        r = idx.query("ACGTACGTACGT", k=3)
        assert sum(1 for d in r[0][1] if d==0.0) >= 2
    def test_all_same(self):
        idx = ANNIndex(["ACGTACGT"]*10, metric="levenshtein")
        r = idx.query("ACGTACGT", k=5)
        assert all(d==0.0 for d in r[0][1])
    def test_many_dup(self):
        base = [rseq(20, i) for i in range(50)]
        seqs = [s for s in base for _ in range(20)]
        idx = ANNIndex(seqs, metric="levenshtein", M=16, ef_construction=200, ef_search=50)
        r = idx.query(base[0], k=10)
        assert sum(1 for d in r[0][1] if d==0.0) >= 5

class TestEdgeCases:
    def test_single(self):
        idx = ANNIndex(["ACGTACGT"], metric="levenshtein")
        r = idx.query("ACGTACGT", k=1); assert r[0][0]==[0]
    def test_batch(self):
        idx = ANNIndex(["ACGTACGT","TTTTAAAA","GGGGCCCC"], metric="levenshtein")
        r = idx.query(["ACGTACGT","TTTTAAAA"], k=2)
        assert len(r)==2 and r[0][0][0]==0 and r[1][0][0]==1
    def test_k_big(self):
        idx = ANNIndex(["ACGT","TTTT"], metric="levenshtein")
        r = idx.query("ACGT", k=10); assert len(r[0][0])<=2
