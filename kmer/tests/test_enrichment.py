"""Tests for the kmer.enrichment interface stubs."""
import numpy as np
import pytest

from kmer.enrichment import (
    DrimustEnrichment,
    EnrichmentResult,
    FireEnrichment,
    FisherEnrichment,
    KmerEnrichment,
    ReduceEnrichment,
)


def _featurizer():
    from kmer.encoders import SpectrumEncoder

    return SpectrumEncoder(k=3, canonical_rc=True)


class TestBaseClass:
    def test_abstract(self):
        with pytest.raises(TypeError):
            KmerEnrichment(_featurizer())

    def test_subclasses_are_kmer_enrichment(self):
        for cls in (FireEnrichment, ReduceEnrichment, FisherEnrichment,
                    DrimustEnrichment):
            assert issubclass(cls, KmerEnrichment)

    def test_fdr_bonferroni(self):
        p = np.array([0.01, 0.02, 0.5])
        q = KmerEnrichment._correct_fdr(p, "bonferroni")
        np.testing.assert_allclose(q, [0.03, 0.06, 1.0])

    def test_fdr_bh_matches_scipy(self):
        from scipy.stats import false_discovery_control

        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 100)
        np.testing.assert_allclose(
            KmerEnrichment._correct_fdr(p, "bh"),
            false_discovery_control(p, method="bh"),
            rtol=1e-9,
        )

    def test_fdr_bh_manual(self):
        # n=8: ranked p * n/rank, then cumulative minimum from the end.
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        q = KmerEnrichment._correct_fdr(p, "bh")
        np.testing.assert_allclose(q, [0.008, 0.032, 0.0672, 0.0672, 0.0672,
                                       0.08, 0.08457, 0.205], rtol=1e-4)

    def test_fdr_bh_is_monotone_in_p(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 100)
        q = KmerEnrichment._correct_fdr(p, "bh")
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-12)
        assert np.all((q >= p - 1e-12) & (q <= 1.0))

    def test_fdr_none_passthrough(self):
        p = np.array([0.01, 0.02, 0.5])
        np.testing.assert_array_equal(KmerEnrichment._correct_fdr(p, None), p)
        np.testing.assert_array_equal(KmerEnrichment._correct_fdr(p, "none"), p)

    def test_fdr_unknown_raises(self):
        with pytest.raises(ValueError):
            KmerEnrichment._correct_fdr(np.array([0.1]), "holm")


class TestSubclassContracts:
    def test_fire_defaults(self):
        fire = FireEnrichment(_featurizer())
        assert fire.presence is True
        assert fire.y_bins is None
        assert fire.n_permutations == 1000
        assert fire.fdr == "bh"

    def test_fire_accepts_method_parameters(self):
        fire = FireEnrichment(_featurizer(), y_bins=4, n_permutations=50,
                              random_state=7)
        assert fire.y_bins == 4
        assert fire.n_permutations == 50
        assert fire.random_state == 7

    def test_reduce_defaults(self):
        reduce_ = ReduceEnrichment(_featurizer())
        assert reduce_.stepwise is True
        assert reduce_.p_in == 1e-3
        assert reduce_.max_terms is None
        assert reduce_.correlation == "pearson"

    def test_fisher_defaults_and_background_slot(self):
        fisher = FisherEnrichment(_featurizer())
        assert fisher.alternative == "two-sided"
        assert fisher.background is None

        from kmer.background import KmerShuffler

        fisher = FisherEnrichment(_featurizer(),
                                  background=KmerShuffler(k=2, seed=1))
        assert fisher.background is not None

    def test_drimust_defaults(self):
        drimust = DrimustEnrichment(_featurizer())
        assert drimust.max_top == 0.25
        assert drimust.exact is True

    def test_statistics_not_implemented(self):
        X = np.zeros((2, 2))
        for est in (FireEnrichment(_featurizer()),
                    ReduceEnrichment(_featurizer(), stepwise=False),
                    DrimustEnrichment(_featurizer())):
            with pytest.raises(NotImplementedError):
                est._score(X, np.array([0, 1]))

    def test_reduce_stepwise_not_implemented(self):
        est = ReduceEnrichment(_featurizer())
        with pytest.raises(NotImplementedError):
            est._analyze(np.zeros((2, 2)), np.array([0.1, 0.2]), ["AAA", "AAC"])

    def test_fisher_fit_not_implemented(self):
        with pytest.raises(NotImplementedError):
            FisherEnrichment(_featurizer()).fit(["ACGT"])


class TestResultAssembly:
    def test_result_dataclass_fields(self):
        r = EnrichmentResult(kmer="AAA", score=0.5, pvalue=0.01, qvalue=0.04,
                             effect=1.2, direction=1)
        assert r.extra == {}
        assert r.kmer == "AAA"

    def test_unfitted_accessors_raise(self):
        for est in (FireEnrichment(_featurizer()),
                    ReduceEnrichment(_featurizer(), stepwise=False),
                    DrimustEnrichment(_featurizer())):
            with pytest.raises(RuntimeError):
                est.top()
            with pytest.raises(RuntimeError):
                est.select()

    def test_assembly_via_minimal_subclass(self):
        class _Fixed(KmerEnrichment):
            def _score(self, X, y):
                scores = np.array([0.9, 0.5, np.nan])
                p = np.array([0.01, 0.02, 1.0])
                return scores, p, np.array([1.0, -1.0, 0.0]), np.array([1, -1, 1])

        est = _Fixed(_featurizer(), fdr="bonferroni")
        est.fit(["ACGT", "AAAACCCC"], [0, 1])
        # nan-score column is dropped; remaining sorted by score desc
        assert len(est.results_) == 2
        assert est.results_[0].score >= est.results_[1].score
        assert est.scores_[est.results_[0].kmer] == est.results_[0].score
        np.testing.assert_allclose([r.qvalue for r in est.results_],
                                   [0.02, 0.04])
