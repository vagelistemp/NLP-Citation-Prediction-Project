import unittest

import numpy as np

from citepred.features import DEFAULT_GROUPS, FeatureBuilder, names_for
from citepred.model import cross_validate, predict, score
from citepred.protocol import build_training_set, make_pairs, pair_codes, sample_negatives
from citepred.synthetic import make_synthetic


class NegativeSamplingTests(unittest.TestCase):
    def test_negatives_are_valid_distinct_and_seeded(self):
        n = 200
        rng0 = np.random.default_rng(0)
        known = rng0.integers(0, n, size=(1500, 2))
        known = known[known[:, 0] != known[:, 1]]
        neg = sample_negatives(n, known, 3000, np.random.default_rng(5))
        self.assertEqual(len(neg), 3000)
        codes = pair_codes(neg, n)
        self.assertEqual(len(np.unique(codes)), 3000)                        # distinct
        self.assertTrue((neg[:, 0] != neg[:, 1]).all())                       # no self loops
        forbidden = set(pair_codes(known, n)) | set(pair_codes(known[:, ::-1], n))
        self.assertFalse(forbidden & set(codes))                              # not an edge, not a reversed edge
        again = sample_negatives(n, known, 3000, np.random.default_rng(5))
        self.assertTrue((neg == again).all())                                 # reproducible


class PairConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = make_synthetic(1200, seed=3)

    def test_pairs_labels_and_folds(self):
        pairs, y, fold = make_pairs(self.ds, n_folds=5, hidden_fraction=0.08, seed=1)
        n = self.ds.n_papers
        known = set(pair_codes(self.ds.edges, n))
        test = set(pair_codes(self.ds.test_pairs, n))
        codes = pair_codes(pairs, n)
        self.assertEqual(y.sum(), (1 - y).sum())                              # balanced
        self.assertTrue(all(c in known for c in codes[y == 1]))               # positives are known edges
        self.assertFalse(any(c in known or c in test for c in codes[y == 0])) # negatives are neither edges nor test pairs
        self.assertEqual(sorted(np.bincount(fold[y == 1])), [y.sum() // 5] * 5)
        self.assertEqual(sorted(np.bincount(fold[y == 0])), [y.sum() // 5] * 5)

    def test_positives_count_follows_the_hidden_fraction(self):
        pairs, y, fold = make_pairs(self.ds, n_folds=4, hidden_fraction=0.05, seed=1)
        self.assertAlmostEqual(y.sum() / len(self.ds.edges), 4 * 0.05, delta=0.01)


class SpyBuilder(FeatureBuilder):
    """Records, for each call, which edges the graph was built from."""
    def graph_dependent(self, edges, pairs, groups):
        self.calls = getattr(self, "calls", [])
        self.calls.append((edges, pairs))
        return super().graph_dependent(edges, pairs, groups)


class LeakFreeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = make_synthetic(1500, seed=4)
        rng = np.random.default_rng(0)
        cls.E = rng.standard_normal((cls.ds.n_papers, 8)).astype(np.float32)
        cls.E /= np.linalg.norm(cls.E, axis=1, keepdims=True)

    def test_the_graph_of_a_fold_never_contains_that_folds_positive_edges(self):
        spy = SpyBuilder(self.ds, self.E, svd_dim=8)
        ts = build_training_set(self.ds, spy, ["degree", "neighbors"], n_folds=4, seed=2, log=lambda *a: None)
        n = self.ds.n_papers
        all_codes = set(pair_codes(self.ds.edges, n))
        self.assertEqual(len(spy.calls), 4)
        for f, (edges, pairs) in enumerate(spy.calls):
            graph_codes = set(pair_codes(edges, n))
            fold_rows = ts.fold == f
            hidden = set(pair_codes(ts.pairs[fold_rows & (ts.y == 1)], n))
            self.assertFalse(graph_codes & hidden)                                  # own edges are absent
            self.assertEqual(graph_codes, all_codes - hidden)                       # and nothing else is removed

    def test_leaky_protocol_inflates_degree_features_of_positives(self):
        b = FeatureBuilder(self.ds, self.E, svd_dim=8)
        leaky = build_training_set(self.ds, b, ["degree"], n_folds=4, seed=2, leak=True, log=lambda *a: None)
        clean = build_training_set(self.ds, b, ["degree"], n_folds=4, seed=2, leak=False, log=lambda *a: None)
        pos = clean.y == 1
        self.assertTrue((leaky.pairs == clean.pairs).all())
        self.assertTrue((leaky.X["out_degree_p1"][pos] >= clean.X["out_degree_p1"][pos]).all())
        self.assertGreater(leaky.X["out_degree_p1"][pos].mean(), clean.X["out_degree_p1"][pos].mean())
        self.assertTrue((leaky.X["out_degree_p1"][pos] >= 1).all())                # the leak: always at least the edge itself
        # negatives are unaffected by removing edges that are not theirs, apart from the (small) change of the graph
        self.assertAlmostEqual(leaky.X["out_degree_p1"][~pos].mean(), clean.X["out_degree_p1"][~pos].mean(), delta=1.0)

    def test_cross_validation_tracks_real_test_performance_only_without_leak(self):
        results = {}
        for leak in (True, False):
            b = FeatureBuilder(self.ds, self.E, svd_dim=8)
            groups = ["degree", "neighbors", "paths", "tfidf", "authors_static", "authors_graph"]
            ts = build_training_set(self.ds, b, groups, n_folds=5, seed=0, leak=leak, log=lambda *a: None)
            res = cross_validate(ts, names_for(groups), params=dict(n_estimators=200), log=lambda *a: None)
            Xt = b.build(self.ds.edges, self.ds.test_pairs, groups)                 # test pairs on the full known graph
            test = score(self.ds.test_labels, predict(res.models, Xt, res.features))
            results[leak] = (res.overall["logloss"], test["logloss"])
        leaky_cv, leaky_test = results[True]
        clean_cv, clean_test = results[False]
        self.assertGreater(leaky_test - leaky_cv, 0.2)                              # the leaky CV is wildly optimistic
        self.assertLess(abs(clean_test - clean_cv), 0.08)                           # the leak-free CV predicts the test score
        self.assertLess(clean_test, leaky_test)                                     # and the honestly trained model is better on the test set


if __name__ == "__main__":
    unittest.main()
