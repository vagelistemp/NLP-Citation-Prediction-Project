import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from citepred import authors as A
from citepred import graph as G
from citepred import text as T
from citepred.features import FEATURE_GROUPS, ORIGINAL_9, FeatureBuilder, groups_for, names_for
from citepred.io import load_dataset, write_submission
from citepred.model import calibration_table, score
from citepred.synthetic import make_synthetic


class TextTests(unittest.TestCase):
    def test_tfidf_cosine_properties(self):
        texts = ["graph neural networks for citation prediction", "graph neural networks for citation prediction",
                 "bananas and apples are fruit", "citation graph prediction with neural networks"]
        X = T.fit_tfidf(texts, min_df=1)
        u, v = np.array([0, 0, 0]), np.array([1, 2, 3])
        c = T.sparse_row_cosine(X, u, v)
        self.assertAlmostEqual(c[0], 1.0, places=5)
        self.assertAlmostEqual(c[1], 0.0, places=5)
        self.assertTrue(0.0 < c[2] < 1.0)

    def test_dense_cosine(self):
        E = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        c = T.dense_row_cosine(E, np.array([0, 0]), np.array([1, 2]))
        self.assertEqual(list(c), [0.0, 1.0])

    def test_graph_embedding_features_by_hand(self):
        # papers 0..3; 1 and 2 cite paper 3; paper 0 cites paper 1
        E = np.array([[1, 0], [0, 1], [0.6, 0.8], [0, 1]], dtype=np.float32)
        ctx = G.GraphContext.build(4, np.array([[1, 3], [2, 3], [0, 1]]))
        f = T.graph_embedding_features(ctx, E, np.array([[0, 3], [3, 0]]))
        centroid_citers_of_3 = (E[1] + E[2]) / np.linalg.norm(E[1] + E[2])
        self.assertAlmostEqual(f["emb_u_vs_citers_v"][0], float(E[0] @ centroid_citers_of_3), places=5)
        centroid_refs_of_0 = E[1]                                  # paper 0 cites only paper 1
        self.assertAlmostEqual(f["emb_refs_u_vs_v"][0], float(centroid_refs_of_0 @ E[3]), places=5)
        self.assertEqual(f["emb_u_vs_citers_v"][1], 0.0)           # paper 0 has no citers: zero, not NaN
        self.assertTrue(np.isfinite(f["emb_refs_u_vs_v"]).all())


class AuthorTests(unittest.TestCase):
    def setUp(self):
        # papers: 0 by {a,b}; 1 by {b}; 2 by {c}; 3 by {a}
        self.P = A.author_incidence([["a", "b"], ["b"], ["c"], ["a"]])

    def test_static_features(self):
        f = A.static_author_features(self.P, np.array([[0, 1], [0, 2], [0, 3]]))
        self.assertEqual(list(f["common_authors"]), [1, 0, 1])
        self.assertAlmostEqual(f["author_jaccard"][0], 1 / 2)
        self.assertEqual(list(f["n_authors_p1"]), [2, 2, 2])

    def test_author_prior_by_hand(self):
        # known edges: 0 -> 2 (authors {a,b} cite {c}), 3 -> 2 (author {a} cites {c}), 1 -> 0 ({b} cites {a,b})
        ctx = G.GraphContext.build(4, np.array([[0, 2], [3, 2], [1, 0]]))
        C = A.author_prior_matrix(ctx, self.P)
        names = {"a": 0, "b": 1, "c": 2}
        self.assertEqual(C[names["a"], names["c"]], 2)     # papers 0 and 3 by a cite a paper by c
        self.assertEqual(C[names["b"], names["c"]], 1)
        self.assertEqual(C[names["b"], names["a"]], 1)
        self.assertEqual(C[names["b"], names["b"]], 1)
        # pair (3, 2): author a of paper 3 -> author c of paper 2 : count 2
        self.assertEqual(A.author_prior(C, self.P, np.array([[3, 2]]))[0], 2)
        # pair (0, 2): a->c (2) + b->c (1) = 3
        self.assertEqual(A.author_prior(C, self.P, np.array([[0, 2]]))[0], 3)

    def test_papers_without_authors(self):
        P = A.author_incidence([[], ["x"], []])
        f = A.static_author_features(P, np.array([[0, 2], [0, 1]]))
        self.assertEqual(list(f["common_authors"]), [0, 0])
        self.assertTrue(np.isfinite(f["author_jaccard"]).all())


class FeatureRegistryTests(unittest.TestCase):
    def test_original_nine_are_all_known_features(self):
        known = set(names_for(FEATURE_GROUPS))
        self.assertTrue(set(ORIGINAL_9) <= known)
        self.assertEqual(len(ORIGINAL_9), 9)

    def test_groups_for(self):
        self.assertEqual(set(groups_for(ORIGINAL_9)), {"authors_static", "sbert", "degree", "neighbors"})

    def test_builder_without_embeddings_skips_embedding_groups(self):
        ds = make_synthetic(300, seed=1)
        b = FeatureBuilder(ds, None, svd_dim=4)
        X = b.build(ds.edges, ds.test_pairs, ["sbert", "emb_graph", "degree", "tfidf"])
        self.assertNotIn("sbert_similarity", X.columns)
        self.assertIn("out_degree_p1", X.columns)
        self.assertIn("tfidf_cosine", X.columns)
        self.assertEqual(len(X), len(ds.test_pairs))
        self.assertFalse(X.isna().any().any())


class IOTests(unittest.TestCase):
    def test_submission_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub.csv"
            write_submission(p, np.array([0.0, 1.0, 0.3]))
            df = pd.read_csv(p)
            self.assertEqual(list(df.columns), ["ID", "Label"])
            self.assertTrue(((df.Label > 0) & (df.Label < 1)).all())        # never exactly 0 or 1 for a log-loss metric
            sample = Path(tmp) / "sample.csv"
            pd.DataFrame({"ID": [10, 11, 12], "Label": [0.5] * 3}).to_csv(sample, index=False)
            write_submission(p, np.array([0.1, 0.2, 0.3]), sample)
            self.assertEqual(list(pd.read_csv(p).ID), [10, 11, 12])
            with self.assertRaises(ValueError):
                write_submission(p, np.array([0.1, 0.2]), sample)

    def test_load_dataset_roundtrip_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "abstracts.txt").write_text("0|--|first abstract\n1|--|second abstract\nbroken line\n2|--|third | with pipe\n", encoding="utf-8")
            (d / "authors.txt").write_text("0|--|A B,C D\n1|--|E F\n2|--|\n", encoding="utf-8")
            (d / "edgelist.txt").write_text("0,1\n1,2\n", encoding="utf-8")
            (d / "test.txt").write_text("2,0\n", encoding="utf-8")
            ds = load_dataset(d)
            self.assertEqual(ds.n_papers, 3)
            self.assertEqual(ds.abstracts[2], "third | with pipe")
            self.assertEqual(ds.authors, [["A B", "C D"], ["E F"], []])
            self.assertEqual(ds.edges.tolist(), [[0, 1], [1, 2]])
            self.assertEqual(ds.test_pairs.tolist(), [[2, 0]])
            self.assertTrue(any("malformed" in n for n in ds.notes))

    def test_missing_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "abstracts.txt").write_text("0|--|a\n2|--|c\n", encoding="utf-8")
            (d / "edgelist.txt").write_text("0,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(d)


class ModelUtilityTests(unittest.TestCase):
    def test_score_and_calibration(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        s = score(y, p)
        self.assertAlmostEqual(s["auc"], 1.0)
        self.assertAlmostEqual(s["accuracy"], 1.0)
        self.assertAlmostEqual(s["logloss"], -np.mean(np.log([0.9, 0.8, 0.8, 0.9])), places=6)
        self.assertTrue(np.isfinite(score(y, np.array([0.0, 0.0, 1.0, 1.0]))["logloss"]))   # clipping avoids infinities
        cal = calibration_table(y, p, bins=5)
        self.assertEqual(cal.n.sum(), 4)
        self.assertTrue(((cal.observed_rate >= 0) & (cal.observed_rate <= 1)).all())


class EmbedTests(unittest.TestCase):
    def test_chunked_cache_and_resume(self):
        try:
            from citepred.embed import embed_abstracts
            texts = ["graph neural networks", "bananas are fruit", "citation prediction", "deep learning on graphs", "apples and pears"]
            with tempfile.TemporaryDirectory() as tmp:
                msgs = []
                e1 = embed_abstracts(texts, tmp, chunk_size=2, log=msgs.append)
        except Exception as exc:
            self.skipTest(f"SBERT model not available: {exc}")
        self.assertEqual(e1.shape, (5, 384))
        self.assertTrue(np.allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-4))
        self.assertEqual(len(msgs), 3)                                    # 5 texts, chunks of 2 -> 3 chunks
        sim = e1 @ e1.T
        self.assertGreater(sim[0, 3], sim[0, 1])                          # graph-related texts are closer than a fruit text
        with tempfile.TemporaryDirectory() as tmp:
            first = embed_abstracts(texts, tmp, chunk_size=2, log=lambda *_: None)
            msgs2 = []
            second = embed_abstracts(texts, tmp, chunk_size=2, log=msgs2.append)    # everything is cached: nothing is re-encoded
            self.assertEqual(msgs2, [])
            self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
