import unittest

import numpy as np

from citepred.io import Dataset
from citepred.kg import build_knowledge_graph
from citepred.synthetic import make_synthetic


def tiny():
    edges = np.array([[0, 1], [0, 2], [3, 1]], dtype=np.int64)
    authors = [["Ann", "Bob"], ["Bob"], [], ["Ann", "Ann"]]
    return Dataset(4, edges, ["a", "b", "c", "d"], authors)


class KnowledgeGraphTest(unittest.TestCase):
    def test_counts_and_duplicates(self):
        kg = build_knowledge_graph(tiny())
        s = kg.summary()
        self.assertEqual((s["paper_nodes"], s["author_nodes"], s["cites_edges"]), (4, 2, 3))
        self.assertEqual(s["wrote_edges"], 4)                    # "Ann" listed twice on paper 3 counts once
        self.assertAlmostEqual(s["papers_without_authors"], 0.25)
        self.assertAlmostEqual(s["papers_without_references"], 0.5)
        self.assertAlmostEqual(s["mean_authors_per_paper"], 1.0)

    def test_triples_roundtrip(self):
        kg = build_knowledge_graph(tiny())
        triples = set(kg.triples())
        self.assertIn(("paper:0", "cites", "paper:1"), triples)
        self.assertIn(("author:Bob", "wrote", "paper:1"), triples)
        self.assertEqual(len(triples), 3 + 4)

    def test_matches_synthetic_dataset(self):
        ds = make_synthetic(300)
        kg = build_knowledge_graph(ds)
        self.assertEqual(len(kg.cites), len(ds.edges))
        self.assertEqual(int(kg.wrote[:, 1].max()) < ds.n_papers, True)
        self.assertEqual(len(kg.author_names), kg.n_authors)


if __name__ == "__main__":
    unittest.main()
