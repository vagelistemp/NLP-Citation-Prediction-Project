import unittest

import networkx as nx
import numpy as np

from citepred import graph as G


def random_graph(n=70, m=380, seed=0, reciprocal=True):
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < m:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and (reciprocal or (b, a) not in edges):
            edges.add((a, b))
    return n, np.array(sorted(edges), dtype=np.int64)


def random_pairs(n, k=300, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.integers(0, n, size=(k, 2))
    return p[p[:, 0] != p[:, 1]]


class GraphFeatureTests(unittest.TestCase):
    def setUp(self):
        self.n, self.edges = random_graph()
        self.pairs = random_pairs(self.n)
        self.ctx = G.GraphContext.build(self.n, self.edges)
        self.f = G.basic_features(self.ctx, self.pairs)
        out, inn = {}, {}
        for a, b in self.edges:
            out.setdefault(int(a), set()).add(int(b))
            inn.setdefault(int(b), set()).add(int(a))
        self.out, self.inn = out, inn
        self.und = {x: out.get(x, set()) | inn.get(x, set()) for x in range(self.n)}

    def test_duplicate_edges_collapse(self):
        A = G.adjacency(4, np.array([[0, 1], [0, 1], [1, 2]]))
        self.assertEqual(A.sum(), 2.0)
        self.assertEqual(G.adjacency(3, np.empty((0, 2), dtype=np.int64)).nnz, 0)

    def test_degrees(self):
        for i, (u, v) in enumerate(self.pairs):
            self.assertEqual(self.f["out_degree_p1"][i], len(self.out.get(u, ())))
            self.assertEqual(self.f["in_degree_p2"][i], len(self.inn.get(v, ())))
            self.assertEqual(self.f["preferential_attachment"][i], len(self.out.get(u, ())) * len(self.inn.get(v, ())))

    def test_overlaps_match_set_arithmetic(self):
        for i, (u, v) in enumerate(self.pairs):
            coupling = len(self.out.get(u, set()) & self.out.get(v, set()))
            cocite = len(self.inn.get(u, set()) & self.inn.get(v, set()))
            cn = len(self.und[u] & self.und[v])
            self.assertEqual(self.f["common_references"][i], coupling)
            self.assertEqual(self.f["common_citers"][i], cocite)
            self.assertEqual(self.f["common_neighbors"][i], cn)
            union = len(self.out.get(u, set()) | self.out.get(v, set()))
            self.assertAlmostEqual(self.f["jaccard_successors"][i], coupling / union if union else 0.0)
            union_n = len(self.und[u] | self.und[v])
            self.assertAlmostEqual(self.f["jaccard_neighbors"][i], cn / union_n if union_n else 0.0)

    def test_adamic_adar_and_resource_allocation_match_networkx(self):
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        g.add_edges_from(map(tuple, self.edges))
        pairs = [tuple(p) for p in self.pairs]
        aa = {(u, v): s for u, v, s in nx.adamic_adar_index(g, pairs)}
        ra = {(u, v): s for u, v, s in nx.resource_allocation_index(g, pairs)}
        for i, p in enumerate(pairs):
            self.assertAlmostEqual(self.f["adamic_adar"][i], aa[p], places=9)
            self.assertAlmostEqual(self.f["resource_alloc"][i], ra[p], places=9)

    def test_two_hop_paths_and_reverse_edge(self):
        for i, (u, v) in enumerate(self.pairs):
            self.assertEqual(self.f["paths_u_w_v"][i], len(self.out.get(u, set()) & self.inn.get(v, set())))
            self.assertEqual(self.f["paths_v_w_u"][i], len(self.out.get(v, set()) & self.inn.get(u, set())))
            self.assertEqual(self.f["reverse_edge"][i], float(u in self.out.get(v, set())))

    def test_degree_one_nodes_get_zero_adamic_adar_weight(self):
        # a path 0 - 1 - 2: node 1 (degree 2) is the shared neighbour of 0 and 2; node 0 has degree 1 and must not blow up
        ctx = G.GraphContext.build(3, np.array([[0, 1], [1, 2]]))
        self.assertEqual(ctx.w_aa[0], 0.0)
        f = G.basic_features(ctx, np.array([[0, 2]]))
        self.assertAlmostEqual(f["adamic_adar"][0], 1.0 / np.log(2))
        self.assertTrue(np.isfinite(f["adamic_adar"]).all())

    def test_pagerank_matches_networkx(self):
        A = G.adjacency(self.n, self.edges)
        ours = G.pagerank(A, iters=200)
        d = nx.DiGraph()
        d.add_nodes_from(range(self.n))
        d.add_edges_from(map(tuple, self.edges))
        ref = nx.pagerank(d, alpha=0.85, tol=1e-12, max_iter=500)
        self.assertAlmostEqual(ours.sum(), 1.0, places=8)
        for i in range(self.n):
            self.assertAlmostEqual(ours[i], ref[i], places=6)

    def test_svd_embedding_approximates_the_adjacency(self):
        A = G.adjacency(self.n, self.edges)
        errs = []
        for k in (2, 10, 30):
            U, V = G.svd_embedding(A, k)
            errs.append(np.linalg.norm(A.toarray() - U @ V.T))
        self.assertLess(errs[2], errs[1])
        self.assertLess(errs[1], errs[0])


if __name__ == "__main__":
    unittest.main()
