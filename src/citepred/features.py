"""Feature groups and the :class:`FeatureBuilder` that turns candidate pairs into a feature table.

Features fall into two kinds, and the difference is the whole point of the leakage-free protocol:

* **static** features depend only on the two papers' own content (text similarity, shared authors, id order);
* **graph-dependent** features depend on the citation graph (degrees, neighbours, paths, PageRank, SVD, graph-aware
  embeddings, author citation prior). They must be computed on a graph that does *not* contain the pair's own edge.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import authors as authors_mod
from . import graph as graph_mod
from . import text as text_mod
from .io import Dataset

FEATURE_GROUPS: Dict[str, List[str]] = {
    "degree": ["out_degree_p1", "in_degree_p1", "out_degree_p2", "in_degree_p2", "preferential_attachment"],
    "neighbors": ["common_references", "common_citers", "jaccard_successors", "jaccard_predecessors", "common_neighbors",
                  "jaccard_neighbors", "adamic_adar", "resource_alloc"],
    "paths": ["paths_u_w_v", "paths_v_w_u", "reverse_edge"],
    "pagerank": ["pagerank_p1", "pagerank_p2"],
    "svd": ["svd_score", "svd_citing_similarity"],
    "emb_graph": ["emb_u_vs_citers_v", "emb_refs_u_vs_v"],
    "authors_graph": ["author_prior"],
    "tfidf": ["tfidf_cosine"],
    "sbert": ["sbert_similarity"],
    "authors_static": ["common_authors", "author_jaccard", "n_authors_p1", "n_authors_p2"],
    "ids": ["id_p1_lt_p2", "id_gap"],
}
GRAPH_GROUPS = ("degree", "neighbors", "paths", "pagerank", "svd", "emb_graph", "authors_graph")
STATIC_GROUPS = ("tfidf", "sbert", "authors_static", "ids")

# The nine baseline features, for the like-for-like comparison of the two constructions.
ORIGINAL_9 = ["common_authors", "sbert_similarity", "out_degree_p1", "in_degree_p2", "preferential_attachment",
              "jaccard_successors", "jaccard_predecessors", "adamic_adar", "resource_alloc"]

# Default feature set: everything except the dataset-artefact id features.
DEFAULT_GROUPS = ["degree", "neighbors", "paths", "pagerank", "svd", "emb_graph", "authors_graph", "tfidf", "sbert", "authors_static"]


def groups_for(features: Iterable[str]) -> List[str]:
    wanted = set(features)
    return [g for g, names in FEATURE_GROUPS.items() if wanted & set(names)]


def names_for(groups: Iterable[str]) -> List[str]:
    return [n for g in groups for n in FEATURE_GROUPS[g]]


class FeatureBuilder:
    """Compute feature tables for candidate pairs.

    ``embeddings``: optional (n_papers, d) unit-length matrix (SBERT). Without it the ``sbert`` and ``emb_graph`` groups are
    skipped. ``svd_dim``: rank of the SVD embedding. The TF-IDF matrix and the author incidence matrix are built once.
    """

    def __init__(self, ds: Dataset, embeddings: Optional[np.ndarray] = None, svd_dim: int = 32, tfidf_features: int = 60_000):
        self.ds = ds
        self.E = embeddings
        self.svd_dim = svd_dim
        self.tfidf_features = tfidf_features
        self._tfidf = None
        self._P = authors_mod.author_incidence(ds.authors)

    @property
    def tfidf(self):
        if self._tfidf is None:
            self._tfidf = text_mod.fit_tfidf(self.ds.abstracts, self.tfidf_features)
        return self._tfidf

    def available(self, groups: Sequence[str]) -> List[str]:
        return [g for g in groups if not (g in ("sbert", "emb_graph") and self.E is None)]

    # ------------------------------------------------------------------ static
    def static(self, pairs: np.ndarray, groups: Sequence[str]) -> pd.DataFrame:
        u, v = pairs[:, 0], pairs[:, 1]
        cols: Dict[str, np.ndarray] = {}
        if "tfidf" in groups:
            cols["tfidf_cosine"] = text_mod.sparse_row_cosine(self.tfidf, u, v)
        if "sbert" in groups and self.E is not None:
            cols["sbert_similarity"] = text_mod.dense_row_cosine(self.E, u, v)
        if "authors_static" in groups:
            cols.update(authors_mod.static_author_features(self._P, pairs))
        if "ids" in groups:
            cols["id_p1_lt_p2"] = (u < v).astype(np.float64)
            cols["id_gap"] = (u - v).astype(np.float64)
        return pd.DataFrame(cols)

    # ------------------------------------------------------------------ graph-dependent
    def graph_dependent(self, edges: np.ndarray, pairs: np.ndarray, groups: Sequence[str]) -> pd.DataFrame:
        """Features of ``pairs`` computed on the graph made of ``edges``. The caller decides which edges the graph contains."""
        wanted = [g for g in groups if g in GRAPH_GROUPS]
        if not wanted:
            return pd.DataFrame(index=range(len(pairs)))
        ctx = graph_mod.GraphContext.build(self.ds.n_papers, edges)
        u, v = pairs[:, 0], pairs[:, 1]
        cols: Dict[str, np.ndarray] = {}
        if {"degree", "neighbors", "paths"} & set(wanted):
            cols.update(graph_mod.basic_features(ctx, pairs))
        if "pagerank" in wanted:
            pr = graph_mod.pagerank(ctx.A)
            cols["pagerank_p1"], cols["pagerank_p2"] = pr[u], pr[v]
        if "svd" in wanted:
            Us, Vs = graph_mod.svd_embedding(ctx.A, self.svd_dim)
            cols["svd_score"] = np.einsum("ij,ij->i", Us[u], Vs[v])
            nu, nv = np.linalg.norm(Us[u], axis=1), np.linalg.norm(Us[v], axis=1)
            cols["svd_citing_similarity"] = np.divide(np.einsum("ij,ij->i", Us[u], Us[v]), nu * nv, out=np.zeros(len(u)), where=(nu * nv) > 0)
        if "emb_graph" in wanted and self.E is not None:
            cols.update(text_mod.graph_embedding_features(ctx, self.E, pairs))
        if "authors_graph" in wanted:
            C = authors_mod.author_prior_matrix(ctx, self._P)
            cols["author_prior"] = authors_mod.author_prior(C, self._P, pairs)
        return pd.DataFrame(cols)

    def build(self, edges: np.ndarray, pairs: np.ndarray, groups: Sequence[str]) -> pd.DataFrame:
        groups = self.available(groups)
        return pd.concat([self.static(pairs, groups), self.graph_dependent(edges, pairs, groups)], axis=1)
