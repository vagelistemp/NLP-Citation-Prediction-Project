"""Author features.

* ``common_authors`` / ``author_jaccard``: how many authors the two papers share (self-citation is common);
* ``author_prior``: *how often did the authors of u already cite the authors of v?* Over the graph's known edges x -> y it
  counts, for every author pair (a, b), how many times a paper by a cited a paper by b; the feature is the sum of that count
  over all pairs (a in authors(u), b in authors(v)). It captures research communities that cite each other. Because it is
  built from a graph, it is computed per fold on the graph without the fold's edges (see ``protocol.py``).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp

from .graph import CHUNK, GraphContext, _jaccard, _row_overlap


def author_incidence(authors: Sequence[Sequence[str]]) -> sp.csr_matrix:
    """Paper x author 0/1 matrix. Author names are matched exactly as written (no name disambiguation)."""
    index: Dict[str, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    for p, names in enumerate(authors):
        for name in sorted(set(names)):
            rows.append(p)
            cols.append(index.setdefault(name, len(index)))
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(authors), max(len(index), 1)))


def static_author_features(P: sp.csr_matrix, pairs: np.ndarray) -> Dict[str, np.ndarray]:
    u, v = pairs[:, 0], pairs[:, 1]
    n_u = np.asarray(P.sum(axis=1)).ravel()[u]
    n_v = np.asarray(P.sum(axis=1)).ravel()[v]
    common = _row_overlap(P, P, u, v)
    return {"common_authors": common, "author_jaccard": _jaccard(common, n_u, n_v), "n_authors_p1": n_u, "n_authors_p2": n_v}


def author_prior_matrix(ctx: GraphContext, P: sp.csr_matrix) -> sp.csr_matrix:
    """C[a, b] = number of known edges x -> y with a an author of x and b an author of y."""
    return (P.T @ (ctx.A @ P)).tocsr()


def author_prior(C: sp.csr_matrix, P: sp.csr_matrix, pairs: np.ndarray) -> np.ndarray:
    """Sum over a in authors(u), b in authors(v) of C[a, b], for every pair (chunked)."""
    u, v = pairs[:, 0], pairs[:, 1]
    out = np.zeros(len(u), dtype=np.float64)
    small = 5_000                                         # the intermediate (chunk x authors) matrix can be large
    for s in range(0, len(u), small):
        left = P[u[s:s + small]] @ C
        out[s:s + small] = np.asarray(left.multiply(P[v[s:s + small]]).sum(axis=1)).ravel()
    return out
