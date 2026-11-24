"""Graph features for candidate citation pairs, computed with sparse matrices (no per-pair Python loops).

Notation: A is the directed adjacency matrix (A[u, v] = 1 if paper u cites paper v). For a candidate pair (u, v) the question
is "does u cite v?". Everything here is computed on *the graph it is given*; the leakage-free protocol (see ``protocol.py``)
is responsible for handing in a graph that does not contain the pair's own edge.

All neighbourhood quantities are verified against networkx / set arithmetic in ``tests/test_graph.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

CHUNK = 20_000


def adjacency(n: int, edges: np.ndarray) -> sp.csr_matrix:
    """Directed 0/1 adjacency matrix (duplicate edges collapse to 1)."""
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    A = sp.csr_matrix((np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A.data[:] = 1.0
    return A


def _row_overlap(M1: sp.csr_matrix, M2: sp.csr_matrix, u: np.ndarray, v: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """For every pair i: sum over columns x of M1[u_i, x] * M2[v_i, x] * weights[x] (chunked)."""
    out = np.zeros(len(u), dtype=np.float64)
    for s in range(0, len(u), CHUNK):
        prod = M1[u[s:s + CHUNK]].multiply(M2[v[s:s + CHUNK]]).tocsr()
        out[s:s + CHUNK] = (np.asarray(prod.sum(axis=1)).ravel() if weights is None
                            else prod.astype(np.float64) @ np.asarray(weights, dtype=np.float64))
    return out


def _jaccard(inter: np.ndarray, size_u: np.ndarray, size_v: np.ndarray) -> np.ndarray:
    union = size_u + size_v - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)


def pagerank(A: sp.csr_matrix, damping: float = 0.85, iters: int = 60) -> np.ndarray:
    """PageRank by power iteration (dangling nodes spread their mass uniformly)."""
    A = A.astype(np.float64)
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out, dtype=np.float64), where=out > 0)
    P = sp.diags(inv) @ A                                  # row-stochastic for non-dangling rows
    r = np.full(n, 1.0 / n)
    dangling = out == 0
    for _ in range(iters):
        r_new = damping * (P.T @ r + r[dangling].sum() / n) + (1 - damping) / n
        if np.abs(r_new - r).sum() < 1e-10:
            r = r_new
            break
        r = r_new
    return r


def svd_embedding(A: sp.csr_matrix, k: int = 32, seed: int = 0):
    """Rank-k truncated SVD of A: U sqrt(S) describes how a paper cites, V sqrt(S) how it is cited. Score(u, v) = U[u] . V[v]."""
    k = min(k, min(A.shape) - 2)
    v0 = np.random.default_rng(seed).standard_normal(min(A.shape))
    U, s, Vt = svds(A.astype(np.float64), k=k, v0=v0)
    root = np.sqrt(s)
    return U * root, Vt.T * root


@dataclass
class GraphContext:
    """Everything derived from one graph, computed once and reused for all candidate pairs."""
    n: int
    A: sp.csr_matrix
    AT: sp.csr_matrix
    B: sp.csr_matrix              # undirected 0/1 adjacency
    out_deg: np.ndarray
    in_deg: np.ndarray
    deg: np.ndarray               # undirected degree
    w_aa: np.ndarray
    w_ra: np.ndarray

    @classmethod
    def build(cls, n: int, edges: np.ndarray) -> "GraphContext":
        A = adjacency(n, edges)
        AT = A.T.tocsr()
        B = (A + AT).tocsr()
        B.data[:] = 1.0
        out_deg = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
        in_deg = np.asarray(A.sum(axis=0), dtype=np.float64).ravel()
        deg = np.asarray(B.sum(axis=1), dtype=np.float64).ravel()
        with np.errstate(divide="ignore"):
            w_aa = np.where(deg >= 2, 1.0 / np.log(np.maximum(deg, 2)), 0.0)       # degree < 2 can never be a *shared* neighbour
            w_ra = np.where(deg >= 1, 1.0 / np.maximum(deg, 1), 0.0)
        return cls(n, A, AT, B, out_deg, in_deg, deg, w_aa, w_ra)


def basic_features(ctx: GraphContext, pairs: np.ndarray) -> Dict[str, np.ndarray]:
    """Degrees, neighbourhood overlaps and short paths for each (u, v) in ``pairs`` (u is the citing paper)."""
    u, v = pairs[:, 0], pairs[:, 1]
    out_u, in_v = ctx.out_deg[u], ctx.in_deg[v]
    f: Dict[str, np.ndarray] = {
        "out_degree_p1": out_u,
        "in_degree_p2": in_v,
        "in_degree_p1": ctx.in_deg[u],
        "out_degree_p2": ctx.out_deg[v],
        "preferential_attachment": out_u * in_v,
    }
    # papers cited by both (bibliographic coupling) and papers citing both (co-citation)
    coupling = _row_overlap(ctx.A, ctx.A, u, v)
    cocitation = _row_overlap(ctx.AT, ctx.AT, u, v)
    f["common_references"] = coupling
    f["common_citers"] = cocitation
    f["jaccard_successors"] = _jaccard(coupling, ctx.out_deg[u], ctx.out_deg[v])
    f["jaccard_predecessors"] = _jaccard(cocitation, ctx.in_deg[u], ctx.in_deg[v])
    # undirected neighbourhood measures
    cn = _row_overlap(ctx.B, ctx.B, u, v)
    f["common_neighbors"] = cn
    f["jaccard_neighbors"] = _jaccard(cn, ctx.deg[u], ctx.deg[v])
    f["adamic_adar"] = _row_overlap(ctx.B, ctx.B, u, v, ctx.w_aa)
    f["resource_alloc"] = _row_overlap(ctx.B, ctx.B, u, v, ctx.w_ra)
    # directed two-hop path u -> w -> v, and the reverse direction v -> w -> u
    f["paths_u_w_v"] = _row_overlap(ctx.A, ctx.AT, u, v)
    f["paths_v_w_u"] = _row_overlap(ctx.AT, ctx.A, u, v)
    f["reverse_edge"] = np.asarray(ctx.A[v, u]).ravel().astype(np.float64)
    return f
