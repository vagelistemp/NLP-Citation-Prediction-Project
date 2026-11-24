"""Text features: TF-IDF cosine, SBERT cosine and *graph-aware* embedding features.

* ``tfidf_cosine``       cosine similarity of the abstracts' TF-IDF vectors (lexical similarity);
* ``sbert_similarity``   cosine similarity of the Sentence-Transformer embeddings (semantic similarity);
* ``emb_u_vs_citers_v``  similarity between the citing paper u and the *centroid of the papers that cite v*: "does u look like
                         the papers that already cite v?";
* ``emb_refs_u_vs_v``    similarity between v and the centroid of the papers u *already cites*: "does v fit the topics u cites?".

The first two depend only on the texts. The last two are built from a graph and are therefore computed per fold on the
graph without the fold's edges (leakage-free), see ``protocol.py``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .graph import CHUNK, GraphContext


def fit_tfidf(texts: Sequence[str], max_features: int = 60_000, min_df: int = 3) -> sp.csr_matrix:
    """L2-normalised TF-IDF matrix (unigrams and bigrams, sublinear tf). Fitted on all abstracts; no labels are involved."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=min_df,
                          sublinear_tf=True, dtype=np.float32)
    return vec.fit_transform(texts).tocsr()


def sparse_row_cosine(X: sp.csr_matrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise dot product of L2-normalised sparse rows (= cosine similarity)."""
    out = np.zeros(len(u), dtype=np.float64)
    for s in range(0, len(u), CHUNK):
        out[s:s + CHUNK] = np.asarray(X[u[s:s + CHUNK]].multiply(X[v[s:s + CHUNK]]).sum(axis=1)).ravel()
    return out


def dense_row_cosine(E: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise dot product of unit-length dense rows (embeddings are normalised by ``embed.py``)."""
    out = np.zeros(len(u), dtype=np.float64)
    for s in range(0, len(u), 200_000):
        out[s:s + 200_000] = np.einsum("ij,ij->i", E[u[s:s + 200_000]], E[v[s:s + 200_000]])
    return out


def _normalise_rows(M: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, norm, out=np.zeros_like(M), where=norm > 0)


def graph_embedding_features(ctx: GraphContext, E: np.ndarray, pairs: np.ndarray) -> Dict[str, np.ndarray]:
    """Similarity of a paper to the *neighbourhood* of the other one (graph-dependent, see module docstring)."""
    u, v = pairs[:, 0], pairs[:, 1]
    citers_centroid = _normalise_rows(np.asarray(ctx.AT @ E, dtype=np.float32))     # mean embedding of papers citing x
    refs_centroid = _normalise_rows(np.asarray(ctx.A @ E, dtype=np.float32))        # mean embedding of papers x cites
    return {
        "emb_u_vs_citers_v": np.einsum("ij,ij->i", E[u], citers_centroid[v]),
        "emb_refs_u_vs_v": np.einsum("ij,ij->i", refs_centroid[u], E[v]),
    }
