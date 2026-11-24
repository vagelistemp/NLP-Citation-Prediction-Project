"""Training-set construction: the leakage-free "edge-out-of-fold" protocol, and the leaky one it replaces.

**The problem.** A positive training example is a known edge (u, v). If graph features are computed on a graph that still
contains that edge, they describe the answer: u's out-degree and v's in-degree are at least 1 *because of this edge*, whereas a
random negative pair often has degree 0. A model learns "degree >= 1 means citation", which is useless on the real test set,
where the positives are edges that were *removed* from the edge list. (On the original course data: 100% of positives but only
72% of random negatives have out-degree(u) >= 1.)

**The fix.** Give every training pair a fold. For fold f, build the graph from all known edges *except the fold-f positive
edges*, and compute every graph-dependent feature of fold-f pairs (positives and negatives) on that graph. Each pair is then
featurised exactly as a hidden test pair is: on a graph that does not contain its edge.

**Mimicking the test set.** The test positives are hidden edges, about 4.7% of all edges on the real data. Each fold therefore
removes about that fraction of the edges (``hidden_fraction``), and the number of training positives is
``n_folds * hidden_fraction * n_edges``. Test pairs are featurised on the full known graph, the same graph the model "sees" in
reality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FeatureBuilder
from .io import Dataset


def pair_codes(pairs: np.ndarray, n: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * n + pairs[:, 1].astype(np.int64)


def sample_negatives(n: int, forbidden_pairs: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` distinct uniformly random pairs (u, v), u != v, avoiding ``forbidden_pairs`` in *either* direction.

    Forbidden pairs are the known edges (a pair that is already an edge is not a negative), their reverses (the real data has no
    mutual citations, so a reversed edge is not a plausible negative either) and the test pairs. Seeded through ``rng``.
    """
    forbidden = np.concatenate([pair_codes(forbidden_pairs, n), pair_codes(forbidden_pairs[:, ::-1], n)])
    forbidden.sort()
    out = np.empty((0, 2), dtype=np.int64)
    seen = np.empty(0, dtype=np.int64)
    while len(out) < count:
        need = int((count - len(out)) * 1.2) + 10
        cand = rng.integers(0, n, size=(need, 2), dtype=np.int64)
        cand = cand[cand[:, 0] != cand[:, 1]]
        codes = pair_codes(cand, n)
        pos = np.searchsorted(forbidden, codes)
        in_forbidden = (pos < len(forbidden)) & (forbidden[np.minimum(pos, len(forbidden) - 1)] == codes)
        cand, codes = cand[~in_forbidden], codes[~in_forbidden]
        _, first = np.unique(codes, return_index=True)             # distinct within this batch
        cand, codes = cand[first], codes[first]
        fresh = ~np.isin(codes, seen)                              # and distinct from earlier batches
        out = np.vstack([out, cand[fresh]])
        seen = np.concatenate([seen, codes[fresh]])
    return out[:count]


@dataclass
class TrainingSet:
    pairs: np.ndarray            # (r, 2)
    y: np.ndarray                # (r,) 0/1
    fold: np.ndarray             # (r,) fold id of every row
    X: pd.DataFrame              # features
    protocol: str                # "leak-free" or "leaky"


def default_hidden_fraction(ds: Dataset) -> float:
    """Share of all edges that is hidden in the test set (positives / (known + positives)); 0.047 on the real data."""
    if ds.test_pairs is None:
        return 0.05
    hidden = len(ds.test_pairs) / 2
    return float(hidden / (len(ds.edges) + hidden))


def make_pairs(ds: Dataset, n_folds: int, hidden_fraction: float, seed: int, max_positives: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive pairs (sampled known edges), equally many negatives, and a fold id for each row."""
    rng = np.random.default_rng(seed)
    m = len(ds.edges)
    n_pos = min(m, int(round(n_folds * hidden_fraction * m)))
    if max_positives is not None:
        n_pos = min(n_pos, max_positives)
    n_pos -= n_pos % n_folds                                      # equal folds
    pos = ds.edges[rng.choice(m, n_pos, replace=False)]
    forbidden = ds.edges if ds.test_pairs is None else np.vstack([ds.edges, ds.test_pairs])
    neg = sample_negatives(ds.n_papers, forbidden, n_pos, rng)
    pairs = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_pos, dtype=np.int8)])
    fold = np.concatenate([rng.permutation(np.arange(n_pos) % n_folds), rng.permutation(np.arange(n_pos) % n_folds)])
    order = rng.permutation(len(pairs))
    return pairs[order], y[order], fold[order]


def build_training_set(ds: Dataset, builder: FeatureBuilder, groups: Sequence[str], n_folds: int = 5,
                       hidden_fraction: Optional[float] = None, seed: int = 0, leak: bool = False,
                       max_positives: Optional[int] = None, log=print) -> TrainingSet:
    """Construct pairs, folds and features.

    ``leak=False`` (default): the leakage-free edge-out-of-fold protocol described in the module docstring.
    ``leak=True``: reproduces the original notebook, which computes every graph feature on the *full* graph.
    """
    hf = default_hidden_fraction(ds) if hidden_fraction is None else hidden_fraction
    pairs, y, fold = make_pairs(ds, n_folds, hf, seed, max_positives)
    static = builder.static(pairs, groups)
    if leak:
        graph_part = builder.graph_dependent(ds.edges, pairs, groups)
        X = pd.concat([static, graph_part], axis=1)
    else:
        n = ds.n_papers
        codes_edges = pair_codes(ds.edges, n)
        pos_mask = y == 1
        pieces: List[Tuple[np.ndarray, pd.DataFrame]] = []
        for f in range(n_folds):
            rows = np.flatnonzero(fold == f)
            hidden_codes = pair_codes(pairs[rows[pos_mask[rows]]], n)
            keep = ~np.isin(codes_edges, hidden_codes)
            log(f"  fold {f + 1}/{n_folds}: {len(rows)} pairs, graph without {int((~keep).sum())} edges ({100 * (~keep).mean():.1f}% of all edges)")
            pieces.append((rows, builder.graph_dependent(ds.edges[keep], pairs[rows], groups)))
        cols = pieces[0][1].columns
        data = np.empty((len(pairs), len(cols)), dtype=np.float64)
        for rows, part in pieces:
            data[rows] = part.to_numpy()
        X = pd.concat([static, pd.DataFrame(data, columns=cols)], axis=1)
    return TrainingSet(pairs, y, fold, X, "leaky" if leak else "leak-free")
