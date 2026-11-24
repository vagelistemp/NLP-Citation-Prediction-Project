"""End-to-end experiments: pseudo-test splits, protocol comparison, ablations and the final fit.

Because the real challenge's test labels are hidden, a *pseudo-test split* recreates the task with known answers: some known
edges are removed from the graph and become positive test pairs, an equal number of random non-edges become negative test
pairs. Everything below is then evaluated on those pairs, featurised on the graph that no longer contains them.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features import FeatureBuilder, groups_for, names_for
from .io import Dataset
from .model import CVResult, calibration_table, cross_validate, predict, score
from .protocol import build_training_set, pair_codes, sample_negatives


def pseudo_test_split(ds: Dataset, n_hidden: Optional[int] = None, seed: int = 123) -> Dataset:
    """Return a copy of ``ds`` in which ``n_hidden`` edges are removed and become labelled positive test pairs.

    ``n_hidden`` defaults to half the size of ``ds.test_pairs`` (so the hidden share of edges mirrors the real test set).
    The original test pairs are kept out of the negative sampling.
    """
    rng = np.random.default_rng(seed)
    if n_hidden is None:
        n_hidden = len(ds.test_pairs) // 2 if ds.test_pairs is not None else max(1, int(0.047 * len(ds.edges)))
    idx = rng.choice(len(ds.edges), n_hidden, replace=False)
    keep = np.ones(len(ds.edges), dtype=bool)
    keep[idx] = False
    hidden = ds.edges[idx]
    forbidden = ds.edges if ds.test_pairs is None else np.vstack([ds.edges, ds.test_pairs])
    neg = sample_negatives(ds.n_papers, forbidden, n_hidden, rng)
    pairs = np.vstack([hidden, neg])
    labels = np.concatenate([np.ones(n_hidden, dtype=np.int8), np.zeros(n_hidden, dtype=np.int8)])
    order = rng.permutation(len(pairs))
    return replace(ds, edges=ds.edges[keep], test_pairs=pairs[order], test_labels=labels[order], name=ds.name + "-pseudo-test",
                   notes=ds.notes + [f"{n_hidden} edges hidden as positive test pairs"])


def run_experiment(ds: Dataset, builder: FeatureBuilder, features: Sequence[str], n_folds: int = 5, leak: bool = False,
                   seed: int = 0, params: Optional[Dict] = None, max_positives: Optional[int] = None, log=print) -> Dict:
    """Build a training set with the chosen protocol, cross-validate, and (if labels exist) score the test pairs.

    Returns a dict with CV metrics, test metrics (``None`` when the test labels are unknown), feature importance, the test
    predictions and timing. Test pairs are always featurised on the full known graph.
    """
    t0 = time.time()
    groups = groups_for(features)
    groups = builder.available(groups)
    ts = build_training_set(ds, builder, groups, n_folds=n_folds, seed=seed, leak=leak, max_positives=max_positives, log=log)
    t_feat = time.time() - t0
    avail = [f for f in features if f in ts.X.columns]
    res: CVResult = cross_validate(ts, avail, params=params, seed=seed, log=log)
    out: Dict = {"protocol": ts.protocol, "n_features": len(avail), "features": avail, "n_train_rows": int(len(ts.y)),
                 "cv": res.overall, "cv_per_fold": res.per_fold, "importance": res.importance[["mean_gain", "share"]],
                 "feature_seconds": t_feat}
    test_pairs = ds.test_pairs
    if test_pairs is not None:
        Xt = builder.build(ds.edges, test_pairs, groups)
        p = predict(res.models, Xt, avail)
        out["test_predictions"] = p
        out["test"] = score(ds.test_labels, p) if ds.test_labels is not None else None
        if ds.test_labels is not None:
            out["calibration"] = calibration_table(ds.test_labels, p)
    out["oof"] = res.oof
    out["y"] = ts.y
    out["seconds"] = time.time() - t0
    return out


def evaluate_subset(ts, X_test: Optional[pd.DataFrame], y_test: Optional[np.ndarray], features: Sequence[str],
                    params: Optional[Dict] = None, seed: int = 0, log=print, folds_to_run=None) -> Dict:
    """Cross-validate on a column subset of an already-built training set and score the (already featurised) test pairs.

    Building the graph features is by far the slowest step, so ablations build them once for *all* features and call this for
    every subset. Returns CV metrics, test metrics (None without labels), importance, and the test predictions.
    """
    res = cross_validate(ts, features, params=params, seed=seed, log=log, folds_to_run=folds_to_run)
    out: Dict = {"protocol": ts.protocol, "n_features": len(res.features), "features": res.features, "cv": res.overall,
                 "cv_per_fold": res.per_fold, "importance": res.importance[["mean_gain", "share"]], "test": None}
    if X_test is not None:
        p = predict(res.models, X_test, res.features)
        out["test_predictions"] = p
        if y_test is not None:
            out["test"] = score(y_test, p)
            out["calibration"] = calibration_table(y_test, p)
    return out


def summarise(runs: Dict[str, Dict]) -> pd.DataFrame:
    """One row per named run: protocol, number of features, CV and test log loss / AUC."""
    rows = []
    for name, r in runs.items():
        rows.append({"run": name, "protocol": r["protocol"], "features": r["n_features"],
                     "cv_logloss": r["cv"]["logloss"], "cv_auc": r["cv"]["auc"],
                     "test_logloss": (r.get("test") or {}).get("logloss", np.nan), "test_auc": (r.get("test") or {}).get("auc", np.nan)})
    return pd.DataFrame(rows)
