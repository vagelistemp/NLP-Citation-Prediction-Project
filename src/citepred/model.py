"""Gradient-boosted classifier, cross-validation over the protocol's folds, and metrics.

Cross-validation uses the folds the pairs were *featurised* with (see ``protocol.py``): training on folds other than f and
scoring fold f keeps every evaluated row honest. Early stopping uses a separate fold (f + 1), never the evaluated one, so the
reported score is not tuned on the data it is measured on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .protocol import TrainingSet

DEFAULT_PARAMS: Dict = dict(
    objective="binary", learning_rate=0.05, n_estimators=1500, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
)


def score(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"logloss": float(log_loss(y, p)), "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "accuracy": float(accuracy_score(y, p > 0.5))}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against observed citation rate in equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


@dataclass
class CVResult:
    features: List[str]
    oof: np.ndarray
    per_fold: List[Dict[str, float]]
    overall: Dict[str, float]
    importance: pd.DataFrame
    models: List[lgb.LGBMClassifier] = field(repr=False, default_factory=list)


def cross_validate(ts: TrainingSet, features: Sequence[str], params: Optional[Dict] = None, early_stopping: int = 100,
                   seed: int = 0, log=print, folds_to_run: Optional[Sequence[int]] = None) -> CVResult:
    params = {**DEFAULT_PARAMS, **(params or {})}
    features = [f for f in features if f in ts.X.columns]
    X, y, fold = ts.X[features], ts.y, ts.fold
    k = int(fold.max()) + 1
    run = list(range(k)) if folds_to_run is None else list(folds_to_run)       # a subset of folds makes ablations cheaper
    oof = np.zeros(len(y))
    models, per_fold = [], []
    imp = pd.DataFrame(index=features)
    for f in run:
        test_rows = fold == f
        stop_rows = fold == (f + 1) % k
        train_rows = ~(test_rows | stop_rows)
        model = lgb.LGBMClassifier(**params, random_state=seed + f)
        model.fit(X[train_rows], y[train_rows], eval_set=[(X[stop_rows], y[stop_rows])], eval_metric="logloss",
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[test_rows] = model.predict_proba(X[test_rows])[:, 1]
        per_fold.append({"fold": f, "best_iteration": int(model.best_iteration_ or params["n_estimators"]), **score(y[test_rows], oof[test_rows])})
        imp[f"fold_{f}"] = model.booster_.feature_importance(importance_type="gain")
        models.append(model)
        log(f"  fold {f + 1}/{k}: logloss {per_fold[-1]['logloss']:.4f}  auc {per_fold[-1]['auc']:.4f}  ({per_fold[-1]['best_iteration']} trees)")
    imp["mean_gain"] = imp.mean(axis=1)
    imp["share"] = imp["mean_gain"] / imp["mean_gain"].sum()
    scored = np.isin(fold, run)
    return CVResult(list(features), oof, per_fold, score(y[scored], oof[scored]), imp.sort_values("mean_gain", ascending=False), models)


def predict(models: Sequence[lgb.LGBMClassifier], X: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Average of the fold models' probabilities."""
    return np.mean([m.predict_proba(X[list(features)])[:, 1] for m in models], axis=0)
