"""Run the experiments on the real challenge data and write results/real_data_results.json (aggregates only).

    python scripts/run_real.py --data-dir DIR --embeddings embeddings.npy [--folds 5] [--out results/real_data_results.json]
                               [--submission submission.csv]

Steps (all leak-free unless stated):
  1. data checks (are the test pairs hidden edges? mutual citations? degree-0 shares?)
  2. a *pseudo-test split*: hide as many known edges as the real test set has positives, add random negatives, remove the
     hidden edges from the graph. Because the answers are known, every score below is a real held-out score.
  3. construction comparison with the 9 baseline features: leaky vs leak-free
  4. cumulative feature ablation and leave-one-group-out ablation (graph features are built once and sub-selected)
  5. final model on the full data -> predictions for the real test pairs

Only aggregate numbers are written; no abstract text and no per-paper rows are stored in the results file.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citepred.features import DEFAULT_GROUPS, FEATURE_GROUPS, ORIGINAL_9, FeatureBuilder, groups_for, names_for
from citepred.io import load_dataset, write_submission
from citepred.model import calibration_table
from citepred.pipeline import evaluate_subset, pseudo_test_split
from citepred.protocol import build_training_set


def data_checks(ds) -> dict:
    n = ds.n_papers
    E, T = ds.edges, ds.test_pairs
    es = set(map(tuple, E.tolist()))
    out_deg = np.bincount(E[:, 0], minlength=n)
    in_deg = np.bincount(E[:, 1], minlength=n)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, n, size=(len(E), 2))
    return {
        "papers": n, "edges": int(len(E)), "test_pairs": int(len(T)),
        "mutual_citations": int(sum((b, a) in es for a, b in es)),
        "test_pairs_in_edgelist": int(sum(tuple(p) in es for p in T.tolist())),
        "test_pairs_reversed_in_edgelist": int(sum((b, a) in es for a, b in T.tolist())),
        "papers_with_no_references": float((out_deg == 0).mean()),
        "edges_with_citing_id_greater_than_cited_id": float((E[:, 0] > E[:, 1]).mean()),
        "test_pairs_with_first_id_greater": float((T[:, 0] > T[:, 1]).mean()),
        "leak_table": {
            "positives_out_degree_u_ge_1_full_graph": float((out_deg[E[:, 0]] >= 1).mean()),
            "random_pairs_out_degree_u_ge_1": float((out_deg[rand[:, 0]] >= 1).mean()),
            "test_pairs_out_degree_u_ge_1": float((out_deg[T[:, 0]] >= 1).mean()),
            "mean_out_degree_u_positives_full_graph": float(out_deg[E[:, 0]].mean()),
            "mean_out_degree_u_random_pairs": float(out_deg[rand[:, 0]].mean()),
            "mean_out_degree_u_test_pairs": float(out_deg[T[:, 0]].mean()),
        },
    }


def brief(r: dict) -> dict:
    return {"protocol": r["protocol"], "n_features": r["n_features"], "cv": r["cv"], "test": r["test"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--embeddings", required=True, help="the .npy matrix written by citepred.embed (rows = paper ids)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results/real_data_results.json")
    ap.add_argument("--submission", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny subsample for a smoke test (writes to a temporary file name)")
    args = ap.parse_args()

    t_start = time.time()
    ds = load_dataset(args.data_dir)
    E = np.load(args.embeddings)
    assert E.shape[0] == ds.n_papers, "embedding rows must match the number of papers"
    results = {"data_checks": data_checks(ds), "settings": {"folds": args.folds, "seed": 0, "embedding_dim": int(E.shape[1]), "ablation_folds_scored": 2}}
    print("data checks done")
    max_pos = 20000 if args.quick else None
    fast = dict(learning_rate=0.1, n_estimators=500)                       # protocol comparison and ablations
    final = dict(learning_rate=0.04, n_estimators=3000)                    # final model
    abl_folds = [0, 1]                                                     # ablations score two folds (the headline runs score all)

    # ---------------------------------------------------------------- pseudo-test split
    dev = pseudo_test_split(ds)
    results["pseudo_test"] = {"hidden_edges": int(len(dev.test_pairs) // 2), "known_edges_left": int(len(dev.edges)),
                              "hidden_share": float(len(dev.test_pairs) // 2 / len(ds.edges))}
    builder = FeatureBuilder(dev, E)
    all_groups = builder.available(DEFAULT_GROUPS)
    all_features = names_for(all_groups)

    # ---------------------------------------------------------------- 3. protocol comparison, original 9 features
    g9 = groups_for(ORIGINAL_9)
    X9_test = builder.build(dev.edges, dev.test_pairs, g9)
    comp = {}
    for leak in (True, False):
        t0 = time.time()
        ts = build_training_set(dev, builder, g9, n_folds=args.folds, leak=leak, max_positives=max_pos, log=lambda *a: None)
        r = evaluate_subset(ts, X9_test, dev.test_labels, ORIGINAL_9, params=fast, log=lambda *a: None)
        comp["leaky" if leak else "leak-free"] = brief(r)
        print(f"original-9 {'leaky' if leak else 'leak-free'}: CV {r['cv']['logloss']:.4f}  test {r['test']['logloss']:.4f}  ({time.time() - t0:.0f}s)")
    results["protocol_comparison_original9"] = comp

    # ---------------------------------------------------------------- 4. build all features once (leak-free), ablate by columns
    t0 = time.time()
    ts_all = build_training_set(dev, builder, all_groups + ["ids"], n_folds=args.folds, leak=False, max_positives=max_pos)
    X_test = builder.build(dev.edges, dev.test_pairs, all_groups + ["ids"])      # 'ids' is only used for the artefact check below
    results["feature_build_seconds"] = time.time() - t0
    results["n_training_rows"] = int(len(ts_all.y))
    print(f"all features built: {len(ts_all.y)} rows, {len(all_features)} features ({time.time() - t0:.0f}s)")

    steps = [("original 9 features", ORIGINAL_9)]
    cumulative = list(ORIGINAL_9)
    for label, groups in [("+ all degree and neighbourhood measures", ["degree", "neighbors"]), ("+ two-hop paths, reverse edge", ["paths"]),
                          ("+ PageRank", ["pagerank"]), ("+ TF-IDF text similarity", ["tfidf"]), ("+ author citation prior", ["authors_graph"]),
                          ("+ graph-aware SBERT features", ["emb_graph"]), ("+ SVD embedding", ["svd"]), ("+ remaining author features", ["authors_static"])]:
        cumulative = cumulative + [f for g in groups for f in FEATURE_GROUPS[g] if f not in cumulative]
        steps.append((label, list(cumulative)))
    cum = []
    for label, feats in steps:
        feats = [f for f in feats if f in ts_all.X.columns]
        r = evaluate_subset(ts_all, X_test, dev.test_labels, feats, params=fast, log=lambda *a: None, folds_to_run=abl_folds)
        cum.append({"step": label, **brief(r)})
        print(f"{label}: CV {r['cv']['logloss']:.4f}  test {r['test']['logloss']:.4f}  auc {r['test']['auc']:.4f}")
    results["cumulative_ablation"] = cum

    loo = []
    for g in all_groups:
        feats = [f for f in all_features if f not in FEATURE_GROUPS[g]]
        r = evaluate_subset(ts_all, X_test, dev.test_labels, feats, params=fast, log=lambda *a: None, folds_to_run=abl_folds)
        loo.append({"removed_group": g, **brief(r)})
        print(f"without {g}: CV {r['cv']['logloss']:.4f}  test {r['test']['logloss']:.4f}")
    results["leave_one_group_out"] = loo

    # ---------------------------------------------------------------- best model on the pseudo-test: importance, calibration, errors
    r = evaluate_subset(ts_all, X_test, dev.test_labels, all_features, params=final, log=lambda *a: None)
    results["all_features_final_params"] = brief(r)
    r_ids = evaluate_subset(ts_all, X_test, dev.test_labels, all_features + FEATURE_GROUPS["ids"], params=fast, log=lambda *a: None, folds_to_run=abl_folds)
    results["id_order_artefact_check"] = {"without_ids": brief(evaluate_subset(ts_all, X_test, dev.test_labels, all_features, params=fast, log=lambda *a: None, folds_to_run=abl_folds)), "with_ids": brief(r_ids)}
    print("with id-order features: test", round(r_ids["test"]["logloss"], 4))
    results["importance"] = r["importance"].round(5).reset_index().rename(columns={"index": "feature"}).to_dict("records")
    results["calibration"] = r["calibration"].to_dict("records")
    p, y = r["test_predictions"], dev.test_labels
    ll = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    out_deg_u = X_test["out_degree_p1"].to_numpy()
    buckets = {"out-degree(u) = 0": out_deg_u == 0, "1-10": (out_deg_u >= 1) & (out_deg_u <= 10), "11-50": (out_deg_u > 10) & (out_deg_u <= 50), "> 50": out_deg_u > 50}
    results["error_analysis_by_out_degree_of_citing_paper"] = [
        {"bucket": k, "share_of_pairs": float(m.mean()), "positive_rate": float(y[m].mean()), "mean_logloss": float(ll[m].mean())} for k, m in buckets.items() if m.any()]
    print("final-params all features: CV", round(r["cv"]["logloss"], 4), "test", round(r["test"]["logloss"], 4), "auc", round(r["test"]["auc"], 4))

    # ---------------------------------------------------------------- 5. final model on the real data, real test pairs
    if args.submission:
        builder_full = FeatureBuilder(ds, E)
        ts_full = build_training_set(ds, builder_full, all_groups, n_folds=args.folds, leak=False, max_positives=max_pos)
        Xt_full = builder_full.build(ds.edges, ds.test_pairs, all_groups)
        rf = evaluate_subset(ts_full, Xt_full, None, all_features, params=final, log=lambda *a: None)
        write_submission(args.submission, rf["test_predictions"], Path(args.data_dir) / "submission_random.csv")
        results["final_model_cv"] = rf["cv"]
        print("final model CV", rf["cv"], "-> submission written")

    results["total_seconds"] = time.time() - t_start
    out = Path(args.out if not args.quick else args.out.replace(".json", "_quick.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
