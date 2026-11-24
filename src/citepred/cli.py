"""Command line:  python -m citepred <command>

    demo                         run the whole pipeline on a synthetic network (needs no data, no downloads)
    run --data-dir DIR ...       run on challenge-format data; writes a submission file for the test pairs

``run`` expects ``edgelist.txt``, ``abstracts.txt``, ``authors.txt`` and ``test.txt`` in DIR (see ``citepred.io``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import DEFAULT_GROUPS, FeatureBuilder, names_for
from .io import load_dataset, write_submission
from .pipeline import run_experiment, summarise
from .synthetic import make_synthetic


def _embeddings(args, ds):
    if args.no_sbert:
        return None
    cache = Path(args.cache_dir) / "sbert"
    from .embed import embed_abstracts
    return embed_abstracts(ds.abstracts, cache, max_seq_length=args.max_seq_length)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="citepred", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="synthetic end-to-end demo")
    d.add_argument("--papers", type=int, default=2500)
    d.add_argument("--folds", type=int, default=5)
    r = sub.add_parser("run", help="run on challenge-format data")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--out", default="submission.csv")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--max-seq-length", type=int, default=128)
    r.add_argument("--no-sbert", action="store_true", help="skip the (slow) SBERT embeddings")
    r.add_argument("--summary", default=None, help="write the CV summary as JSON to this path")
    args = ap.parse_args(argv)

    if args.command == "demo":
        ds = make_synthetic(args.papers)
        builder = FeatureBuilder(ds, None, svd_dim=16)                     # the synthetic abstracts are bags of topic words: no SBERT needed
        feats = names_for(builder.available(DEFAULT_GROUPS))
        runs = {"full-graph features (baseline)": run_experiment(ds, builder, feats, args.folds, leak=True, params=dict(n_estimators=300), log=lambda *a: None),
                "edge-out-of-fold (this fix)": run_experiment(ds, builder, feats, args.folds, leak=False, params=dict(n_estimators=300), log=lambda *a: None)}
        print(summarise(runs).round(4).to_string(index=False))
        return 0

    ds = load_dataset(args.data_dir)
    for note in ds.notes:
        print("note:", note)
    E = _embeddings(args, ds)
    builder = FeatureBuilder(ds, E)
    feats = names_for(builder.available(DEFAULT_GROUPS))
    res = run_experiment(ds, builder, feats, args.folds, leak=False)
    print({k: round(v, 4) for k, v in res["cv"].items()})
    write_submission(args.out, res["test_predictions"], Path(args.data_dir) / "submission_random.csv")
    print("submission written to", args.out)
    if args.summary:
        Path(args.summary).write_text(json.dumps({"cv": res["cv"], "features": res["features"]}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
