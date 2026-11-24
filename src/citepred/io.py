"""Data loading for the citation-prediction task.

Expected files in the data directory (the format of the course's Kaggle challenge):

* ``edgelist.txt``   one known citation ``citing_id,cited_id`` per line
* ``abstracts.txt``  ``paper_id|--|abstract text`` per line
* ``authors.txt``    ``paper_id|--|author 1,author 2,...`` per line
* ``test.txt``       pairs ``paper_id_1,paper_id_2`` whose citation status must be predicted
* ``submission_random.csv``  (optional) a sample submission with the ``ID,Label`` header

The loaders are vectorised and tolerant of malformed lines (they are counted and reported, not silently dropped).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

SEP = "|--|"


@dataclass
class Dataset:
    """A citation network with text and authors. Paper ids are the integers ``0 .. n_papers - 1``."""
    n_papers: int
    edges: np.ndarray                                  # (m, 2) int64: citing, cited
    abstracts: List[str]
    authors: List[List[str]]
    test_pairs: Optional[np.ndarray] = None            # (t, 2) int64
    test_labels: Optional[np.ndarray] = None           # only known for synthetic data
    name: str = "dataset"
    notes: List[str] = field(default_factory=list)


def _read_keyed(path: Path) -> Tuple[List[int], List[str], int]:
    ids, vals, bad = [], [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            key, sep, rest = line.rstrip("\n").partition(SEP)
            if not sep or not key.strip().lstrip("-").isdigit():
                bad += 1
                continue
            ids.append(int(key))
            vals.append(rest)
    return ids, vals, bad


def load_edges(path: str | Path) -> np.ndarray:
    return pd.read_csv(path, header=None, dtype=np.int64).to_numpy()


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load a challenge-format directory. Paper ids must be ``0 .. n-1`` (the abstracts file defines n)."""
    d = Path(data_dir)
    notes: List[str] = []
    a_ids, a_texts, bad_a = _read_keyed(d / "abstracts.txt")
    n = max(a_ids) + 1
    if sorted(a_ids) != list(range(n)):
        raise ValueError("abstracts.txt must contain every paper id 0..n-1 exactly once")
    abstracts = [""] * n
    for i, t in zip(a_ids, a_texts):
        abstracts[i] = t
    authors: List[List[str]] = [[] for _ in range(n)]
    if (d / "authors.txt").exists():
        p_ids, p_vals, bad_p = _read_keyed(d / "authors.txt")
        for i, v in zip(p_ids, p_vals):
            if 0 <= i < n:
                authors[i] = [a.strip() for a in v.split(",") if a.strip()]
        if bad_p:
            notes.append(f"{bad_p} malformed lines in authors.txt")
    if bad_a:
        notes.append(f"{bad_a} malformed lines in abstracts.txt")
    edges = load_edges(d / "edgelist.txt")
    if edges.max() >= n or edges.min() < 0:
        raise ValueError("edge endpoints outside 0..n-1")
    test = load_edges(d / "test.txt") if (d / "test.txt").exists() else None
    return Dataset(n, edges, abstracts, authors, test, None, name=d.name, notes=notes)


def write_submission(path: str | Path, probabilities: np.ndarray, sample_submission: Optional[str | Path] = None) -> None:
    """Write ``ID,Label``. ``ID`` is taken from the sample submission when given (and its length must match), else 0..t-1."""
    ids = np.arange(len(probabilities))
    if sample_submission is not None and Path(sample_submission).exists():
        sample = pd.read_csv(sample_submission)
        if len(sample) != len(probabilities):
            raise ValueError(f"sample submission has {len(sample)} rows but there are {len(probabilities)} predictions")
        ids = sample["ID"].to_numpy()
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)   # never submit exact 0 or 1 to a log-loss metric
    pd.DataFrame({"ID": ids, "Label": probabilities}).to_csv(path, index=False)
