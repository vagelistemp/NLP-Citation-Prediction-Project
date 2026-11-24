"""Sentence-Transformer (SBERT) embeddings of the abstracts, cached on disk in resumable chunks.

Encoding 138k abstracts on a CPU takes a long time, so the work is written to ``cache_dir`` in chunks of ``chunk_size``
abstracts. If the run is interrupted, calling :func:`embed_abstracts` again resumes after the last finished chunk. The
result is a float32 matrix with L2-normalised rows (so the inner product of two rows is their cosine similarity), in the
same row order as the input texts. It is stored as a plain ``.npy`` matrix and never as a pickled dictionary.

The default model is ``all-MiniLM-L6-v2``: a 6-layer Transformer encoder with 384-dimensional output. Its default
maximum input length is 256 word pieces, so abstracts longer than that are truncated (documented, not hidden).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

if sys.platform == "win32":      # PyTorch + another OpenMP runtime on Windows: see the project README
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_abstracts(texts: Sequence[str], cache_dir: str | Path, model_name: str = DEFAULT_MODEL, max_seq_length: int = 256,
                    batch_size: int = 64, chunk_size: int = 5000, limit: Optional[int] = None,
                    log: Callable[[str], None] = print) -> np.ndarray:
    """Return the (n, dim) float32 matrix of unit-length embeddings of ``texts`` (resumable, cached)."""
    from sentence_transformers import SentenceTransformer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts) if limit is None else min(limit, len(texts))
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = max_seq_length
    starts = list(range(0, n, chunk_size))
    t0 = time.time()
    done_items = 0
    for ci, start in enumerate(starts):
        path = cache_dir / f"chunk_{ci:05d}.npy"
        end = min(n, start + chunk_size)
        if path.exists() and np.load(path, mmap_mode="r").shape[0] == end - start:
            continue
        emb = model.encode(list(texts[start:end]), batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(path, emb)
        done_items += end - start
        rate = done_items / max(time.time() - t0, 1e-9)
        log(f"chunk {ci + 1}/{len(starts)} done ({end}/{n} abstracts, {rate:.1f}/s, ~{(n - end) / rate / 60:.0f} min left)")
    out = np.concatenate([np.load(cache_dir / f"chunk_{ci:05d}.npy") for ci in range(len(starts))], axis=0)
    assert out.shape[0] == n
    return out
