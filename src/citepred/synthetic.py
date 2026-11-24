"""A small synthetic citation network (text + authors + edges) so that everything runs without the real data.

The generator makes the same kinds of signal the real task has, so that every feature has something to find:

* each paper belongs to one of several *topics*; its abstract is drawn from that topic's vocabulary plus common words;
* a quarter of the papers cite nothing (as on the real data) and the number of references is heavy-tailed;
* papers cite other papers mostly *within their topic* (text similarity helps) and prefer already well-cited papers
  (preferential attachment, so degree features help);
* authors come from per-topic pools, and a paper often cites a paper written by one of its own authors (author features);
* citations only go to earlier papers, so there are no mutual citations (as in the real data); paper ids are shuffled,
  so the id order carries no information.

Test pairs are *hidden edges* (removed from the edge list) plus uniformly random non-edges, in equal numbers; their labels are
returned, so a model can be scored exactly. This mirrors how the real challenge's test set is built (verified on the real
data: no test pair appears in the edge list).
"""
from __future__ import annotations

import numpy as np

from .io import Dataset

_COMMON = ("we propose method results show approach based using paper analysis new data model system problem performance "
           "algorithm study experimental").split()


def make_synthetic(n_papers: int = 2500, n_topics: int = 8, refs_per_paper: float = 10.0, hidden_fraction: float = 0.06,
                   no_reference_fraction: float = 0.25, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    topic = rng.integers(0, n_topics, n_papers)
    vocab = [[f"t{t}w{j}" for j in range(60)] for t in range(n_topics)]
    abstracts = []
    for i in range(n_papers):
        length = int(rng.integers(40, 90))
        own = rng.choice(vocab[topic[i]], size=int(length * 0.6))
        other = rng.choice(vocab[int(rng.integers(0, n_topics))], size=int(length * 0.1))
        common = rng.choice(_COMMON, size=length - len(own) - len(other))
        words = np.concatenate([own, other, common])
        rng.shuffle(words)
        abstracts.append(" ".join(words))
    pools = [[f"Author T{t}-{k}" for k in range(40)] for t in range(n_topics)]
    authors = [list(rng.choice(pools[topic[i]], size=int(rng.integers(1, 5)), replace=False)) for i in range(n_papers)]

    # time order = a random permutation; a paper cites only earlier papers
    order = rng.permutation(n_papers)
    rank = np.empty(n_papers, dtype=np.int64)
    rank[order] = np.arange(n_papers)
    indeg = np.ones(n_papers)
    edges = set()
    author_sets = [set(a) for a in authors]
    for pos, i in enumerate(order):
        i = int(i)
        if pos == 0:
            continue
        earlier = order[:pos]
        if rng.random() < no_reference_fraction:                                    # like the real data: ~28% of papers cite nothing
            continue
        k = min(pos, max(1, int(rng.lognormal(np.log(refs_per_paper) - 0.4, 0.9))))   # heavy-tailed number of references
        same_topic = topic[earlier] == topic[i]
        probs = np.where(same_topic, indeg[earlier], indeg[earlier] * 0.04)          # topical + preferential attachment
        shared_author = np.array([bool(author_sets[i] & author_sets[int(j)]) for j in earlier])
        probs = np.where(shared_author, probs * 6.0, probs)                          # self-citation
        chosen = rng.choice(earlier, size=k, replace=False, p=probs / probs.sum())
        for j in chosen:
            edges.add((i, int(j)))
            indeg[int(j)] += 1
    edges_arr = np.array(sorted(edges), dtype=np.int64)
    # hide some edges as positive test pairs, add the same number of random non-edges
    n_hidden = int(len(edges_arr) * hidden_fraction)
    hidden_idx = rng.choice(len(edges_arr), n_hidden, replace=False)
    hidden = edges_arr[hidden_idx]
    keep = np.ones(len(edges_arr), dtype=bool)
    keep[hidden_idx] = False
    known = edges_arr[keep]
    forbidden = {(int(a), int(b)) for a, b in edges_arr} | {(int(b), int(a)) for a, b in edges_arr}
    neg = set()
    while len(neg) < n_hidden:
        a, b = int(rng.integers(0, n_papers)), int(rng.integers(0, n_papers))
        if a != b and (a, b) not in forbidden and (a, b) not in neg:
            neg.add((a, b))
    neg_arr = np.array(sorted(neg), dtype=np.int64)
    pairs = np.vstack([hidden, neg_arr])
    labels = np.concatenate([np.ones(n_hidden, dtype=int), np.zeros(n_hidden, dtype=int)])
    perm = rng.permutation(len(pairs))
    return Dataset(n_papers, known, abstracts, authors, pairs[perm], labels[perm], name=f"synthetic-{n_papers}")
