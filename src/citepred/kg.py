"""The scholarly knowledge graph behind the task.

The data is a small knowledge graph with two node types and three relations:

    (Paper)  --cites-->      (Paper)        known citations
    (Author) --wrote-->      (Paper)        authorship
    (Paper)  --has_abstract-> (Text)        the abstract, used for text similarity

Every feature of the pipeline is a question asked of this graph: how many references does a paper have (``cites``), do two
papers share an author (``wrote``), how similar are two abstracts (``has_abstract``), have the authors of u cited the authors
of v before (``wrote`` + ``cites``). This module makes the graph explicit: typed edge arrays, summary statistics and a triple
export for use with other graph tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np

from .io import Dataset


@dataclass
class KnowledgeGraph:
    n_papers: int
    n_authors: int
    cites: np.ndarray                 # (m, 2) paper -> paper
    wrote: np.ndarray                 # (k, 2) author -> paper
    author_names: List[str]
    abstracts: List[str]

    def summary(self) -> Dict[str, float]:
        """Node and edge counts and the most useful degree statistics."""
        papers_per_author = np.bincount(self.wrote[:, 0], minlength=self.n_authors)
        authors_per_paper = np.bincount(self.wrote[:, 1], minlength=self.n_papers)
        out_deg = np.bincount(self.cites[:, 0], minlength=self.n_papers)
        in_deg = np.bincount(self.cites[:, 1], minlength=self.n_papers)
        return {
            "paper_nodes": int(self.n_papers),
            "author_nodes": int(self.n_authors),
            "cites_edges": int(len(self.cites)),
            "wrote_edges": int(len(self.wrote)),
            "mean_references_per_paper": float(out_deg.mean()),
            "mean_citations_per_paper": float(in_deg.mean()),
            "papers_without_references": float((out_deg == 0).mean()),
            "papers_without_authors": float((authors_per_paper == 0).mean()),
            "mean_authors_per_paper": float(authors_per_paper.mean()),
            "mean_papers_per_author": float(papers_per_author.mean()) if self.n_authors else 0.0,
            "authors_with_several_papers": float((papers_per_author > 1).mean()) if self.n_authors else 0.0,
        }

    def triples(self, limit: int | None = None) -> Iterator[Tuple[str, str, str]]:
        """(subject, relation, object) triples, e.g. ``("paper:3", "cites", "paper:17")``."""
        for i, (u, v) in enumerate(self.cites):
            if limit is not None and i >= limit:
                break
            yield f"paper:{u}", "cites", f"paper:{v}"
        for i, (a, p) in enumerate(self.wrote):
            if limit is not None and i >= limit:
                break
            yield f"author:{self.author_names[a]}", "wrote", f"paper:{p}"


def build_knowledge_graph(ds: Dataset) -> KnowledgeGraph:
    """Build the typed graph from a dataset. Author names are matched exactly as written (no disambiguation)."""
    index: Dict[str, int] = {}
    pairs: List[Tuple[int, int]] = []
    for paper, names in enumerate(ds.authors):
        for name in sorted(set(names)):
            pairs.append((index.setdefault(name, len(index)), paper))
    wrote = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    names = [""] * len(index)
    for name, i in index.items():
        names[i] = name
    return KnowledgeGraph(ds.n_papers, len(index), ds.edges.astype(np.int64), wrote, names, ds.abstracts)
