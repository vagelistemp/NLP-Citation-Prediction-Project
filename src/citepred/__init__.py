"""citepred: leakage-free citation (link) prediction with text, author and graph features.

Modules
-------
io          loading the challenge-format data, writing submissions
synthetic   a small synthetic citation network (text, authors, edges, labelled test pairs)
graph       sparse-matrix graph features (degrees, neighbours, Adamic-Adar, paths, PageRank, SVD)
text        TF-IDF and SBERT similarity, graph-aware embedding features
authors     shared authors and the author-to-author citation prior
features    feature groups and the FeatureBuilder
protocol    the leakage-free edge-out-of-fold training-set construction (and the leaky one it replaces)
model       LightGBM cross-validation and metrics
embed       resumable SBERT embedding of the abstracts
"""
import os as _os
import sys as _sys

# Windows only: PyTorch (SBERT) and LightGBM/others each ship an OpenMP runtime; loading both can abort the process with
# "OMP: Error #15". Intel's documented workaround is KMP_DUPLICATE_LIB_OK=TRUE (unsupported, hence only set if you have not
# chosen a value yourself). Numerical results are checked in the tests.
if _sys.platform == "win32":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
