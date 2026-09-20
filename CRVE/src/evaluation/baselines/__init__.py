"""
CRVE Evaluation Baselines (PyTerrier Classical + Neural).
"""

from .pyterrier_harness import (
    PyTerrierIndexManager,
    PyTerrierBaselineHarness,
    init_pyterrier,
    sanitize_default_query,
)
from .pyterrier_qe import (
    TerrierQueryAnalyzer,
    BGEVocabSidecarManager,
    get_terrier_analyzer,
)
from .dense_rag import DenseRAGBaseline
from .splade import SPLADEBaseline, SparseInvertedIndex

__all__ = [
    "PyTerrierIndexManager",
    "PyTerrierBaselineHarness",
    "init_pyterrier",
    "sanitize_default_query",
    "TerrierQueryAnalyzer",
    "BGEVocabSidecarManager",
    "get_terrier_analyzer",
    "DenseRAGBaseline",
    "SPLADEBaseline",
    "SparseInvertedIndex",
]
