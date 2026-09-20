"""
Edge-RAG Phase 2: Selection Module.
Implements Gate 1 Candidate Proposal, Metrics, and Evaluation Harneeses.
"""

from .gate1_metrics import (
    TAU,
    EPSILON,
    DEFAULT_DELTA,
    DEFAULT_RHO,
    compute_safe_ranking_gain,
    compute_safe_recall_gain,
    get_ranking_helpful_terms,
    get_recall_helpful_terms,
    get_near_best_terms,
    compute_reference_bor,
    compute_near_best_hit,
    compute_term_recall,
    compute_term_precision,
    compute_recall_hit,
    compute_doc_opportunity_recall,
    compute_waste_metrics,
    classify_document_transition,
)

__all__ = [
    "TAU",
    "EPSILON",
    "DEFAULT_DELTA",
    "DEFAULT_RHO",
    "compute_safe_ranking_gain",
    "compute_safe_recall_gain",
    "get_ranking_helpful_terms",
    "get_recall_helpful_terms",
    "get_near_best_terms",
    "compute_reference_bor",
    "compute_near_best_hit",
    "compute_term_recall",
    "compute_term_precision",
    "compute_recall_hit",
    "compute_doc_opportunity_recall",
    "compute_waste_metrics",
    "classify_document_transition",
]
