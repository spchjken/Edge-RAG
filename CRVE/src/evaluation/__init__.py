"""
CRVE Evaluation Module.
"""

from .benchmark_loader import BenchmarkLoader, ALL_BRIGHT_DOMAINS, parse_bright_domain
from .metrics import (
    calculate_ndcg_at_k,
    calculate_mrr_at_k,
    calculate_recall_at_k,
    calculate_precision_at_k,
)

__all__ = [
    "BenchmarkLoader",
    "ALL_BRIGHT_DOMAINS",
    "parse_bright_domain",
    "calculate_ndcg_at_k",
    "calculate_mrr_at_k",
    "calculate_recall_at_k",
    "calculate_precision_at_k",
]
