"""
tests/test_metrics_graded_ndcg.py

Unit tests for graded nDCG@K and backward-compatible metrics in src/evaluation/metrics.py.
Includes hand-computed ground truth verification.
"""

import math
import pytest
from evaluation.metrics import (
    calculate_ndcg_at_k,
    calculate_mrr_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k
)


def test_graded_ndcg_hand_computed():
    """
    Hand-computed test case:
    Ground truth:
      doc_A: rel = 3.0  (gain = 2^3 - 1 = 7.0)
      doc_B: rel = 1.0  (gain = 2^1 - 1 = 1.0)
      doc_C: rel = 0.0  (gain = 0.0)

    Ideal ranking (sorted descending by rel):
      rank 1: doc_A (rel=3, gain=7, discount = 1/log2(2) = 1.0) -> 7.0
      rank 2: doc_B (rel=1, gain=1, discount = 1/log2(3) = 0.63092975) -> 0.63092975
      IDCG@2 = 7.0 + 0.63092975 = 7.63092975

    Case 1: Perfect retrieval [doc_A, doc_B] -> DCG@2 = IDCG@2 -> nDCG = 1.0
    """
    qrels = {"doc_A": 3.0, "doc_B": 1.0, "doc_C": 0.0}
    ndcg_perfect = calculate_ndcg_at_k(["doc_A", "doc_B"], qrels, k=2)
    assert pytest.approx(ndcg_perfect, 1e-6) == 1.0

    # Case 2: Inverted retrieval [doc_B, doc_A]
    # DCG@2 = 1.0 / log2(2) + 7.0 / log2(3) = 1.0 + 7.0 / 1.5849625 = 1.0 + 4.416508 = 5.416508
    # nDCG@2 = 5.416508 / 7.63092975 = 0.7098097
    dcg_inverted = 1.0 / math.log2(2) + 7.0 / math.log2(3)
    idcg_expected = 7.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_ndcg = dcg_inverted / idcg_expected

    ndcg_inverted = calculate_ndcg_at_k(["doc_B", "doc_A"], qrels, k=2)
    assert pytest.approx(ndcg_inverted, 1e-6) == expected_ndcg

    # Case 3: Partial retrieval with distractor [doc_X, doc_A]
    # DCG@2 = 0.0 + 7.0 / log2(3) = 4.416508
    # nDCG@2 = 4.416508 / 7.63092975 = 0.578764
    dcg_partial = 7.0 / math.log2(3)
    expected_partial = dcg_partial / idcg_expected
    ndcg_partial = calculate_ndcg_at_k(["doc_X", "doc_A"], qrels, k=2)
    assert pytest.approx(ndcg_partial, 1e-6) == expected_partial


def test_binary_backward_compatibility():
    """Verify that List[str] input produces exact binary nDCG results."""
    gold_list = ["doc_A", "doc_B"]
    # Perfect retrieval: 1/log2(2) + 1/log2(3) / (1/log2(2) + 1/log2(3)) = 1.0
    assert calculate_ndcg_at_k(["doc_A", "doc_B"], gold_list, k=10) == 1.0

    # Half hit at rank 1:
    # DCG = 1/log2(2) = 1.0. IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert pytest.approx(calculate_ndcg_at_k(["doc_A", "doc_X"], gold_list, k=10), 1e-6) == expected


def test_metrics_dict_backward_compatibility():
    """Verify that Precision, Recall, and MRR correctly interpret Dict[str, float]."""
    qrels = {"doc_1": 2.0, "doc_2": 1.0, "doc_3": 0.0}
    retrieved = ["doc_X", "doc_2", "doc_1"]

    # MRR: first hit is doc_2 at rank 2 -> 1/2 = 0.5
    assert calculate_mrr_at_k(retrieved, qrels, k=10) == 0.5

    # Recall: retrieved 2 positive docs out of 2 positive docs -> 1.0
    assert calculate_recall_at_k(retrieved, qrels) == 1.0

    # Precision: 2 hits out of 3 retrieved -> 2/3
    assert pytest.approx(calculate_precision_at_k(retrieved, qrels), 1e-6) == 2.0 / 3.0
