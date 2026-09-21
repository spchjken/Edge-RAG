"""
tests/test_pyterrier_metrics_parity.py

Phase 0 Verification Test:
Verifies mathematical parity between internal metrics (src/evaluation/metrics.py)
and ir_measures library to ensure standardized, un-confounded evaluation across
all BEIR and BRIGHT baselines.
"""

import math
import pytest
import ir_measures
from ir_measures import nDCG, RR, R, P

from evaluation.metrics import (
    calculate_ndcg_at_k,
    calculate_mrr_at_k,
    calculate_recall_at_k,
    calculate_precision_at_k,
)

# Standard BEIR Table 2 exponential gain mapping (2^rel - 1)
BEIR_EXP_GAINS = {1: 1, 2: 3, 3: 7, 4: 15}


def test_binary_relevance_parity():
    """Verify 16-decimal parity on binary relevance for nDCG@10, RR@10, R@10, P@10."""
    qrels = [
        ir_measures.Qrel("q1", "d1", 1),
        ir_measures.Qrel("q1", "d3", 1),
    ]
    # Provide 10 retrieved documents so rank cutoff K=10 matches list length
    docs = [f"d{i}" for i in range(1, 11)]
    run = [ir_measures.ScoredDoc("q1", doc, 1.0 - i * 0.05) for i, doc in enumerate(docs)]
    retrieved = list(docs)
    gold = ["d1", "d3"]

    ir_res = ir_measures.calc_aggregate([nDCG@10, RR@10, R@10, P@10], qrels, run)

    internal_ndcg = calculate_ndcg_at_k(retrieved, gold, k=10)
    internal_mrr = calculate_mrr_at_k(retrieved, gold, k=10)
    internal_recall = calculate_recall_at_k(retrieved, gold)
    internal_precision = calculate_precision_at_k(retrieved, gold)

    assert math.isclose(ir_res[nDCG@10], internal_ndcg, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ir_res[RR@10], internal_mrr, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ir_res[R@10], internal_recall, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ir_res[P@10], internal_precision, rel_tol=1e-12, abs_tol=1e-12)


def test_graded_relevance_parity_beir_exp_gains():
    """Verify exact parity on graded relevance using BEIR exponential gain mapping."""
    qrels = [
        ir_measures.Qrel("q1", "d1", 2),
        ir_measures.Qrel("q1", "d3", 1),
    ]
    run = [
        ir_measures.ScoredDoc("q1", "d1", 0.9),
        ir_measures.ScoredDoc("q1", "d2", 0.8),
        ir_measures.ScoredDoc("q1", "d3", 0.7),
        ir_measures.ScoredDoc("q1", "d4", 0.6),
    ]

    retrieved = ["d1", "d2", "d3", "d4"]
    gold = {"d1": 2.0, "d3": 1.0}

    # ir_measures with pinned BEIR exponential gains: {1: 1, 2: 3, 3: 7, 4: 15}
    measure = nDCG(gains=BEIR_EXP_GAINS)@10
    ir_res = ir_measures.calc_aggregate([measure], qrels, run)

    internal_ndcg = calculate_ndcg_at_k(retrieved, gold, k=10)

    assert math.isclose(ir_res[measure], internal_ndcg, rel_tol=1e-12, abs_tol=1e-12)


def test_multi_query_mean_aggregation():
    """Verify aggregation across multiple queries matches between metrics."""
    qrels = [
        ir_measures.Qrel("q1", "d1", 1),
        ir_measures.Qrel("q1", "d2", 1),
        ir_measures.Qrel("q2", "d10", 2),
        ir_measures.Qrel("q2", "d20", 1),
    ]
    run = [
        ir_measures.ScoredDoc("q1", "d2", 0.95),
        ir_measures.ScoredDoc("q1", "d1", 0.85),
        ir_measures.ScoredDoc("q2", "d99", 0.99),
        ir_measures.ScoredDoc("q2", "d10", 0.80),
        ir_measures.ScoredDoc("q2", "d20", 0.70),
    ]

    measure_ndcg = nDCG(gains=BEIR_EXP_GAINS)@10
    measure_mrr = RR@10
    measure_recall = R@10

    ir_res = ir_measures.calc_aggregate([measure_ndcg, measure_mrr, measure_recall], qrels, run)

    # Compute internal mean
    q1_ret = ["d2", "d1"]
    q1_gold = {"d1": 1.0, "d2": 1.0}
    q2_ret = ["d99", "d10", "d20"]
    q2_gold = {"d10": 2.0, "d20": 1.0}

    mean_ndcg = (calculate_ndcg_at_k(q1_ret, q1_gold, k=10) + calculate_ndcg_at_k(q2_ret, q2_gold, k=10)) / 2.0
    mean_mrr = (calculate_mrr_at_k(q1_ret, q1_gold, k=10) + calculate_mrr_at_k(q2_ret, q2_gold, k=10)) / 2.0
    mean_recall = (calculate_recall_at_k(q1_ret, q1_gold) + calculate_recall_at_k(q2_ret, q2_gold)) / 2.0

    assert math.isclose(ir_res[measure_ndcg], mean_ndcg, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ir_res[measure_mrr], mean_mrr, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ir_res[measure_recall], mean_recall, rel_tol=1e-12, abs_tol=1e-12)


def test_edge_cases_empty_and_zero_hits():
    """Verify graceful handling of empty retrieved lists and zero hits."""
    qrels = [ir_measures.Qrel("q1", "d1", 1)]
    run_empty = []
    run_miss = [ir_measures.ScoredDoc("q1", "d99", 0.5)]

    measure = nDCG@10
    res_empty = ir_measures.calc_aggregate([measure], qrels, run_empty)
    res_miss = ir_measures.calc_aggregate([measure], qrels, run_miss)

    assert res_empty[measure] == 0.0
    assert res_miss[measure] == 0.0

    assert calculate_ndcg_at_k([], ["d1"], k=10) == 0.0
    assert calculate_ndcg_at_k(["d99"], ["d1"], k=10) == 0.0
    assert calculate_mrr_at_k(["d99"], ["d1"], k=10) == 0.0
    assert calculate_recall_at_k(["d99"], ["d1"]) == 0.0
    assert calculate_precision_at_k(["d99"], ["d1"]) == 0.0
