"""
tests/test_pyterrier_harness_v2.py

Unit and parity tests for the refactored PyTerrier Baseline Harness:
1. Pre/post exclusion filtering preserves PyTerrier schema and recomputes contiguous ranks.
2. Oracle nDCG@10 linear gain calculation against full IDCG.
3. Linear nDCG vs. Exponential nDCG differentiation on graded qrels.
4. Deterministic percentile calculation for API latency.
5. Pre-PRF exclusion filtering prevents leakage into expansion models.
"""

import math
import pytest
import pandas as pd
import numpy as np
import pyterrier as pt

from evaluation.baselines.pyterrier_harness import (
    init_pyterrier,
    create_exclusion_filter,
    calc_oracle_ndcg_10,
    BEIR_EXP_GAINS,
)


@pytest.fixture(scope="module")
def setup_pt():
    init_pyterrier(mem=1024)


def test_exclusion_filter_padding_and_ranks():
    """Verifies exclusion filter drops target docnos, slices to max_docs, and recomputes ranks."""
    df = pd.DataFrame([
        {"qid": "q1", "docno": "doc_ex1", "score": 10.0, "query": "test query"},
        {"qid": "q1", "docno": "doc_keep1", "score": 9.0, "query": "test query"},
        {"qid": "q1", "docno": "doc_ex2", "score": 8.0, "query": "test query"},
        {"qid": "q1", "docno": "doc_keep2", "score": 7.0, "query": "test query"},
        {"qid": "q1", "docno": "doc_keep3", "score": 6.0, "query": "test query"},
        {"qid": "q2", "docno": "doc_keep4", "score": 15.0, "query": "test query 2"},
    ])

    excluded_map = {"q1": {"doc_ex1", "doc_ex2"}}
    filter_tf = create_exclusion_filter(excluded_map, max_docs=2)
    filtered = filter_tf.transform(df)

    # q1 should only have doc_keep1 and doc_keep2 (sliced to max_docs=2)
    q1_docs = filtered[filtered["qid"] == "q1"]["docno"].tolist()
    assert q1_docs == ["doc_keep1", "doc_keep2"]

    # Ranks must be contiguous 0, 1
    q1_ranks = filtered[filtered["qid"] == "q1"]["rank"].tolist()
    assert q1_ranks == [0, 1]

    # q2 should retain doc_keep4 with rank 0
    q2_docs = filtered[filtered["qid"] == "q2"]["docno"].tolist()
    assert q2_docs == ["doc_keep4"]
    assert filtered[filtered["qid"] == "q2"]["rank"].tolist() == [0]


def test_oracle_ndcg_calc():
    """Verifies Oracle nDCG@10 uses linear gain against full IDCG."""
    gold_map = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
    
    # Perfect retrieval: d1, d2, d3 in candidate list -> Oracle nDCG = 1.0
    cand_perfect = ["d3", "d2", "d1", "d_irrel1"]
    assert math.isclose(calc_oracle_ndcg_10(cand_perfect, gold_map), 1.0, rel_tol=1e-5)

    # Incomplete candidate list: only d3 (rel=1) retrieved
    cand_incomplete = ["d3", "d_irrel1"]
    # Ideal DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5 = 4.76186
    # Oracle DCG = 1/log2(2) = 1.0
    expected_oracle = 1.0 / (3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4))
    assert math.isclose(calc_oracle_ndcg_10(cand_incomplete, gold_map), expected_oracle, rel_tol=1e-5)


def test_linear_vs_exp_ndcg_graded():
    """Verifies that linear nDCG differs from exponential nDCG on graded relevance."""
    import ir_measures
    from ir_measures import nDCG

    qrels = [
        ir_measures.Qrel("q1", "d1", 3),  # High grade
        ir_measures.Qrel("q1", "d2", 1),  # Low grade
    ]
    # Inverted ranking: d2 ranked at 1, d1 ranked at 2
    run = [
        ir_measures.ScoredDoc("q1", "d2", 2.0),
        ir_measures.ScoredDoc("q1", "d1", 1.0),
    ]

    meas_linear = [nDCG @ 10]
    meas_exp = [nDCG(gains=BEIR_EXP_GAINS) @ 10]

    val_linear = ir_measures.calc_aggregate(meas_linear, qrels, run)[meas_linear[0]]
    val_exp = ir_measures.calc_aggregate(meas_exp, qrels, run)[meas_exp[0]]

    # Exponential gains penalize inverted high-grade relevance much more severely
    assert val_linear > 0.0
    assert val_exp > 0.0
    assert not math.isclose(val_linear, val_exp, rel_tol=1e-3)


def test_deterministic_latency_percentiles():
    """Verifies percentile calculations for P50, P90, P99 on synthetic timings."""
    timings = list(range(1, 101))  # 1 to 100 ms
    p50 = float(np.percentile(timings, 50))
    p90 = float(np.percentile(timings, 90))
    p99 = float(np.percentile(timings, 99))

    assert math.isclose(p50, 50.5, rel_tol=1e-3)
    assert math.isclose(p90, 90.1, rel_tol=1e-3)
    assert math.isclose(p99, 99.01, rel_tol=1e-3)


def test_pre_prf_exclusion_prevents_leakage(setup_pt, tmp_path):
    """
    Verifies that inserting create_exclusion_filter before RM3 prevents
    an excluded document from supplying feedback terms.
    """
    import os
    scifact_index = os.path.abspath("data/cache/terrier_indices/scifact_default")
    if os.path.exists(os.path.join(scifact_index, "data.properties")):
        index = pt.IndexFactory.of(scifact_index)
    else:
        # Fallback toy index
        docs = [{"docno": f"doc_{i}", "text": f"asthma symptoms treatment clinical trial {i}"} for i in range(100)]
        docs.append({"docno": "doc_ex", "text": "asthma fluticasone montelukast corticosteroid inhaler"})
        index_ref = pt.IterDictIndexer(str(tmp_path / "toy"), overwrite=True, meta={"docno": 64, "text": 512}).index(docs)
        index = pt.IndexFactory.of(index_ref)

    df_q = pd.DataFrame([{"qid": "q1", "query": "asthma symptoms treatment"}])
    pass1 = pt.terrier.Retriever(index, wmodel="BM25", num_results=10)
    top_hits = pass1.transform(df_q)
    top_docno = str(top_hits.iloc[0]["docno"])

    # 1. Unfiltered pipeline: top_docno participates in PRF
    rm3 = pt.rewrite.RM3(index, fb_terms=5, fb_docs=3, fb_lambda=0.5)
    q_unfiltered = (pass1 >> rm3).transform(df_q).iloc[0]["query"]

    # 2. Filtered pipeline: top_docno is excluded before RM3
    excluded_map = {"q1": {top_docno}}
    filter1 = create_exclusion_filter(excluded_map, max_docs=3)
    q_filtered = (pass1 >> filter1 >> rm3).transform(df_q).iloc[0]["query"]

    # The queries must differ because the top feedback document was excluded
    assert q_unfiltered != q_filtered
    assert len(q_filtered.split()) >= 2
