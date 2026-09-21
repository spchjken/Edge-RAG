"""
tests/test_bright_parity.py

Verification tests for BRIGHT reasoning benchmark loading, streaming,
and candidate exclusion protocol.
"""

import os
import pytest
import pyterrier as pt
import ir_measures

from evaluation.benchmark_loader import BenchmarkLoader, ALL_BRIGHT_DOMAINS, parse_bright_domain
from evaluation.baselines.pyterrier_harness import (
    PyTerrierIndexManager,
    PyTerrierBaselineHarness,
    init_pyterrier,
)


def test_bright_domain_recognition():
    """Verify all 12 BRIGHT domains are recognized by parse_bright_domain."""
    assert len(ALL_BRIGHT_DOMAINS) == 12
    for d in ALL_BRIGHT_DOMAINS:
        assert parse_bright_domain(f"bright_{d}") == d
        assert parse_bright_domain(d) == d
        assert parse_bright_domain(f"bright_{d}_doc_level") == d


def test_bright_metadata_doc_counts():
    """Verify O(1) parquet metadata doc count across all 12 domains."""
    expected_domains = [
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode", "pony", "aops",
        "theoremqa_questions", "theoremqa_theorems"
    ]
    for d in expected_domains:
        cnt = BenchmarkLoader.get_corpus_doc_count(f"bright_{d}")
        assert cnt > 0, f"Expected positive doc count for bright_{d}, got {cnt}"


def test_bright_streaming_corpus():
    """Verify streaming parquet generator yields valid (doc_id, text) tuples."""
    stream = BenchmarkLoader.stream_corpus("bright_pony")
    sample_docs = []
    for _ in range(5):
        sample_docs.append(next(stream))

    assert len(sample_docs) == 5
    for did, text in sample_docs:
        assert isinstance(did, str) and len(did) > 0
        assert isinstance(text, str) and len(text) > 0


def test_bright_query_loading_and_excluded_ids():
    """Verify query loading and extracted excluded_doc_ids for reasoning domains."""
    # 1. Pony
    pony_queries, pony_stats = BenchmarkLoader.load_queries("bright_pony")
    assert len(pony_queries) > 0
    assert "dataset" in pony_stats
    q0 = pony_queries[0]
    assert "query_id" in q0 and "question" in q0 and "gold_doc_ids" in q0 and "qrels" in q0
    assert "excluded_doc_ids" in q0

    # 2. LeetCode (known to contain excluded_ids in official BRIGHT)
    lc_queries, lc_stats = BenchmarkLoader.load_queries("bright_leetcode")
    assert len(lc_queries) > 0
    # Confirm some queries in leetcode have excluded_doc_ids
    has_excluded = any(len(q["excluded_doc_ids"]) > 0 for q in lc_queries)
    assert has_excluded, "LeetCode benchmark is expected to have excluded_ids"


def test_bright_candidate_exclusion_filtering():
    """Verifies that evaluate_pipeline cleanly filters out excluded_doc_ids."""
    init_pyterrier()

    # Small synthetic query set with excluded doc
    queries = [
        {
            "query_id": "q_test_1",
            "question": "test query",
            "gold_doc_ids": ["d2"],
            "qrels": {"d2": 1.0},
            "excluded_doc_ids": {"d1"}  # d1 is excluded
        }
    ]
    qrels_ir = [ir_measures.Qrel("q_test_1", "d2", 1)]
    gold_map = {"q_test_1": {"d2": 1.0}}

    # Synthetic transformer that returns d1 and d2
    import pandas as pd
    class MockTransformer:
        def transform(self, df_q):
            return pd.DataFrame([
                {"qid": "q_test_1", "docno": "d1", "score": 10.0, "rank": 0},
                {"qid": "q_test_1", "docno": "d2", "score": 5.0, "rank": 1},
            ])

    harness = PyTerrierBaselineHarness()
    harness.pipelines = {"MockPipe": MockTransformer()}

    res = harness.evaluate_pipeline(
        "MockPipe", queries, qrels_ir, gold_map, chunk_size=1
    )

    # d1 was excluded, so d2 is top hit (rank 1 -> 1.0 ndcg / mrr)
    assert res["mrr_10"] == 1.0
    assert res["strict_10"] == 1.0
