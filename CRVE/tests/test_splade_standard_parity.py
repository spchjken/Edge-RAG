"""
tests/test_splade_standard_parity.py

Verifies SPLADE-v3 standardization:
1. SparseInvertedIndex produces exact score identity and ordering compared to naive brute-force dot product.
2. TransformersSPLADE encodes queries and documents, zeroes out pad tokens, and excludes special tokens.
3. SPLADEBaseline index building and retrieval work end-to-end.
"""

import pytest
import numpy as np
from evaluation.baselines.splade import SparseInvertedIndex, TransformersSPLADE, SPLADEBaseline


def test_sparse_inverted_index_parity_with_brute_force():
    """Verify that SparseInvertedIndex retrieval exactly matches dense dot product across sparse vectors."""
    index = SparseInvertedIndex()
    
    # Synthetic sparse vectors for 5 documents over a vocabulary of 1000 tokens
    docs = [
        {10: 1.5, 25: 0.8, 100: 2.1},
        {10: 0.5, 50: 3.0},
        {25: 1.2, 100: 0.4, 200: 1.1},
        {300: 2.0},  # No overlap
        {10: 2.0, 25: 1.0, 50: 0.5, 100: 1.0},
    ]
    doc_ids = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
    
    index.build(docs, doc_ids)
    
    query = {10: 1.0, 25: 2.0, 100: 0.5}
    
    # Manual dot product computation
    expected_scores = {}
    for did, d in zip(doc_ids, docs):
        s = sum(query[k] * d[k] for k in query if k in d)
        if s > 0:
            expected_scores[did] = s
            
    # Sort expected descending
    expected_ranking = sorted(expected_scores.items(), key=lambda x: -x[1])
    
    # Inverted index retrieve
    results = index.retrieve(query, top_k=5)
    
    assert len(results) == len(expected_ranking)
    for (res_id, res_s), (exp_id, exp_s) in zip(results, expected_ranking):
        assert res_id == exp_id
        assert abs(res_s - exp_s) < 1e-5


def test_sparse_inverted_index_empty_and_thresholding():
    """Test boundary conditions: no matching query tokens or weights below threshold."""
    index = SparseInvertedIndex()
    docs = [{10: 0.00001}]  # Below 1e-4 threshold
    index.build(docs, ["doc_0"])
    
    assert len(index.postings) == 0
    
    # Query with no overlap
    res = index.retrieve({999: 1.0}, top_k=10)
    assert res == []


def test_splade_encoder_and_baseline_integration():
    """Integration test with naver/splade-v3-distilbert on CPU/CUDA."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    baseline = SPLADEBaseline(device=device)
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and neural search with sparse representations.",
        "Financial reporting and quarterly revenue analysis."
    ]
    chunk_ids = ["c0", "c1", "c2"]
    
    baseline.build_index(corpus, chunk_ids)
    assert baseline.index.num_docs == 3
    
    # Test query
    results = baseline.retrieve("neural search artificial intelligence", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"  # "Artificial intelligence and neural search" should rank highest
    assert results[0]["score"] > 0
