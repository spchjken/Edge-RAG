"""
tests/test_dense_splade_safety.py

Crash safety, resource boundary, and edge-case regression tests for:
- DenseRAGBaseline (BAAI/bge-small-en-v1.5)
- SPLADEBaseline (naver/splade-v3-distilbert)

Verifies:
1. Model loading, FP16 initialization, and warmup without CUDA/OOM faults.
2. Indexing and retrieval across micro and medium batches.
3. Edge-case resilience: empty queries, single characters, long queries, non-ASCII/unicode, special symbols.
4. Absence of memory leaks (VRAM drift) over repeated query executions.
"""

import gc
import pytest
import numpy as np
import torch

from evaluation.baselines.dense_rag import DenseRAGBaseline
from evaluation.baselines.splade import SPLADEBaseline


@pytest.fixture(scope="module")
def sample_corpus():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning and artificial intelligence are transforming modern research.",
        "Edge-RAG reduces VRAM consumption using statistical query expansion and cascade routing.",
        "Lucene BM25 and SPLADE neural sparse models provide effective keyword search.",
    ]


@pytest.fixture(scope="module")
def sample_chunk_ids(sample_corpus):
    return [f"c_{i}" for i in range(len(sample_corpus))]


def test_dense_rag_end_to_end(sample_corpus, sample_chunk_ids):
    """Verifies DenseRAGBaseline initialization, warmup, indexing, and retrieval."""
    dense = DenseRAGBaseline(use_fp16=True)
    dense.warmup()
    dense.build_index(sample_corpus, sample_chunk_ids)

    assert dense._corpus_embeddings is not None
    assert dense._corpus_embeddings.shape == (len(sample_corpus), 384)

    results = dense.retrieve("Edge RAG sparse retrieval BM25", top_k=2)
    assert len(results) == 2
    assert all("chunk_id" in r and "score" in r and "text" in r for r in results)
    assert results[0]["score"] > results[1]["score"]


def test_dense_rag_stress_and_edge_cases():
    """Verifies DenseRAGBaseline handles scaling to 500 documents and extreme query edge cases."""
    dense = DenseRAGBaseline(use_fp16=True)
    dense.warmup()

    med_corpus = [
        f"Passage {i}: Distributed systems and financial clearing networks require high throughput low latency. Doc index {i%10}."
        for i in range(500)
    ]
    med_ids = [f"doc_{i}" for i in range(500)]
    dense.build_index(med_corpus, med_ids)

    edge_cases = [
        "",  # Empty query
        "a",  # Single character
        "financial clearing " * 100,  # Long query (>512 tokens)
        "こんにちは 世界 🚀 Schrödinger's cat éàü",  # Non-ASCII / Unicode
        "!@#$%^&*()_+=-`~[]{}|;':\",./<>?",  # Special symbols
    ]
    for q in edge_cases:
        res = dense.retrieve(q, top_k=5)
        assert len(res) == 5

    # Request more results than corpus size
    res_large = dense.retrieve("financial clearing", top_k=1000)
    assert len(res_large) == 500


def test_splade_end_to_end(sample_corpus, sample_chunk_ids):
    """Verifies SPLADEBaseline initialization, warmup, indexing, and retrieval."""
    splade = SPLADEBaseline()
    splade.warmup()
    splade.build_index(sample_corpus, sample_chunk_ids)

    assert splade.index.num_docs == len(sample_corpus)
    assert len(splade.index.postings) > 0

    results = splade.retrieve("Edge RAG sparse retrieval BM25", top_k=2)
    assert len(results) > 0
    assert all("chunk_id" in r and "score" in r and "text" in r for r in results)
    assert results[0]["chunk_id"] == "c_2" or results[0]["chunk_id"] == "c_3"


def test_splade_stress_and_edge_cases():
    """Verifies SPLADEBaseline handles scaling to 500 documents and extreme query edge cases."""
    splade = SPLADEBaseline()
    splade.warmup()

    med_corpus = [
        f"Passage {i}: Distributed systems and financial clearing networks require high throughput low latency. Doc index {i%10}."
        for i in range(500)
    ]
    med_ids = [f"doc_{i}" for i in range(500)]
    splade.build_index(med_corpus, med_ids)

    edge_cases = [
        ("", 0),  # Empty query -> 0 hits
        ("a", 5),  # Single character -> valid hits
        ("financial clearing " * 100, 5),  # Long query (>512 tokens)
        ("こんにちは 世界 🚀 Schrödinger's cat éàü", 1),  # Non-ASCII / Unicode -> at least 1 hit
        ("!@#$%^&*()_+=-`~[]{}|;':\",./<>?", 5),  # Special symbols
    ]
    for q, min_expected in edge_cases:
        res = splade.retrieve(q, top_k=5)
        assert len(res) >= min_expected

    # Large top_k clamped to corpus size
    res_large = splade.retrieve("financial clearing", top_k=1000)
    assert len(res_large) <= 500


def test_memory_leak_stability():
    """Verifies that running repeated queries does not cause runaway VRAM allocation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dense = DenseRAGBaseline(use_fp16=True)
    dense.warmup()
    dense.build_index(["Sample document for memory leak verification."], ["d0"])

    vram_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for _ in range(30):
        _ = dense.retrieve("leak check query", top_k=1)
    vram_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    assert (vram_end - vram_start) < 1024 * 1024, "Dense baseline leaked >1MB VRAM across 30 queries"
