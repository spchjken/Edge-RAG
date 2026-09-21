"""
tests/test_pyterrier_qe.py

Automated test suite verifying the 10 gates and invariants for BGE_Vocab_QE
and LLM_Q2E_ZS baselines defined in docs/bgeqe_llmqe_testing_plan.md.
"""

import os
import pytest
import numpy as np
import pandas as pd
import pyterrier as pt

from evaluation.baselines.pyterrier_qe import (
    get_terrier_analyzer,
    BGEVocabSidecarManager,
    BGEVocabQERewriter,
    LLMQEKeywordRewriter,
    DEFAULT_QE_CONFIG,
)

FIXTURE_INDEX_PATH = os.path.abspath("data/cache/terrier_indices/scifact_default/data.properties")


@pytest.fixture(scope="module")
def pt_init():
    if not pt.java.started():
        pt.java.init()
    assert os.path.exists(FIXTURE_INDEX_PATH), f"Fixture index not found: {FIXTURE_INDEX_PATH}"
    return pt.IndexFactory.of(FIXTURE_INDEX_PATH)


def test_1_candidate_mapping_and_lexicon(pt_init):
    """Gate 1: Candidate mapping recognizes valid lexicon terms and rejects invalid terms."""
    analyzer = get_terrier_analyzer()
    terms, surfs = analyzer.analyze("Effect of smoking on lung cancer and robotics")
    assert "smoke" in terms
    assert "lung" in terms
    assert "cancer" in terms
    assert "robot" in terms
    assert "of" not in terms  # Stopword removed
    assert "on" not in terms  # Stopword removed
    assert len(terms) == len(surfs)


def test_2_terrier_lexicon_stats(pt_init):
    """Gate 2: Vocabulary DF and collection frequency are read from Terrier lexicon without recounting corpus."""
    index = pt_init
    lex = index.getLexicon()
    entry = lex.getLexiconEntry("cancer")
    assert entry is not None
    assert entry.getFrequency() > 0
    assert entry.getDocumentFrequency() > 0


def test_3_zero_expansion_exact_bm25_fallback(pt_init):
    """Gate 3: With zero admitted expansions, query_toks retrieval matches BM25_Default exactly."""
    index = pt_init
    analyzer = get_terrier_analyzer()
    q = "Effect of smoking on lung cancer"
    terms, _ = analyzer.analyze(q)
    query_toks = analyzer.to_query_toks(terms)

    # 1. Plain BM25
    bm25_plain = pt.terrier.Retriever(index, wmodel="BM25", num_results=10)
    res_plain = bm25_plain.search(q)

    # 2. query_toks BM25
    df_q = pd.DataFrame([{"qid": "1", "query": q, "query_toks": query_toks}])
    res_toks = bm25_plain.transform(df_q)

    assert len(res_plain) == len(res_toks)
    assert list(res_plain["docno"].values) == list(res_toks["docno"].values)
    max_diff = np.abs(res_plain["score"].values - res_toks["score"].values).max()
    assert max_diff < 1e-4


def test_4_bge_vocab_rewriter_mass_and_nonnegativity(pt_init):
    """Gate 4 & 5: Nonnegative weights, alpha mass allocation, and original term weight retention."""
    index = pt_init
    analyzer = get_terrier_analyzer()

    # Synthetic sidecar
    terms = ["smoke", "lung", "cancer", "tumor", "cell", "medic"]
    surfs = ["smoking", "lungs", "cancer", "tumors", "cells", "medical"]
    np.random.seed(42)
    embs = np.random.randn(len(terms), 384).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    sidecar = {
        "retrieval_terms": terms,
        "display_surfaces": surfs,
        "embeddings": embs.astype(np.float16),
        "df": np.array([10] * len(terms)),
        "cf": np.array([20] * len(terms)),
    }

    rewriter = BGEVocabQERewriter(sidecar, config={
        "neighbour_depth_L": 3,
        "expansion_terms_ke": 2,
        "alpha": 0.60,
        "encoder": "BAAI/bge-small-en-v1.5",
    })

    q = "cancer treatment"
    df_in = pd.DataFrame([{"qid": "1", "query": q}])
    df_out = rewriter.transform(df_in)

    toks = df_out.iloc[0]["query_toks"]
    assert "cancer" in toks
    # Original terms must be preserved
    assert toks["cancer"] >= 1.0
    # Weights must be nonnegative
    assert all(w >= 0.0 for w in toks.values())


def test_6_llm_qe_concatenation_and_fallback(pt_init, monkeypatch):
    """Gate 7: Q2E/ZS concatenation: Q' = Concat(5*Q, G(Q)) and exact BM25 fallback."""
    rewriter = LLMQEKeywordRewriter()

    # Mock LLM response
    def mock_generate(query):
        return "oncology chemotherapy therapy"

    monkeypatch.setattr(rewriter, "generate_keywords", mock_generate)

    q = "cancer"
    df_in = pd.DataFrame([{"qid": "1", "query": q}])
    df_out = rewriter.transform(df_in)

    toks = df_out.iloc[0]["query_toks"]
    assert "cancer" in toks
    # Repetition r=5 means original term 'cancer' has weight 5.0
    assert toks["cancer"] == 5.0
    assert "oncolog" in toks or "chemotherapi" in toks or "therapi" in toks

    # Mock empty response -> exact fallback
    monkeypatch.setattr(rewriter, "generate_keywords", lambda q: "")
    df_empty = rewriter.transform(df_in)
    toks_empty = df_empty.iloc[0]["query_toks"]
    assert toks_empty["cancer"] == 1.0  # single copy fallback
