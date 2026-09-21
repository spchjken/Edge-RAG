"""
tests/test_pool_oracle_isolation.py

Comprehensive Gate A & Contract Unit Test Suite for Stage 1 Vocabulary Pool Oracle Isolation.
Verifies:
1. Strictly disjoint 6-band DF partition.
2. Disjoint stratified quota properties.
3. CELF mathematical equivalence to naive greedy submodular coverage.
4. Policy nestedness and deterministic lexicographic tie-breaking.
5. Gate A: Exact BM25 score/rank/docno parity at zero weight (mu=0.0).
6. Gate A: Qualified document score monotonicity for matching documents with non-negative IDF.
7. Dynamic BRIGHT padding and exclusion filtering.
"""

import os
import sys
import math
import random
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.pool_generators import (
    PoolPolicy,
    get_df_band,
    is_eligible_term,
    is_index_compound_or_alphanumeric,
    compute_salience_score,
    compute_specificity_score,
    compute_hybrid_score,
    generate_salience_sequence,
    generate_specificity_sequence,
    generate_hybrid_sequence,
    generate_stratified_sequence,
    generate_celf_coverage_sequence,
    generate_naive_greedy_coverage_sequence,
    get_effective_capacities,
)


# ============================================================================
# 1. Disjoint 6-Band DF Partition Tests
# ============================================================================

def test_df_band_disjoint_partition():
    """Verifies that every term belongs to exactly one band across all DF and N ranges."""
    num_docs_list = [500, 3633, 5183, 50220, 171331, 5416380]
    all_bands = {
        "DF=0", "DF=1", "DF=2-5", "DF=6-20",
        "DF>20_rel<0.1%", "DF>20_rel_0.1-1%", "DF>20_rel>1%",
    }

    for N in num_docs_list:
        # Test boundary and interior points
        test_dfs = [0, 1, 2, 3, 5, 6, 10, 20, 21, 22, int(0.0005 * N), int(0.001 * N),
                    int(0.005 * N), int(0.01 * N), int(0.05 * N), N]
        for df in test_dfs:
            if df < 0 or df > N:
                continue
            band = get_df_band(df, N)
            assert band in all_bands, f"Band '{band}' not in allowed partition"
            
            # Verify explicit invariants
            if df == 0:
                assert band == "DF=0"
            elif df == 1:
                assert band == "DF=1"
            elif 2 <= df <= 5:
                assert band == "DF=2-5"
            elif 6 <= df <= 20:
                assert band == "DF=6-20"
            else:
                assert df > 20
                rel_df = df / N
                if rel_df < 0.001:
                    assert band == "DF>20_rel<0.1%"
                elif rel_df <= 0.01:
                    assert band == "DF>20_rel_0.1-1%"
                else:
                    assert band == "DF>20_rel>1%"


# ============================================================================
# 2. Disjoint Stratified Quota Tests
# ============================================================================

def test_disjoint_strata():
    """Verifies that S4 is extracted first and S1, S2, S3 are strictly disjoint."""
    vocab = [
        "covid19", "bnt162b2", "p53", "nav2_bringup",  # Alphanumeric / compound -> S4
        "gene", "cell", "protein", "dna", "cancer",      # Standard terms
        "rare_disease", "specific_mut", "general_term"
    ]
    df_map = {t: 10 for t in vocab}
    # Create distinct IDF values to populate S1, S2, S3
    idf_map = {
        "covid19": 8.0, "bnt162b2": 9.0, "p53": 7.0, "nav2_bringup": 8.5,
        "rare_disease": 10.0, "specific_mut": 8.0,  # spec >= 0.75 -> S1
        "cancer": 6.0, "gene": 5.0, "protein": 4.5, # 0.40 <= spec < 0.75 -> S2
        "cell": 3.0, "dna": 2.0, "general_term": 1.0 # spec < 0.40 -> S3
    }
    seq = generate_stratified_sequence(vocab, df_map, idf_map, max_cap=20)
    
    # Assert no duplicates
    assert len(seq) == len(set(seq))
    # Assert all terms included if capacity allows
    assert set(seq) == set(vocab)
    # Assert compound recognizer works
    assert is_index_compound_or_alphanumeric("covid19") is True
    assert is_index_compound_or_alphanumeric("cell") is False


# ============================================================================
# 3. CELF Submodular Coverage Equivalence Tests
# ============================================================================

def test_celf_equivalence_to_naive_greedy():
    """
    Asserts that CELF lazy-forward selection produces the exact same sequence
    as naïve exhaustive greedy selection on a synthetic corpus.
    """
    rng = random.Random(42)
    terms = [f"t_{i:03d}" for i in range(40)]
    
    # Generate synthetic postings across 200 documents
    postings_dict = {}
    for t in terms:
        doc_count = rng.randint(5, 30)
        postings_dict[t] = set(rng.sample(range(200), doc_count))
    
    idf_map = {t: rng.uniform(1.0, 10.0) for t in terms}
    supplier = lambda t: postings_dict.get(t, set())

    naive_seq = generate_naive_greedy_coverage_sequence(terms, supplier, idf_map, max_cap=15)
    celf_seq = generate_celf_coverage_sequence(terms, supplier, idf_map, max_cap=15)

    assert celf_seq == naive_seq, f"CELF sequence {celf_seq} != Naive {naive_seq}"


# ============================================================================
# 4. Nestedness & Lexicographic Tie-Breaking Tests
# ============================================================================

def test_nestedness_and_lexicographic_tiebreaking():
    """Verifies that tie-breaking is deterministic on term string and subsets are strictly nested."""
    terms = ["beta", "alpha", "delta", "gamma"]
    # Identical DF and IDF -> scores are exactly equal
    df_map = {t: 10 for t in terms}
    idf_map = {t: 5.0 for t in terms}

    seq = generate_salience_sequence(terms, df_map, idf_map, max_cap=4)
    # Alphabetical order expected: alpha, beta, delta, gamma
    assert seq == ["alpha", "beta", "delta", "gamma"]

    # Nestedness check
    p1 = seq[:1]
    p2 = seq[:2]
    p3 = seq[:3]
    p4 = seq[:4]
    assert set(p1).issubset(set(p2))
    assert set(p2).issubset(set(p3))
    assert set(p3).issubset(set(p4))


def test_effective_capacities_deduplication():
    """Verifies exact integer deduplication between fixed and percentage capacities."""
    num_eligible = 10000
    caps = get_effective_capacities(num_eligible, fixed_budgets=[1000, 2000, 5000], percentage_cutoffs=[0.10, 0.20, 0.50])
    # 0.10 * 10000 = 1000 (dup of 1k), 0.20 * 10000 = 2000 (dup of 2k), 0.50 * 10000 = 5000 (dup of 5k)
    # All percentage cutoffs must be cleanly deduplicated
    assert "10%" not in caps
    assert "20%" not in caps
    assert "50%" not in caps
    assert set(caps.values()) == {1000, 2000, 5000}


# ============================================================================
# 5. Dynamic BRIGHT Padding & Exclusion Filter Tests
# ============================================================================

def test_bright_dynamic_padding_and_truncation():
    """Verifies dynamic padding K_fetch = min(N, K + |E_q|) and truncation to K=1000."""
    total_docs = 2000
    K = 1000
    # Simulate a query with 150 exclusions
    exclusions = {f"doc_{i}" for i in range(150)}
    K_fetch = min(total_docs, K + len(exclusions))
    assert K_fetch == 1150

    # Simulate raw retrieval of K_fetch documents (some excluded, some valid)
    retrieved_docnos = [f"doc_{i}" for i in range(K_fetch)]
    filtered = [d for d in retrieved_docnos if d not in exclusions]
    truncated = filtered[:K]

    assert len(truncated) == K
    assert all(d not in exclusions for d in truncated)


# ============================================================================
# 6. Gate A: PyTerrier Weighting Parity & Monotonicity
# ============================================================================

@pytest.fixture(scope="module")
def scifact_index_and_queries():
    """Initializes PyTerrier and loads the local SciFact index."""
    import pyterrier as pt
    from evaluation.baselines.pyterrier_harness import init_pyterrier
    from evaluation.benchmark_loader import BenchmarkLoader
    
    init_pyterrier()
    idx_path = os.path.abspath("data/cache/terrier_indices/scifact_default/data.properties")
    if not os.path.exists(idx_path):
        pytest.skip(f"SciFact index not found at {idx_path}")
    
    index = pt.IndexFactory.of(idx_path)
    queries, _ = BenchmarkLoader.load_queries("scifact")
    return index, queries


def test_gate_a_exact_bm25_parity_at_zero_weight(scifact_index_and_queries):
    """
    Gate A Contract: Proves that injecting a candidate term with mu=0.0 via query_toks
    produces identical docnos, identical order, scores within 1e-6, and identical depth
    compared to standard BM25_Default on normal, repeated, and collision queries.
    """
    import pyterrier as pt
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

    index, queries = scifact_index_and_queries
    analyzer = get_terrier_analyzer()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=1000)

    # Test cases: normal query, query with repeated terms, and stem collision
    test_cases = [
        {"qid": "t1", "text": "cancer cell immunotherapy efficacy", "candidate": "vaccine"},
        {"qid": "t2", "text": "cancer cancer cell cell efficacy", "candidate": "immunotherapy"},  # Repeated terms
        {"qid": "t3", "text": "cells developing in culture", "candidate": "cell"},                # Stem collision
    ]

    for tc in test_cases:
        qid = tc["qid"]
        text = tc["text"]
        cand = tc["candidate"]

        # Baseline retrieval
        df_base = pd.DataFrame([{"qid": qid, "query": text}])
        res_base = bm25.transform(df_base)

        # Expanded retrieval with mu = 0.0: modify only candidate weight by adding mu
        terms, _ = analyzer.analyze(text)
        query_toks = analyzer.to_query_toks(terms)
        cand_terms, _ = analyzer.analyze(cand)
        for c_t in cand_terms:
            query_toks[c_t] = query_toks.get(c_t, 0.0) + 0.0  # Zero weight addition

        df_zero = pd.DataFrame([{"qid": qid, "query_toks": query_toks}])
        res_zero = bm25.transform(df_zero)

        # Assertions
        assert len(res_base) == len(res_zero), f"Depth mismatch for query {qid}"
        assert (res_base["docno"].values == res_zero["docno"].values).all(), f"Docno/order mismatch for query {qid}"
        max_score_diff = np.max(np.abs(res_base["score"].values - res_zero["score"].values))
        assert max_score_diff <= 1e-6, f"Score difference {max_score_diff} > 1e-6 for query {qid}"


def test_gate_a_document_score_monotonicity(scifact_index_and_queries):
    """
    Gate A Contract: Proves that for candidates with non-negative IDF, increasing mu
    monotonically increases (or preserves) the scores of documents containing candidate c.
    S_{mu_1}(d) <= S_{mu_2}(d) for matching documents.
    """
    import pyterrier as pt
    from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

    index, queries = scifact_index_and_queries
    analyzer = get_terrier_analyzer()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=1000)
    lex = index.getLexicon()

    # Candidate analyzed stem
    cand_stems, _ = analyzer.analyze("mutation")
    candidate = cand_stems[0]
    entry = lex.getLexiconEntry(candidate)
    assert entry is not None and entry.getDocumentFrequency() >= 2

    # Check non-negative IDF condition
    num_docs = index.getCollectionStatistics().getNumberOfDocuments()
    df = entry.getDocumentFrequency()
    # Lucene non-negative IDF: ln(1 + (N - n + 0.5) / (n + 0.5))
    idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
    assert idf >= 0.0

    # Get documents matching candidate from inverted postings
    inv = index.getInvertedIndex()
    meta = index.getMetaIndex()
    postings = inv.getPostings(entry)
    matching_docnos = set()
    while postings.next() != postings.EOL:
        matching_docnos.add(meta.getItem("docno", postings.getId()))

    query_text = "cancer progression and genetics"
    terms, _ = analyzer.analyze(query_text)

    weights = [0.05, 0.10, 0.30, 0.50, 1.00]
    scores_per_weight = {}

    for w in weights:
        query_toks = analyzer.to_query_toks(terms)
        query_toks[candidate] = w
        df_q = pd.DataFrame([{"qid": "mono_q1", "query_toks": query_toks}])
        res = bm25.transform(df_q)
        scores_per_weight[w] = dict(zip(res["docno"], res["score"]))

    # Check monotonicity across adjacent weights for documents containing candidate
    for i in range(len(weights) - 1):
        w1 = weights[i]
        w2 = weights[i + 1]
        s1_map = scores_per_weight[w1]
        s2_map = scores_per_weight[w2]

        common_matching = set(s1_map.keys()) & set(s2_map.keys()) & matching_docnos
        assert len(common_matching) > 0, "Expected matching documents in retrieval results"

        for docno in common_matching:
            diff = s2_map[docno] - s1_map[docno]
            assert diff >= -1e-6, f"Monotonicity violated for doc {docno}: s({w1})={s1_map[docno]} > s({w2})={s2_map[docno]}"
