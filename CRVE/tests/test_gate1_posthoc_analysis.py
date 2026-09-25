#!/usr/bin/env python3
"""
test_gate1_posthoc_analysis.py - Unit test suite for Gate 1 Post-Hoc Lexical Analysis
"""

import os
import sys
import tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from compile_gate1_posthoc_analysis import (
    paired_stratified_bootstrap,
    Gate1PostHocAnalyzer,
    format_pct,
)


def test_paired_stratified_bootstrap_identical():
    """When two policies have identical per-query values, diff should be exactly zero."""
    data_a = {
        "scifact": {"q1": 0.5, "q2": 0.7, "q3": 0.9},
        "nfcorpus": {"q4": 0.2, "q5": 0.4},
    }
    data_b = {
        "scifact": {"q1": 0.5, "q2": 0.7, "q3": 0.9},
        "nfcorpus": {"q4": 0.2, "q5": 0.4},
    }
    pt_est, low, high = paired_stratified_bootstrap(data_a, data_b, b_resamples=200, seed=42)
    assert abs(pt_est) < 1e-9
    assert abs(low) < 1e-9
    assert abs(high) < 1e-9


def test_paired_stratified_bootstrap_single_policy():
    """Single policy bootstrap point estimate should equal unweighted corpus-macro mean."""
    data = {
        "scifact": {"q1": 0.6, "q2": 0.8},  # mean 0.7
        "nfcorpus": {"q3": 0.3, "q4": 0.5},  # mean 0.4
    }
    # Expected macro mean = (0.7 + 0.4) / 2 = 0.55
    pt_est, low, high = paired_stratified_bootstrap(data, b_resamples=500, seed=42)
    assert abs(pt_est - 0.55) < 1e-9
    assert low <= pt_est <= high
    assert 0.3 <= low <= 0.8


def test_rrf_lexical2_logic():
    """Test RRF_Lexical2 fusion, rank depth cap 500, and deterministic 3-tier tie breaking."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.query_term_actions = {
        ("scifact", "q1"): {
            # Term A: in PPMI at 10, in Sparse at 10 -> score = 1/70 + 1/70 = 2/70 ~ 0.02857, best_rank = 10
            "term_a": [{"ppmi_sidecar_rank": 10, "sparse_lex_rank": 10}],
            # Term B: in PPMI at 5 -> score = 1/65 ~ 0.01538, best_rank = 5
            "term_b": [{"ppmi_sidecar_rank": 5, "sparse_lex_rank": None}],
            # Term C: in Sparse at 5 -> score = 1/65 ~ 0.01538, best_rank = 5
            # Same score and best_rank as term_b; tie break alphabetically: "term_b" < "term_c"
            "term_c": [{"ppmi_sidecar_rank": None, "sparse_lex_rank": 5}],
            # Term D: in PPMI at 501 (exceeds depth 500 cap) -> score = 0
            "term_d": [{"ppmi_sidecar_rank": 501, "sparse_lex_rank": None}],
        }
    }

    terms = analyzer.get_query_terms_for_policy("scifact", "q1", "RRF_Lexical2", budget_l=10)
    # Expected ranking: term_a (highest score), then term_b (tie break over term_c), then term_c
    # term_d should not be included
    assert terms == ["term_a", "term_b", "term_c"]


def test_lexical_union_logic():
    """Test deduplicated union of PPMI top-M and Sparse top-M with symmetric allocation."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.query_term_actions = {
        ("scifact", "q1"): {
            "p1": [{"ppmi_sidecar_rank": 1, "sparse_lex_rank": 50}],
            "p2": [{"ppmi_sidecar_rank": 2, "sparse_lex_rank": None}],
            "overlap": [{"ppmi_sidecar_rank": 3, "sparse_lex_rank": 1}],
            "s1": [{"ppmi_sidecar_rank": None, "sparse_lex_rank": 2}],
            "s2": [{"ppmi_sidecar_rank": None, "sparse_lex_rank": 3}],
            "out_of_bounds": [{"ppmi_sidecar_rank": 10, "sparse_lex_rank": 10}],
        }
    }

    # With union_m=3 and budget_l=10:
    # PPMI top-3: p1 (1), p2 (2), overlap (3)
    # Sparse top-3: overlap (1), s1 (2), s2 (3)
    # Deduplicated union: [p1, p2, overlap, s1, s2]
    terms = analyzer.get_query_terms_for_policy(
        "scifact", "q1", "LexicalUnion", budget_l=10, union_m=3
    )
    assert terms == ["p1", "p2", "overlap", "s1", "s2"]

    # Test truncation to budget_l=4
    terms_trunc = analyzer.get_query_terms_for_policy(
        "scifact", "q1", "LexicalUnion", budget_l=4, union_m=3
    )
    assert terms_trunc == ["p1", "p2", "overlap", "s1"]


def test_reproduction_gate_mismatch_raises_fatal():
    """Verifies that the reproduction gate raises ValueError if computed metrics diverge from Table 2."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.ref_table2_path = None
    analyzer.audit_path = "results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet"

    # Create dummy reference Table 2 CSV
    df_dummy = pd.DataFrame([{
        "Channel": "PPMISidecar",
        "Corpus-Macro TermRecall@L": "99.9%",  # Intentionally diverging from reality
        "Corpus-Macro TermPrecision@L": "6.2%",
        "Corpus-Macro NearBestHit@L": "50.6%",
        "Corpus-Macro ReferenceBOR@L": "69.7%",
        "Corpus-Macro RecallHit@1000": "88.5%",
        "Corpus-Macro RawDocOppRecall@1000": "72.0%",
        "Corpus-Macro SafeDocOppRecall@1000": "69.4%",
    }])

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        df_dummy.to_csv(f.name, index=False)
        dummy_csv_path = f.name

    try:
        analyzer.ref_table2_path = dummy_csv_path
        # Mock evaluate_single_channel_policy and aggregate_macro_metrics
        analyzer.evaluate_single_channel_policy = lambda *args, **kwargs: {}
        analyzer.aggregate_macro_metrics = lambda *args, **kwargs: {
            "term_recall": 0.205,  # 20.5% != 99.9%
            "term_precision": 0.062,
            "near_best_hit_90": 0.506,
            "reference_bor": 0.697,
            "recall_hit_1000": 0.885,
            "raw_doc_opp_recall": 0.720,
            "safe_doc_opp_recall": 0.694,
        }
        with pytest.raises(ValueError, match="Reproduction mismatch"):
            analyzer.run_reproduction_gate()
    finally:
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)


def test_output_cardinality_matched_runtime_invariant():
    """Verifies that evaluate_output_matched_extended_policy enforces len(ext) == len(union)."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.available_datasets = ["scifact"]
    analyzer.query_term_actions = {
        ("scifact", "q1"): {
            "p1": [{"ppmi_sidecar_rank": 1, "sparse_lex_rank": None, "rrf_ext_rank": 1}],
            "p2": [{"ppmi_sidecar_rank": 2, "sparse_lex_rank": None, "rrf_ext_rank": 2}],
            "s1": [{"ppmi_sidecar_rank": None, "sparse_lex_rank": 1, "rrf_ext_rank": 3}],
        }
    }
    analyzer.query_total_raw_docs = defaultdict(set)
    analyzer.query_total_safe_docs = defaultdict(set)
    analyzer.cutoff_raw_docs = defaultdict(set)
    analyzer.cutoff_safe_docs = defaultdict(set)
    analyzer.delta = 0.005
    analyzer.frozen_rho = 0.90
    analyzer.supplemental_rho = 0.80

    # Union with m=2 gives p1, p2, s1 -> len = 3
    # Extended with k=3 gives p1, p2, s1 -> len = 3
    res = analyzer.evaluate_output_matched_extended_policy(
        union_policy_name="LexicalUnion", union_budget_l=400, union_m=2
    )
    assert res["scifact"]["q1"]["cand_count"] == 3

    # Now simulate a query where Extended has fewer candidates than Union
    analyzer.query_term_actions[("scifact", "q2")] = {
        "p1": [{"ppmi_sidecar_rank": 1, "sparse_lex_rank": None, "rrf_ext_rank": 1}],
        "s1": [{"ppmi_sidecar_rank": None, "sparse_lex_rank": 1, "rrf_ext_rank": None}],  # Not in Extended
    }
    # Union gives 2 candidates (p1, s1), but Extended only has 1 (p1)
    with pytest.raises(AssertionError, match="Runtime Invariant Violation"):
        analyzer.evaluate_output_matched_extended_policy(
            union_policy_name="LexicalUnion", union_budget_l=400, union_m=2
        )


def test_ppmi_micro_precision_algebraic_identity():
    """Verifies that mean(|C_q \cap H_q|) = mean(|C_q|) * Precision_micro holds algebraically."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.delta = 0.005
    analyzer.query_term_actions = {
        # Query 1: 100 candidates, 10 helpful -> Precision = 10%
        ("scifact", "q1"): {
            f"t_{i}": [{"ppmi_sidecar_rank": i + 1, "delta_ndcg10": 0.05 if i < 10 else 0.0, "net_rel_docs_k10": 0}]
            for i in range(100)
        },
        # Query 2: 200 candidates, 10 helpful -> Precision = 5%
        ("scifact", "q2"): {
            f"t_{i}": [{"ppmi_sidecar_rank": i + 1, "delta_ndcg10": 0.05 if i < 10 else 0.0, "net_rel_docs_k10": 0}]
            for i in range(200)
        },
    }

    stats = analyzer.analyze_ppmi_emissions()
    # Query 1: cands = 100, helpful = 10
    # Query 2: cands = 200, helpful = 10
    # Mean cands = 150, Mean helpful = 10
    # Micro precision = (10 + 10) / (100 + 200) = 20 / 300 = 1/15
    # Product: 150 * (1/15) = 10 == Mean helpful!
    assert stats["mean_count"] == 150.0
    assert stats["mean_helpful_unconditioned"] == 10.0
    assert abs(stats["micro_precision_unconditioned"] - (1.0 / 15.0)) < 1e-9
    assert abs(stats["mean_count"] * stats["micro_precision_unconditioned"] - stats["mean_helpful_unconditioned"]) < 1e-9


def test_frozen_rho_invariance():
    """Verifies that primary NearBestHit is locked to rho=0.90, and supplemental uses supplemental_rho."""
    analyzer = Gate1PostHocAnalyzer.__new__(Gate1PostHocAnalyzer)
    analyzer.delta = 0.005
    analyzer.frozen_rho = 0.90
    analyzer.supplemental_rho = 0.80
    analyzer.query_total_raw_docs = defaultdict(set)
    analyzer.query_total_safe_docs = defaultdict(set)
    analyzer.cutoff_raw_docs = defaultdict(set)
    analyzer.cutoff_safe_docs = defaultdict(set)

    # Let best candidate have gain 0.10 -> g* = 0.10
    # A candidate with gain 0.085 is in supplemental (0.085 >= 0.80 * 0.10 = 0.080)
    # but NOT in primary (0.085 < 0.90 * 0.10 = 0.090)
    analyzer.query_term_actions = {
        ("scifact", "q1"): {
            "best_term": [{"delta_ndcg10": 0.10, "net_rel_docs_k10": 0}],
            "mid_term": [{"delta_ndcg10": 0.085, "net_rel_docs_k10": 0}],
            "low_term": [{"delta_ndcg10": 0.01, "net_rel_docs_k10": 0}],
        }
    }

    # Proposing only "mid_term"
    res = analyzer.evaluate_query("scifact", "q1", ["mid_term"])
    assert res["near_best_hit_90"] == 0.0  # Misses primary rho=0.90
    assert res["near_best_hit_supp"] == 1.0  # Hits supplemental rho=0.80
