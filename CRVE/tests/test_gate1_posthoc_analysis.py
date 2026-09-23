#!/usr/bin/env python3
"""
test_gate1_posthoc_analysis.py - Unit test suite for Gate 1 Post-Hoc Lexical Analysis
"""

import os
import sys
import tempfile
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
    """Test deduplicated union of PPMI top-M and Sparse top-M."""
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
            "near_best_hit": 0.506,
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
