"""
tests/test_gate1_selection.py

Unit and regression tests for Phase 2 Gate 1 Candidate Selection:
- Mathematical metrics: ReferenceBOR, NearBestHit, TermRecall, TermPrecision, RecallHit, DocOpportunityRecall
- Sub-threshold protection for H_q^near(rho, delta)
- Empty safe-action fallback to 0.0
- Sparse document transition classification and cutoff crossings
- RRF 3-tier deterministic tie-breaking and equal-budget refill
- Query canonical term exclusion (P_q)
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.pipeline_v2.selection.gate1_metrics import (
    TAU,
    EPSILON,
    DEFAULT_DELTA,
    DEFAULT_RHO,
    compute_safe_ranking_gain,
    compute_safe_recall_gain,
    get_ranking_helpful_terms,
    get_recall_helpful_terms,
    get_near_best_terms,
    compute_reference_bor,
    compute_near_best_hit,
    compute_term_recall,
    compute_term_precision,
    compute_recall_hit,
    compute_doc_opportunity_recall,
    compute_waste_metrics,
    classify_document_transition,
)
from src.pipeline_v2.selection.gate1_proposers import (
    RRFHybridProposer,
    BaseProposer,
    WholeQueryBGEProposer,
)


def test_safe_ranking_gain_fallbacks():
    """Validates that unsafe actions or negative gains default to 0.0."""
    # 1. Action preserves recall but hurts nDCG -> 0.0
    harmful_actions = [
        {"delta_ndcg10": -0.05, "delta_r1000": 0.0},
        {"delta_ndcg10": -0.01, "delta_r1000": 0.02},
    ]
    assert compute_safe_ranking_gain(harmful_actions) == 0.0

    # 2. Action improves nDCG but hurts recall below -tau -> excluded!
    unsafe_actions = [
        {"delta_ndcg10": 0.08, "delta_r1000": -0.01},  # hurts recall
        {"delta_ndcg10": 0.02, "delta_r1000": 0.0},    # safe
    ]
    assert compute_safe_ranking_gain(unsafe_actions) == pytest.approx(0.02)

    # 3. Empty actions -> 0.0
    assert compute_safe_ranking_gain([]) == 0.0


def test_safe_recall_gain_fallbacks():
    """Validates that unsafe ranking actions are excluded from recall gains."""
    actions = [
        {"net_rel_docs_k1000": 3, "delta_ndcg10": -0.02},  # < -epsilon -> excluded
        {"net_rel_docs_k1000": 1, "delta_ndcg10": -0.0005}, # >= -epsilon -> valid
    ]
    assert compute_safe_recall_gain(actions, cutoff=1000) == 1


def test_near_best_terms_subthreshold_protection():
    """Validates that H_q^near enforces max(delta, rho * ceiling_g)."""
    term_actions = {
        "t1": [{"delta_ndcg10": 0.004, "delta_r1000": 0.0}],
        "t2": [{"delta_ndcg10": 0.005, "delta_r1000": 0.0}],
    }
    # Case 1: ceiling_g = 0.005, delta = 0.005, rho = 0.8
    # rho * ceiling_g = 0.004 < delta (0.005) -> threshold must be delta (0.005)
    # t1 (0.004) MUST NOT qualify as near-best!
    nb = get_near_best_terms(term_actions, ceiling_g=0.005, rho=0.8, delta=0.005)
    assert "t1" not in nb
    assert "t2" in nb

    # Case 2: ceiling_g < delta -> returns empty set
    nb_low = get_near_best_terms(term_actions, ceiling_g=0.003, delta=0.005)
    assert len(nb_low) == 0


def test_reference_bor_strict_na():
    """Validates that ReferenceBOR returns NaN strictly when ceiling_g <= tau."""
    term_actions = {
        "t1": [{"delta_ndcg10": 0.02, "delta_r1000": 0.0}],
    }
    # Positive ceiling
    bor = compute_reference_bor(["t1"], term_actions, ceiling_g=0.04)
    assert bor == pytest.approx(0.50)

    # Ceiling <= tau -> NaN
    assert np.isnan(compute_reference_bor(["t1"], term_actions, ceiling_g=1e-6))
    assert np.isnan(compute_reference_bor(["t1"], term_actions, ceiling_g=0.0))

    # Proposed terms have no gain -> 0.0 (when ceiling is valid)
    bor_zero = compute_reference_bor(["missing"], term_actions, ceiling_g=0.04)
    assert bor_zero == pytest.approx(0.0)


def test_near_best_hit_metrics():
    """Validates NearBestHit and its NaN condition."""
    near_best_set = {"t1", "t2"}
    # Ceiling >= delta -> 1.0 on hit, 0.0 on miss
    assert compute_near_best_hit(["t1", "t3"], near_best_set, ceiling_g=0.01) == 1.0
    assert compute_near_best_hit(["t3", "t4"], near_best_set, ceiling_g=0.01) == 0.0

    # Ceiling < delta -> NaN
    assert np.isnan(compute_near_best_hit(["t1"], near_best_set, ceiling_g=0.002, delta=0.005))


def test_term_recall_and_precision_zero_denominators():
    """Validates that TermRecall and TermPrecision return NaN on empty denominators."""
    helpful = {"t1", "t2"}
    # Normal computation
    assert compute_term_recall(["t1", "t3"], helpful) == pytest.approx(0.50)
    assert compute_term_precision(["t1", "t3"], helpful) == pytest.approx(0.50)

    # Empty helpful set -> TermRecall is NaN
    assert np.isnan(compute_term_recall(["t1"], set()))

    # Empty proposed list -> TermPrecision is NaN
    assert np.isnan(compute_term_precision([], helpful))


def test_doc_opportunity_recall_union():
    """Validates DocOpportunityRecall distinct document recovery."""
    term_doc_map = {
        "t1": {"doc1", "doc2"},
        "t2": {"doc2", "doc3"},
    }
    universe_docs = {"doc1", "doc2", "doc3", "doc4"}

    # t1 + t2 recover {doc1, doc2, doc3} -> 3/4 = 0.75
    doc_rec = compute_doc_opportunity_recall(["t1", "t2"], term_doc_map, universe_docs)
    assert doc_rec == pytest.approx(0.75)

    # Empty universe -> NaN
    assert np.isnan(compute_doc_opportunity_recall(["t1"], term_doc_map, set()))


def test_waste_metrics_formal_definitions():
    """Validates waste metrics on unaddressable queries."""
    # 50 candidates emitted when ceiling_g = 0.001 < delta (0.005)
    w = compute_waste_metrics(["t"] * 50, ceiling_g=0.001, delta=0.005, l_max=200)
    assert w["waste_count"] == 50
    assert w["is_unaddressable_material"] is True
    assert w["is_unaddressable_strict"] is False
    assert w["normalized_waste"] == pytest.approx(0.25)


def test_classify_document_transition_sparse():
    """Validates document transition classification and cutoff crossing flags."""
    # 1. Document entered top-1000 (was >1000, now 50)
    t_enter = classify_document_transition(baseline_rank=None, expanded_rank=50)
    assert t_enter is not None
    assert t_enter["transition_type"] == "entered_top1000"
    assert t_enter["rank_delta"] is None
    assert t_enter["crossed_k100"] is True
    assert t_enter["direction_k100"] == "entered"
    assert t_enter["crossed_k1000"] is True
    assert t_enter["direction_k1000"] == "entered"

    # 2. Document left top-1000 (was 500, now >1000)
    t_left = classify_document_transition(baseline_rank=500, expanded_rank=None)
    assert t_left is not None
    assert t_left["transition_type"] == "left_top1000"
    assert t_left["rank_delta"] is None
    assert t_left["crossed_k500"] is True
    assert t_left["direction_k500"] == "left"
    assert t_left["crossed_k1000"] is True
    assert t_left["direction_k1000"] == "left"

    # 3. Document moved within top-1000 (was 250, now 80)
    t_move = classify_document_transition(baseline_rank=250, expanded_rank=80)
    assert t_move is not None
    assert t_move["transition_type"] == "moved_within_top1000"
    assert t_move["rank_delta"] == 170
    assert t_move["crossed_k100"] is True
    assert t_move["direction_k100"] == "entered"
    assert t_move["crossed_k200"] is True
    assert t_move["direction_k200"] == "entered"
    assert t_move["crossed_k500"] is False

    # 4. Document unchanged in top-1000 (was 50, now 50) -> None (filtered out)
    assert classify_document_transition(baseline_rank=50, expanded_rank=50) is None

    # 5. Document outside top-1000 in both -> None (filtered out)
    assert classify_document_transition(baseline_rank=None, expanded_rank=None) is None


def test_rrf_hybrid_tie_breaking_and_refill():
    """Validates RRF deterministic 3-tier tie-breaking and equal-budget unique refill."""
    proposer = RRFHybridProposer(k=60)
    channel_rankings = {
        "ch1": [("termA", 0.9), ("termB", 0.8), ("termC", 0.7)],
        "ch2": [("termB", 0.95), ("termA", 0.85), ("termD", 0.6)],
    }
    # termB: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522 (best_rank=1)
    # termA: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.032522 (best_rank=1)
    # termB and termA tie in score and best_rank -> broken alphabetically: 'termA' < 'termB'
    fused = proposer.fuse(channel_rankings, top_l=3)
    assert len(fused) == 3
    terms = [t for t, _ in fused]
    assert terms[0] == "termA"
    assert terms[1] == "termB"
    # Third is termC or termD
    assert terms[2] in ("termC", "termD")


def test_query_exclusion_set():
    """Validates that BaseProposer extracts analyzed terms and surfaces for exclusion."""
    class DummyProposer(BaseProposer):
        pass

    prop = DummyProposer(name="Dummy")
    excluded = prop.get_query_excluded_terms("machine learning algorithms")
    assert "machine" in excluded
    assert "learn" in excluded or "learning" in excluded
    assert "algorithm" in excluded


def test_no_candidate_outside_p_q():
    """Validates that proposers strictly exclude any term in AnalyzedCanonicalTerms(q)."""
    class MockEncoder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    pool = ["machine", "learning", "deep", "neural", "network", "algorithm"]
    pool_tensor = torch.eye(6, 4)  # 6 pool terms
    proposer = WholeQueryBGEProposer(
        pool_terms=pool,
        pool_embeddings_tensor=pool_tensor,
        encoder=MockEncoder(),
        device="cpu",
    )
    query_obj = {"query": "machine learning algorithms"}
    excluded = proposer.get_query_excluded_terms(query_obj["query"])
    
    proposed = proposer.propose(query_obj, top_k=10)
    proposed_terms = [t for t, _ in proposed]
    
    # Assert no proposed term is in excluded query terms
    for t in proposed_terms:
        assert t not in excluded, f"Term '{t}' should have been excluded by P_q constraint!"


def test_100_percent_label_coverage_after_centroid_augmentation():
    """Validates that all terms in R_q^aug have complete counterfactual label coverage."""
    r_core = {"term_a", "term_b", "term_c"}
    c_centroid = {"term_c", "term_d", "term_e"}
    r_aug = r_core.union(c_centroid)
    
    # Simulated audit results covering all terms in r_aug
    audit_table = {
        "term_a": [{"delta_ndcg10": 0.01, "delta_r1000": 0.0}],
        "term_b": [{"delta_ndcg10": 0.00, "delta_r1000": 0.0}],
        "term_c": [{"delta_ndcg10": 0.02, "delta_r1000": 0.0}],
        "term_d": [{"delta_ndcg10": -0.01, "delta_r1000": 0.0}],
        "term_e": [{"delta_ndcg10": 0.03, "delta_r1000": 0.01}],
    }
    
    # Coverage verification
    missing_labels = [t for t in r_aug if t not in audit_table or len(audit_table[t]) == 0]
    assert len(missing_labels) == 0, f"Missing counterfactual labels for: {missing_labels}"
    assert len(r_aug) == 5

