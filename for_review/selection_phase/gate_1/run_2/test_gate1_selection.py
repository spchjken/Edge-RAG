"""
CRVE/tests/test_gate1_selection.py

Focused unit test suite verifying Phase 2.1a Gate 1 Candidate Selection invariants:
1. Canonical pool properties, sizes, and SHA-256 hashes.
2. Fast O(1) pool_set membership and query-term exclusion P_q.
3. Mutually exclusive and exhaustive action partition (P_H + P_N + P_M = 1.0).
4. Raw vs Recall-Safe cutoff entry opportunity contract.
5. RRF tie-breaking (-S_RRF, best_individual_rank, term).
6. Frozen configuration schema and hash validation.
"""

import os
import sys
import json
import yaml
import pytest
import numpy as np
import pandas as pd

# Ensure CRVE and CRVE/src are on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from crve.selection.gate1_proposers import RRFHybridProposer


def test_canonical_pool_properties():
    """Validates canonical pool artifacts, expected sizes, and SHA-256 hashes."""
    config_path = "CRVE/configs/gate1_phase2_1a.yaml"
    assert os.path.exists(config_path), f"Missing config: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    expected_sizes = config["pool_spec"]["sizes"]
    expected_hashes = config["pool_spec"]["hashes"]

    assert expected_sizes["scifact"] == 5595
    assert expected_sizes["bright_aops"] == 10000
    assert expected_sizes["nfcorpus"] == 7783
    assert expected_sizes["trec_covid"] == 10000

    for ds, target_size in expected_sizes.items():
        artifact_path = f"data/cache/canonical_pools/{ds}_canonical_pool.json"
        assert os.path.exists(artifact_path), f"Missing canonical pool artifact: {artifact_path}"
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["terms"]) == target_size
        assert data["sha256"] == expected_hashes[ds]
        # Verify terms are unique and non-empty
        assert len(set(data["terms"])) == target_size
        assert all(len(t) >= 2 for t in data["terms"])


def test_pool_set_membership_invariance():
    """Verifies that pool_set membership check matches list check and excludes query terms."""
    pool_terms = ["robot", "ai", "learning", "neural", "vision"]
    pool_set = set(pool_terms)
    query_terms = {"robot", "deep"}

    # P_q = P \ query_terms
    candidates = ["robot", "ai", "vision", "unknown", "deep"]
    clean_cands = [t for t in candidates if t in pool_set and t not in query_terms]

    assert clean_cands == ["ai", "vision"]
    assert "robot" not in clean_cands
    assert "unknown" not in clean_cands


def test_action_partition_exclusivity():
    """
    Validates that every evaluated action belongs to exactly one partition:
    Helpful, Harmful, or Neutral/Marginal (P_H + P_N + P_M = 1.0),
    and waste is an overlapping diagnostic.
    """
    delta = 0.005
    epsilon = 0.001

    test_actions = [
        # (d_ndcg10, net_rel_k1000, expected_class, expected_waste)
        (0.010, 2, "helpful", False),
        (0.005, 0, "helpful", False),
        (0.010, -1, "harmful", False),  # gain high but net recall negative -> harmful
        (-0.002, 1, "harmful", False),  # ndcg loss > epsilon -> harmful, but net_k > 0 so not waste
        (-0.005, -1, "harmful", True),  # both negative -> harmful & waste
        (0.002, 1, "neutral", False),   # gain between 0 and delta, net recall >= 0 -> neutral
        (0.000, 0, "neutral", True),    # zero gain, zero net -> neutral & waste
        (-0.0005, 0, "neutral", True),  # tiny drop < epsilon, net recall >= 0 -> neutral & waste
    ]

    for d_ndcg, net_k, exp_class, exp_waste in test_actions:
        is_helpful = (d_ndcg >= delta) and (net_k >= 0)
        is_harmful = (d_ndcg < -epsilon) or (net_k < 0)
        is_neutral = (not is_helpful) and (not is_harmful)
        is_waste = (d_ndcg <= 0.0) and (net_k <= 0)

        # Mutually exclusive and exhaustive
        classes = [is_helpful, is_harmful, is_neutral]
        assert sum(classes) == 1, f"Partition failed for d_ndcg={d_ndcg}, net_k={net_k}: {classes}"

        assigned = "helpful" if is_helpful else ("harmful" if is_harmful else "neutral")
        assert assigned == exp_class, f"Expected {exp_class}, got {assigned} for d_ndcg={d_ndcg}, net_k={net_k}"
        assert is_waste == exp_waste


def test_cutoff_entries_contract():
    """Tests raw vs recall-safe cutoff entry logic for documents outside baseline top-K."""
    k = 100
    epsilon = 0.001

    # Case 1: Document was already in top-K at baseline -> cannot enter
    base_rank_in = 50
    exp_rank_in = 20
    assert not (base_rank_in > k)

    # Case 2: Document was outside top-K, enters under a safe action
    base_rank_out = 150
    exp_rank_enter = 80
    d_ndcg10_safe = 0.02
    net_rel_k = 1

    raw_entry = (base_rank_out > k) and (exp_rank_enter <= k)
    safe_entry = raw_entry and (d_ndcg10_safe >= -epsilon) and (net_rel_k >= 1)
    assert raw_entry is True
    assert safe_entry is True

    # Case 3: Document enters, but action is unsafe (large NDCG drop)
    d_ndcg10_unsafe = -0.05
    safe_entry_unsafe = raw_entry and (d_ndcg10_unsafe >= -epsilon) and (net_rel_k >= 1)
    assert safe_entry_unsafe is False


def test_rrf_tie_breaking():
    """Verifies RRF tie-breaking rule: (-S_RRF, best_individual_rank, term)."""
    proposer = RRFHybridProposer(k=60)
    # Channel rankings:
    # ch1: term_a (rank 1), term_b (rank 2)
    # ch2: term_b (rank 1), term_a (rank 2)
    # Both term_a and term_b have identical RRF score: 1/(60+1) + 1/(60+2)
    # Both have best_individual_rank = 1
    # Tie broken alphabetically: term_a before term_b
    channel_rankings = {
        "ch1": [("term_b", 0.9), ("term_a", 0.8)],
        "ch2": [("term_a", 0.9), ("term_b", 0.8)],
    }
    fused = proposer.fuse(channel_rankings, top_l=2)
    assert [t for t, _ in fused] == ["term_a", "term_b"]


def test_frozen_config_validation():
    """Verifies frozen config loads properly and contains all required sections."""
    config_path = "CRVE/configs/gate1_phase2_1a.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "version" in cfg
    assert "seed" in cfg and cfg["seed"] == 42
    assert "thresholds" in cfg
    assert "pool_spec" in cfg
    assert "query_manifests" in cfg
    assert len(cfg["query_manifests"]["dev_200_qids"]) == 4
    for ds, qids in cfg["query_manifests"]["dev_200_qids"].items():
        assert len(qids) == 50
    for ds, qids in cfg["query_manifests"]["probe_40_qids"].items():
        assert len(qids) == 10
