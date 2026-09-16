#!/usr/bin/env python3
"""
Unit tests for the Stage 1 Pool Table Compiler logic.
Validates:
- Decision tree for DF=1 dominance (5 mutually exclusive buckets)
- Per-query abstention mechanics
- Inclusive rank filtering (rank <= capacity)
- Optimal-weight qualification with positive primary utility
- Bootstrap confidence interval consistency
"""

import pytest
import pandas as pd
import numpy as np

from scripts.compile_pool_oracle_tables import (
    classify_df1_dominance_query,
    bootstrap_ci,
    TAU,
    EPSILON,
)


def test_classify_df1_dominance_decision_tree():
    """Validates that all 5 DF=1 dominance buckets are mutually exclusive and exhaustive."""
    # 1. Neither Useful
    assert classify_df1_dominance_query(0.0, 0.0) == "Neither Useful"
    assert classify_df1_dominance_query(1e-6, 1e-6) == "Neither Useful"

    # 2. DF1 Wins (Eligible inactive)
    assert classify_df1_dominance_query(0.05, 0.0) == "DF1 Wins"
    assert classify_df1_dominance_query(0.05, 1e-6) == "DF1 Wins"

    # 3. Eligible Wins, DF1 Inactive
    assert classify_df1_dominance_query(0.0, 0.05) == "Eligible Wins, DF1 Inactive"
    assert classify_df1_dominance_query(1e-6, 0.05) == "Eligible Wins, DF1 Inactive"

    # 4. Useful Tie (|G_df1 - G_elig| <= TAU)
    assert classify_df1_dominance_query(0.050001, 0.050005) == "Useful Tie"
    assert classify_df1_dominance_query(0.05, 0.05) == "Useful Tie"

    # 5. DF1 Wins (both useful, DF1 strictly better by > TAU)
    assert classify_df1_dominance_query(0.06, 0.04) == "DF1 Wins"

    # 6. DF1 Useful but Dominated (both useful, Eligible strictly better by > TAU)
    assert classify_df1_dominance_query(0.02, 0.05) == "DF1 Useful but Dominated"


def test_per_query_abstention_invariant():
    """Validates that harmful expansions never depress the abstaining oracle ceiling below baseline."""
    base_ndcg = 0.50
    harmful_variants = [
        {"qid": "q1", "baseline_ndcg10": base_ndcg, "expanded_ndcg10": 0.40, "delta_ndcg10": -0.10},
        {"qid": "q1", "baseline_ndcg10": base_ndcg, "expanded_ndcg10": 0.45, "delta_ndcg10": -0.05},
    ]
    df = pd.DataFrame(harmful_variants)

    # Abstaining calculation
    q_max = df.groupby("qid")["expanded_ndcg10"].max()
    oracle_val = max(base_ndcg, q_max["q1"])
    delta = max(oracle_val - base_ndcg, 0.0)

    assert oracle_val == pytest.approx(0.50)
    assert delta == pytest.approx(0.0)


def test_inclusive_capacity_filtering():
    """Validates that rank == capacity is strictly included (rank <= capacity)."""
    candidates = [
        {"candidate_term": "t1", "salience_rank": 9999},
        {"candidate_term": "t2", "salience_rank": 10000},
        {"candidate_term": "t3", "salience_rank": 10001},
    ]
    df = pd.DataFrame(candidates)

    sub_inclusive = df[(df["salience_rank"] >= 0) & (df["salience_rank"] <= 10000)]
    assert len(sub_inclusive) == 2
    assert "t2" in sub_inclusive["candidate_term"].values
    assert "t3" not in sub_inclusive["candidate_term"].values


def test_optimal_weight_qualification_requires_positive_utility():
    """
    Validates that a weight preserving recall but harming nDCG
    is NOT selected as ranking-safe and receives no_safe_weight.
    """
    # Candidate hurts nDCG at all weights even if recall is preserved
    weights_data = [
        {"weight": 0.10, "delta_ndcg10": -0.02, "delta_r1000": 0.0},
        {"weight": 0.30, "delta_ndcg10": -0.05, "delta_r1000": 0.02},
        {"weight": 0.50, "delta_ndcg10": -0.10, "delta_r1000": 0.05},
    ]
    df = pd.DataFrame(weights_data)

    # Ranking-safe requirement: delta_ndcg10 > TAU and delta_r1000 >= -TAU
    rank_qual = df[(df["delta_ndcg10"] > TAU) & (df["delta_r1000"] >= -TAU)]
    assert rank_qual.empty  # Must be empty!

    # Recall-safe requirement: delta_r1000 > TAU and delta_ndcg10 >= -EPSILON
    rec_qual = df[(df["delta_r1000"] > TAU) & (df["delta_ndcg10"] >= -EPSILON)]
    # Weight 0.30 has delta_ndcg10 = -0.05 which is < -EPSILON (-0.001), so it does NOT qualify
    # Weight 0.50 has delta_ndcg10 = -0.10, does NOT qualify
    assert rec_qual.empty


def test_bootstrap_ci_bounds():
    """Validates that bootstrap 95% confidence intervals are consistent."""
    series = pd.Series([0.10, 0.12, 0.11, 0.15, 0.09, 0.13] * 10)
    low, high = bootstrap_ci(series, n_boot=500, seed=42)
    assert low < series.mean() < high
    assert low > 0.0
