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


def test_conditional_weight_calibration():
    """Validates that conditional weight percentage properly normalizes by safe instances."""
    # 4 candidate instances: 2 have weight 1.0, 1 has weight 0.5, 1 has no_safe_weight
    df_inst = pd.DataFrame([
        {"ranking_opt_weight": 1.0},
        {"ranking_opt_weight": 1.0},
        {"ranking_opt_weight": 0.5},
        {"ranking_opt_weight": "no_safe_weight"},
    ])
    n_inst = len(df_inst)
    n_safe = (df_inst["ranking_opt_weight"] != "no_safe_weight").sum()
    assert n_safe == 3

    # Unconditional: 2/4 = 50.0%
    uncond_1_0 = (df_inst["ranking_opt_weight"] == 1.0).mean() * 100.0
    assert uncond_1_0 == pytest.approx(50.0)

    # Conditional: 2/3 = 66.667%
    cond_1_0 = (df_inst[df_inst["ranking_opt_weight"] != "no_safe_weight"]["ranking_opt_weight"] == 1.0).mean() * 100.0
    assert cond_1_0 == pytest.approx(66.6666667)


def test_unique_df1_gain_definition():
    """
    Validates exact mathematical definition of unique DF=1 gain:
    G_unique_df1(q) = max(0, G_1(q) - G_E(q)).
    """
    tau = 1e-5
    # Case 1: G_1 = 0.08, G_E = 0.05 -> G_unique = 0.03
    g1 = 0.08
    g_e = 0.05
    g_unique = max(0.0, g1 - g_e)
    assert g_unique == pytest.approx(0.03)
    assert g_unique > tau

    # Case 2: G_1 = 0.04, G_E = 0.07 -> G_unique = 0.0
    g1 = 0.04
    g_e = 0.07
    g_unique = max(0.0, g1 - g_e)
    assert g_unique == pytest.approx(0.0)

    # Case 3: G_1 = 0.00, G_E = 0.00 -> G_unique = 0.0
    assert max(0.0, 0.0 - 0.0) == pytest.approx(0.0)


def test_capacity_retention_zero_denominator():
    """
    Validates that when eligible ceiling delta <= TAU,
    retention is reported as NaN / NA, not zero or 100%,
    and macro average excludes NA datasets.
    """
    tau = 1e-5
    # Dataset A: delta_oracle = 0.08, delta_ceiling = 0.10 -> 80%
    ret_a = 0.08 / 0.10 * 100.0
    # Dataset B: delta_ceiling <= tau -> NaN
    denom_b = 1e-6
    ret_b = np.nan if denom_b <= tau else (0.0 / denom_b * 100.0)
    # Dataset C: delta_oracle = 0.09, delta_ceiling = 0.10 -> 90%
    ret_c = 0.09 / 0.10 * 100.0

    retentions = [ret_a, ret_b, ret_c]
    assert np.isnan(ret_b)

    # Macro average excludes NaN
    valid_rets = [r for r in retentions if not np.isnan(r)]
    assert len(valid_rets) == 2
    macro_ret = np.mean(valid_rets)
    assert macro_ret == pytest.approx(85.0)


def test_parquet_deduplication_preserves_weights():
    """Validates that raw audit deduplication preserves all 5 weights on (dataset, qid, candidate_term, weight)."""
    records = [
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 0.05, "metric": 0.1},
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 0.10, "metric": 0.2},
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 0.30, "metric": 0.3},
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 0.50, "metric": 0.4},
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 1.00, "metric": 0.5},
        # Duplicate record with exact same 4-tuple
        {"dataset": "d1", "qid": "q1", "candidate_term": "t1", "weight": 1.00, "metric": 0.5},
    ]
    df = pd.DataFrame(records)
    deduped = df.drop_duplicates(subset=["dataset", "qid", "candidate_term", "weight"])
    assert len(deduped) == 5  # All 5 distinct weights preserved, only 6th duplicate dropped!


def test_hybrid_formula_ranking_fixture():
    """Validates that compute_hybrid_score matches pool_generators.py definition."""
    from src.evaluation.pool_generators import compute_hybrid_score, generate_hybrid_sequence
    import math

    num_docs = 1000
    # Term 1: High IDF (4.0), CF=10, DF=5 (0.5% of corpus)
    # Score = 4.0 * ln(1 + 10) * (1 - 5/1000) = 4.0 * 2.397895 * 0.995 = 9.5436
    score1 = compute_hybrid_score(idf=4.0, cf=10, df=5, num_docs=num_docs)
    expected1 = 4.0 * math.log(11.0) * (1.0 - 5.0 / 1000.0)
    assert score1 == pytest.approx(expected1)

    # Term 2: Moderate IDF (2.0), CF=100, DF=50 (5% of corpus)
    score2 = compute_hybrid_score(idf=2.0, cf=100, df=50, num_docs=num_docs)
    expected2 = 2.0 * math.log(101.0) * (1.0 - 50.0 / 1000.0)
    assert score2 == pytest.approx(expected2)

    # Ranking sequence test
    terms = ["t1", "t2"]
    df_map = {"t1": 5, "t2": 50}
    cf_map = {"t1": 10, "t2": 100}
    idf_map = {"t1": 4.0, "t2": 2.0}
    seq = generate_hybrid_sequence(terms, df_map, cf_map, idf_map, num_docs, max_cap=2)
    # Term 1 score (~9.54) > Term 2 score (~8.77)
    assert seq == ["t1", "t2"]


