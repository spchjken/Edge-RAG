#!/usr/bin/env python3
"""
Compile Stage 1 Vocabulary Pool Oracle & Capacity Isolation Tables.
Reads results/pool_isolation/pool_candidate_audit.parquet (or shards)
and outputs rigorously vetted artifacts adhering to reviewer contracts:
- pool_oracle_summary.csv (Table 1: Independent + Joint Safe Ceilings, 95% Bootstrap CIs)
- pool_df_band_attribution.csv (Table 2: DF Band Attribution across Candidate Instances)
- pool_df1_dominance.csv (Dedicated 5-bucket mutually exclusive DF=1 vs Eligible Dominance)
- pool_capacity_knee_curve.csv (Table 3: Fixed + Index-Derived Percentage Capacities)
- pool_weight_calibration.csv (Optimal weight distributions across DF bands and within-dataset IDF deciles)
- metadata_manifest.json (Environment, doc counts, and exact definitions)
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import numpy as np

# Exact mathematical tolerances
TAU = 1e-5         # Zero tolerance for utility and safety
EPSILON = 0.001     # Allowable nDCG drop for recall safety

# Capacities and policies
FIXED_CAPACITIES = [1000, 2500, 5000, 10000, 15000, 20000]
PERCENTAGE_CAPACITIES = [0.02, 0.05, 0.10, 0.20]
POLICIES = ["salience", "specificity", "hybrid", "stratified", "coverage"]
WEIGHTS = [0.05, 0.10, 0.30, 0.50, 1.00]

# Index-derived eligible vocabulary counts and collection sizes
ELIGIBLE_VOCAB_SIZES = {
    "scifact": 5595,
    "bright_aops": 33656,
    "nfcorpus": 11811,
    "trec_covid": 59395,
}

DOC_COUNTS = {
    "scifact": 5183,
    "bright_aops": 188002,
    "nfcorpus": 3633,
    "trec_covid": 171332,
}


def bootstrap_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42) -> tuple:
    """Computes percentile bootstrap 95% confidence intervals over query values."""
    if len(series) == 0 or series.isna().all():
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    values = series.dropna().to_numpy()
    if len(values) == 0:
        return 0.0, 0.0
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def classify_df1_dominance_query(g_df1: float, g_elig: float, tau: float = TAU) -> str:
    """
    Classifies a query into exactly one of five mutually exclusive dominance buckets
    using a strict decision tree.
    """
    if g_df1 <= tau and g_elig <= tau:
        return "Neither Useful"
    elif g_df1 > tau and g_elig <= tau:
        return "DF1 Wins"
    elif g_elig > tau and g_df1 <= tau:
        return "Eligible Wins, DF1 Inactive"
    elif abs(g_df1 - g_elig) <= tau:
        return "Useful Tie"
    elif g_df1 > g_elig:
        return "DF1 Wins"
    else:
        return "DF1 Useful but Dominated"


def compile_oracle_tables(audit_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading audit data from {audit_path}...")
    if os.path.isdir(audit_path):
        import glob
        shard_files = glob.glob(os.path.join(audit_path, "*.parquet"))
        print(f"Loading {len(shard_files)} shards from {audit_path}...")
        df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    else:
        df = pd.read_parquet(audit_path)

    print(f"Loaded {len(df):,} records across {df['dataset'].nunique()} datasets: {df['dataset'].unique().tolist()}")

    # Ensure qid is string and check column parity
    df["qid"] = df["qid"].astype(str)

    # Calculate delta metrics directly if not present
    if "delta_ndcg10" not in df.columns:
        df["delta_ndcg10"] = df["expanded_ndcg10"] - df["baseline_ndcg10"]
    if "delta_r1000" not in df.columns:
        df["delta_r1000"] = df["expanded_r1000"] - df["baseline_r1000"]

    # ----------------------------------------------------
    # 1. Macro Oracle Summary (Table 1)
    # ----------------------------------------------------
    print("Compiling Table 1: Macro Oracle Summary with Dual Joint-Safe Ceilings & Bootstrap CIs...")
    summary_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sorted(sub["qid"].unique())
        n_q = len(all_qids)

        # Baseline per query
        q_base_ndcg = sub.groupby("qid")["baseline_ndcg10"].first()
        q_base_r100 = sub.groupby("qid")["baseline_r100"].first()
        q_base_r1000 = sub.groupby("qid")["baseline_r1000"].first()

        base_ndcg10 = q_base_ndcg.mean()
        base_r100 = q_base_r100.mean()
        base_r1000 = q_base_r1000.mean()

        # Helper for per-query abstaining ceiling
        def compute_abstaining_ceiling(candidates_df, score_col, base_series):
            if candidates_df.empty:
                return base_series.copy(), pd.Series(0.0, index=all_qids)
            q_max = candidates_df.groupby("qid")[score_col].max()
            q_oracle = pd.Series(base_series, index=all_qids)
            for q in all_qids:
                if q in q_max:
                    q_oracle[q] = max(base_series[q], q_max[q])
            q_delta = q_oracle - base_series
            # Guarantee numerical non-negativity
            q_delta = q_delta.apply(lambda d: max(d, 0.0))
            return q_oracle, q_delta

        # A. Raw Lexical Ceilings
        raw_best_oracle, raw_best_delta = compute_abstaining_ceiling(sub, "expanded_ndcg10", q_base_ndcg)
        raw_r1000_oracle, raw_r1000_delta = compute_abstaining_ceiling(sub, "expanded_r1000", q_base_r1000)

        sub_w10 = sub[sub["weight"] == 0.10]
        raw_w10_oracle, raw_w10_delta = compute_abstaining_ceiling(sub_w10, "expanded_ndcg10", q_base_ndcg)

        # B. Operational Eligible Ceilings (is_raw_only == False)
        elig = sub[sub["is_raw_only"] == False]
        elig_best_oracle, elig_best_delta = compute_abstaining_ceiling(elig, "expanded_ndcg10", q_base_ndcg)
        elig_r1000_oracle, elig_r1000_delta = compute_abstaining_ceiling(elig, "expanded_r1000", q_base_r1000)

        elig_w10 = elig[elig["weight"] == 0.10]
        elig_w10_oracle, elig_w10_delta = compute_abstaining_ceiling(elig_w10, "expanded_ndcg10", q_base_ndcg)

        # C. Same-Intervention Joint Safe Oracles (Eligible vocabulary)
        # 1. Ranking-Priority Joint Safe Oracle: max delta_ndcg10 s.t. delta_r1000 >= -TAU
        q_rp_ndcg = pd.Series(q_base_ndcg, index=all_qids)
        q_rp_delta_ndcg = pd.Series(0.0, index=all_qids)
        q_rp_delta_r1000 = pd.Series(0.0, index=all_qids)

        # 2. Recall-Priority Joint Safe Oracle: max delta_r1000 s.t. delta_ndcg10 >= -EPSILON
        q_recp_r1000 = pd.Series(q_base_r1000, index=all_qids)
        q_recp_delta_r1000 = pd.Series(0.0, index=all_qids)
        q_recp_delta_ndcg = pd.Series(0.0, index=all_qids)

        for q in all_qids:
            q_elig = elig[elig["qid"] == q]
            if not q_elig.empty:
                # Ranking-priority qualifying candidates
                rp_qual = q_elig[q_elig["delta_r1000"] >= -TAU]
                if not rp_qual.empty:
                    # Sort by primary delta desc, secondary delta desc, weight asc, term asc
                    rp_sorted = rp_qual.sort_values(
                        by=["delta_ndcg10", "delta_r1000", "weight", "candidate_term"],
                        ascending=[False, False, True, True],
                    )
                    top_rp = rp_sorted.iloc[0]
                    if top_rp["delta_ndcg10"] > TAU:
                        q_rp_delta_ndcg[q] = top_rp["delta_ndcg10"]
                        q_rp_ndcg[q] = q_base_ndcg[q] + top_rp["delta_ndcg10"]
                        q_rp_delta_r1000[q] = top_rp["delta_r1000"]

                # Recall-priority qualifying candidates
                recp_qual = q_elig[q_elig["delta_ndcg10"] >= -EPSILON]
                if not recp_qual.empty:
                    recp_sorted = recp_qual.sort_values(
                        by=["delta_r1000", "delta_ndcg10", "weight", "candidate_term"],
                        ascending=[False, False, True, True],
                    )
                    top_recp = recp_sorted.iloc[0]
                    if top_recp["delta_r1000"] > TAU:
                        q_recp_delta_r1000[q] = top_recp["delta_r1000"]
                        q_recp_r1000[q] = q_base_r1000[q] + top_recp["delta_r1000"]
                        q_recp_delta_ndcg[q] = top_recp["delta_ndcg10"]

        # Bootstrap CIs for key gains
        ci_raw_best = bootstrap_ci(raw_best_delta)
        ci_elig_best = bootstrap_ci(elig_best_delta)
        ci_rp_joint = bootstrap_ci(q_rp_delta_ndcg)
        ci_recp_joint = bootstrap_ci(q_recp_delta_r1000)

        row = {
            "dataset": ds,
            "num_docs": DOC_COUNTS.get(ds, 0),
            "eligible_vocab_size": ELIGIBLE_VOCAB_SIZES.get(ds, 0),
            "num_queries": n_q,
            "baseline_ndcg10": base_ndcg10,
            "baseline_r100": base_r100,
            "baseline_r1000": base_r1000,
            # Independent Ceilings
            "raw_ceiling_screened_best_w_ndcg10": raw_best_oracle.mean(),
            "raw_ceiling_screened_best_w_delta": raw_best_delta.mean(),
            "raw_ceiling_screened_best_w_delta_ci95": f"[{ci_raw_best[0]:.4f}, {ci_raw_best[1]:.4f}]",
            "raw_ceiling_mu010_ndcg10_exact": raw_w10_oracle.mean(),
            "raw_ceiling_mu010_delta_exact": raw_w10_delta.mean(),
            "eligible_ceiling_screened_best_w_ndcg10": elig_best_oracle.mean(),
            "eligible_ceiling_screened_best_w_delta": elig_best_delta.mean(),
            "eligible_ceiling_screened_best_w_delta_ci95": f"[{ci_elig_best[0]:.4f}, {ci_elig_best[1]:.4f}]",
            "eligible_ceiling_mu010_ndcg10_exact": elig_w10_oracle.mean(),
            "eligible_ceiling_mu010_delta_exact": elig_w10_delta.mean(),
            "eligible_ceiling_screened_best_w_r1000": elig_r1000_oracle.mean(),
            "eligible_ceiling_screened_best_w_delta_r1000": elig_r1000_delta.mean(),
            # Dual Joint Safe Oracles
            "ranking_priority_joint_safe_ndcg10": q_rp_ndcg.mean(),
            "ranking_priority_joint_safe_delta_ndcg10": q_rp_delta_ndcg.mean(),
            "ranking_priority_joint_safe_delta_ci95": f"[{ci_rp_joint[0]:.4f}, {ci_rp_joint[1]:.4f}]",
            "ranking_priority_joint_safe_paired_delta_r1000": q_rp_delta_r1000.mean(),
            "recall_priority_joint_safe_r1000": q_recp_r1000.mean(),
            "recall_priority_joint_safe_delta_r1000": q_recp_delta_r1000.mean(),
            "recall_priority_joint_safe_delta_ci95": f"[{ci_recp_joint[0]:.4f}, {ci_recp_joint[1]:.4f}]",
            "recall_priority_joint_safe_paired_delta_ndcg10": q_recp_delta_ndcg.mean(),
        }

        # Deployable policies at 10k capacity (exact 5-weight grid with abstention)
        for pol in POLICIES:
            rank_col = f"{pol}_rank"
            p_sub = sub[(sub[rank_col] >= 0) & (sub[rank_col] <= 10000)]
            p_oracle, p_delta = compute_abstaining_ceiling(p_sub, "expanded_ndcg10", q_base_ndcg)
            row[f"{pol}_10k_ndcg10_exact"] = p_oracle.mean()
            row[f"{pol}_10k_delta_exact"] = p_delta.mean()

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "pool_oracle_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # ----------------------------------------------------
    # 2. Dedicated DF=1 Dominance Table (pool_df1_dominance.csv)
    # ----------------------------------------------------
    print("Compiling Dedicated DF=1 vs. Eligible Dominance Table with Strict Precedence...")
    dominance_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sorted(sub["qid"].unique())
        n_q = len(all_qids)

        q_base_ndcg = sub.groupby("qid")["baseline_ndcg10"].first()

        for scope_label, sub_scope in [("exact_mu010", sub[sub["weight"] == 0.10]), ("screened_multi_weight", sub)]:
            bucket_counts = {
                "Neither Useful": 0,
                "DF1 Wins": 0,
                "Eligible Wins, DF1 Inactive": 0,
                "Useful Tie": 0,
                "DF1 Useful but Dominated": 0,
            }

            for q in all_qids:
                q_sub = sub_scope[sub_scope["qid"] == q]
                if q_sub.empty:
                    bucket_counts["Neither Useful"] += 1
                    continue

                # Max delta for DF=1 candidates
                df1_cands = q_sub[q_sub["df_band"] == "DF=1"]
                g_df1 = max(df1_cands["delta_ndcg10"].max(), 0.0) if not df1_cands.empty else 0.0

                # Max delta for Eligible candidates
                elig_cands = q_sub[q_sub["is_raw_only"] == False]
                g_elig = max(elig_cands["delta_ndcg10"].max(), 0.0) if not elig_cands.empty else 0.0

                bucket = classify_df1_dominance_query(g_df1, g_elig, tau=TAU)
                bucket_counts[bucket] += 1

            dom_row = {
                "dataset": ds,
                "evaluation_scope": scope_label,
                "num_queries": n_q,
                "neither_useful_count": bucket_counts["Neither Useful"],
                "neither_useful_pct": bucket_counts["Neither Useful"] / n_q * 100.0,
                "df1_wins_count": bucket_counts["DF1 Wins"],
                "df1_wins_pct": bucket_counts["DF1 Wins"] / n_q * 100.0,
                "eligible_wins_df1_inactive_count": bucket_counts["Eligible Wins, DF1 Inactive"],
                "eligible_wins_df1_inactive_pct": bucket_counts["Eligible Wins, DF1 Inactive"] / n_q * 100.0,
                "useful_tie_count": bucket_counts["Useful Tie"],
                "useful_tie_pct": bucket_counts["Useful Tie"] / n_q * 100.0,
                "df1_useful_dominated_count": bucket_counts["DF1 Useful but Dominated"],
                "df1_useful_dominated_pct": bucket_counts["DF1 Useful but Dominated"] / n_q * 100.0,
            }
            dominance_rows.append(dom_row)

    df_dom = pd.DataFrame(dominance_rows)
    dom_csv = os.path.join(output_dir, "pool_df1_dominance.csv")
    df_dom.to_csv(dom_csv, index=False)
    print(f"Wrote {dom_csv}")

    # ----------------------------------------------------
    # 3. DF Band Attribution Table (Table 2)
    # ----------------------------------------------------
    print("Compiling Table 2: DF Band Attribution across Candidate Instances...")
    attribution_rows = []

    # Identify deployable pool membership: any candidate with rank <= 20000 in any policy
    # Candidates in deployable pools received the full 5-weight grid.
    rank_cols = [f"{pol}_rank" for pol in POLICIES]
    df["is_deployable"] = False
    for col in rank_cols:
        df["is_deployable"] = df["is_deployable"] | ((df[col] >= 0) & (df[col] <= 20000))

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sorted(sub["qid"].unique())
        n_q = len(all_qids)

        for band in sorted(sub["df_band"].unique()):
            b_sub = sub[sub["df_band"] == band]
            if b_sub.empty:
                continue

            # 1. Deduplicate by query-term candidate instance
            inst_sub = b_sub.groupby(["dataset", "qid", "candidate_term"])

            # Per-instance metrics
            inst_ref_w10 = b_sub[b_sub["weight"] == 0.10].groupby(["dataset", "qid", "candidate_term"]).first()
            inst_max_deltas = inst_sub.agg({
                "delta_ndcg10": "max",
                "delta_r1000": "max",
                "is_deployable": "first",
            })

            n_instances = len(inst_max_deltas)
            n_unique_terms = b_sub["candidate_term"].nunique()
            queries_with_candidates = b_sub["qid"].nunique() / n_q

            # Per-query maximum deltas (with abstention)
            max_delta_per_q = b_sub.groupby("qid")["delta_ndcg10"].max()
            q_deltas = pd.Series(0.0, index=all_qids)
            for q, d in max_delta_per_q.items():
                q_deltas[q] = max(d, 0.0)

            max_delta_r1000_per_q = b_sub.groupby("qid")["delta_r1000"].max()
            q_r_deltas = pd.Series(0.0, index=all_qids)
            for q, d in max_delta_r1000_per_q.items():
                q_r_deltas[q] = max(d, 0.0)

            queries_helped = (max_delta_per_q > TAU).sum() / n_q

            # Exact reference-weight utility across all candidate instances
            if not inst_ref_w10.empty:
                ref_useful = (inst_ref_w10["delta_ndcg10"] > TAU).mean()
                ref_ranking_safe = ((inst_ref_w10["delta_ndcg10"] > TAU) & (inst_ref_w10["delta_r1000"] >= -TAU)).mean()
                ref_recall_safe = ((inst_ref_w10["delta_r1000"] > TAU) & (inst_ref_w10["delta_ndcg10"] >= -EPSILON)).mean()
            else:
                ref_useful = ref_ranking_safe = ref_recall_safe = 0.0

            # Screened any-weight utility across all candidate instances (lower bound)
            screened_useful = (inst_max_deltas["delta_ndcg10"] > TAU).mean()

            # Exact any-weight utility across deployable candidate instances
            deployable_instances = inst_max_deltas[inst_max_deltas["is_deployable"] == True]
            if not deployable_instances.empty:
                deployable_useful = (deployable_instances["delta_ndcg10"] > TAU).mean()

                # Robustness across weights: instance is safe at >= 3 weights
                dep_full = b_sub[b_sub["is_deployable"] == True]
                ranking_safe_weights_per_inst = dep_full[
                    (dep_full["delta_ndcg10"] > TAU) & (dep_full["delta_r1000"] >= -TAU)
                ].groupby(["dataset", "qid", "candidate_term"])["weight"].nunique()

                recall_safe_weights_per_inst = dep_full[
                    (dep_full["delta_r1000"] > TAU) & (dep_full["delta_ndcg10"] >= -EPSILON)
                ].groupby(["dataset", "qid", "candidate_term"])["weight"].nunique()

                ranking_robust = (ranking_safe_weights_per_inst >= 3).reindex(deployable_instances.index, fill_value=False).mean()
                recall_robust = (recall_safe_weights_per_inst >= 3).reindex(deployable_instances.index, fill_value=False).mean()
            else:
                deployable_useful = ranking_robust = recall_robust = 0.0

            attribution_rows.append({
                "dataset": ds,
                "df_band": band,
                "candidate_instances_count": n_instances,
                "unique_terms_count": n_unique_terms,
                "queries_with_candidates_pct": queries_with_candidates,
                "mean_max_delta_ndcg10_abstain": q_deltas.mean(),
                "mean_max_delta_r1000_abstain": q_r_deltas.mean(),
                "pct_queries_helped": queries_helped,
                "pct_instances_ref_weight_useful_exact": ref_useful,
                "pct_instances_ref_weight_ranking_safe_exact": ref_ranking_safe,
                "pct_instances_ref_weight_recall_safe_exact": ref_recall_safe,
                "pct_instances_any_weight_useful_screened": screened_useful,
                "pct_instances_any_weight_useful_deployable_exact": deployable_useful,
                "pct_instances_ranking_robust_deployable_exact": ranking_robust,
                "pct_instances_recall_robust_deployable_exact": recall_robust,
            })

    df_attribution = pd.DataFrame(attribution_rows)
    attribution_csv = os.path.join(output_dir, "pool_df_band_attribution.csv")
    df_attribution.to_csv(attribution_csv, index=False)
    print(f"Wrote {attribution_csv}")

    # ----------------------------------------------------
    # 4. Capacity Knee Curve (Table 3: Fixed + Percentage Capacities)
    # ----------------------------------------------------
    print("Compiling Table 3: Fixed & Relative Percentage Capacities with Inclusive Rank Filter...")
    knee_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sorted(sub["qid"].unique())
        n_q = len(all_qids)

        q_base_ndcg = sub.groupby("qid")["baseline_ndcg10"].first()
        q_base_r1000 = sub.groupby("qid")["baseline_r1000"].first()
        base_ndcg = q_base_ndcg.mean()
        base_r1000 = q_base_r1000.mean()

        v_elig_total = ELIGIBLE_VOCAB_SIZES.get(ds, sub[sub["is_raw_only"] == False]["candidate_term"].nunique())

        # Baseline eligible headroom with abstention
        elig = sub[sub["is_raw_only"] == False]
        if not elig.empty:
            q_elig_max = elig.groupby("qid")["expanded_ndcg10"].max()
            q_elig_delta = pd.Series(0.0, index=all_qids)
            for q in all_qids:
                if q in q_elig_max:
                    q_elig_delta[q] = max(q_elig_max[q] - q_base_ndcg[q], 0.0)
            elig_best_delta = max(q_elig_delta.mean(), TAU)
        else:
            elig_best_delta = TAU

        for pol in POLICIES:
            rank_col = f"{pol}_rank"

            # Combine Fixed capacities and Percentage capacities
            capacity_evaluations = []
            for cap in FIXED_CAPACITIES:
                capacity_evaluations.append(("fixed", cap, cap, (cap / v_elig_total) * 100.0))

            for p in PERCENTAGE_CAPACITIES:
                eff_cap = min(20000, max(1, int(np.floor(p * v_elig_total))))
                capacity_evaluations.append(("percentage", p, eff_cap, p * 100.0))

            for cap_type, req_cap, eff_cap, actual_pct in capacity_evaluations:
                cap_sub = sub[(sub[rank_col] >= 0) & (sub[rank_col] <= eff_cap)]

                if cap_sub.empty:
                    knee_rows.append({
                        "dataset": ds,
                        "policy": pol,
                        "capacity_type": cap_type,
                        "requested_capacity": req_cap,
                        "effective_capacity_terms": eff_cap,
                        "actual_pct_eligible_vocab": actual_pct,
                        "oracle_ndcg10": base_ndcg,
                        "delta_ndcg10": 0.0,
                        "oracle_r1000": base_r1000,
                        "delta_r1000": 0.0,
                        "pct_retained_vs_eligible": 0.0,
                        "recall_safe_query_rate": 0.0,
                        "ranking_safe_query_rate": 0.0,
                    })
                    continue

                # Abstaining max per query
                q_max_ndcg = cap_sub.groupby("qid")["expanded_ndcg10"].max()
                q_ndcg = pd.Series(q_base_ndcg, index=all_qids)
                q_delta_ndcg = pd.Series(0.0, index=all_qids)
                for q in all_qids:
                    if q in q_max_ndcg:
                        q_ndcg[q] = max(q_base_ndcg[q], q_max_ndcg[q])
                        q_delta_ndcg[q] = max(q_max_ndcg[q] - q_base_ndcg[q], 0.0)

                q_max_r1000 = cap_sub.groupby("qid")["expanded_r1000"].max()
                q_r = pd.Series(q_base_r1000, index=all_qids)
                q_delta_r = pd.Series(0.0, index=all_qids)
                for q in all_qids:
                    if q in q_max_r1000:
                        q_r[q] = max(q_base_r1000[q], q_max_r1000[q])
                        q_delta_r[q] = max(q_max_r1000[q] - q_base_r1000[q], 0.0)

                # Predeclared query safety rates at reference weight mu=0.10
                w10_cap_sub = cap_sub[cap_sub["weight"] == 0.10]
                if not w10_cap_sub.empty:
                    q_w10_qual_rec = w10_cap_sub[
                        (w10_cap_sub["delta_r1000"] > TAU) & (w10_cap_sub["delta_ndcg10"] >= -EPSILON)
                    ].groupby("qid").size()
                    recall_safe_rate = (q_w10_qual_rec > 0).reindex(all_qids, fill_value=False).sum() / n_q

                    q_w10_qual_rank = w10_cap_sub[
                        (w10_cap_sub["delta_ndcg10"] > TAU) & (w10_cap_sub["delta_r1000"] >= -TAU)
                    ].groupby("qid").size()
                    ranking_safe_rate = (q_w10_qual_rank > 0).reindex(all_qids, fill_value=False).sum() / n_q
                else:
                    recall_safe_rate = ranking_safe_rate = 0.0

                mean_delta_ndcg = q_delta_ndcg.mean()
                mean_delta_r1000 = q_delta_r.mean()
                pct_retained = min((mean_delta_ndcg / elig_best_delta) * 100.0, 100.0)

                knee_rows.append({
                    "dataset": ds,
                    "policy": pol,
                    "capacity_type": cap_type,
                    "requested_capacity": req_cap,
                    "effective_capacity_terms": eff_cap,
                    "actual_pct_eligible_vocab": actual_pct,
                    "oracle_ndcg10": q_ndcg.mean(),
                    "delta_ndcg10": mean_delta_ndcg,
                    "oracle_r1000": q_r.mean(),
                    "delta_r1000": mean_delta_r1000,
                    "pct_retained_vs_eligible": pct_retained,
                    "recall_safe_query_rate": recall_safe_rate,
                    "ranking_safe_query_rate": ranking_safe_rate,
                })

    df_knee = pd.DataFrame(knee_rows)
    knee_csv = os.path.join(output_dir, "pool_capacity_knee_curve.csv")
    df_knee.to_csv(knee_csv, index=False)
    print(f"Wrote {knee_csv}")

    # ----------------------------------------------------
    # 5. Optimal Weight Distribution (pool_weight_calibration.csv)
    # ----------------------------------------------------
    print("Compiling Weight Calibration: Optimal Weights across DF Bands & Within-Dataset IDF Deciles...")
    calib_rows = []

    # Analyze deployable candidates only (which received the full 5-weight grid)
    deployable_df = df[df["is_deployable"] == True].copy()

    for ds in deployable_df["dataset"].unique():
        ds_sub = deployable_df[deployable_df["dataset"] == ds].copy()

        # Compute within-dataset IDF deciles
        # Extract unique terms and their IDFs for this dataset
        term_idf = ds_sub.groupby("candidate_term")["candidate_idf"].first()
        try:
            term_deciles = pd.qcut(term_idf, q=10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop")
            ds_sub["idf_decile"] = ds_sub["candidate_term"].map(term_deciles).astype(str)
        except Exception:
            ds_sub["idf_decile"] = "D1"

        # Compute optimal weight per candidate instance: (dataset, qid, candidate_term)
        inst_groups = ds_sub.groupby(["dataset", "qid", "candidate_term"])

        instance_results = []
        for (d, q, t), grp in inst_groups:
            df_band = grp["df_band"].iloc[0]
            idf_dec = grp["idf_decile"].iloc[0]

            # 1. Ranking-Safe Optimum: delta_ndcg10 > TAU and delta_r1000 >= -TAU
            rank_qual = grp[(grp["delta_ndcg10"] > TAU) & (grp["delta_r1000"] >= -TAU)]
            if not rank_qual.empty:
                # Primary metric desc, secondary metric desc, weight asc
                best_rank_w = rank_qual.sort_values(
                    by=["delta_ndcg10", "delta_r1000", "weight"],
                    ascending=[False, False, True]
                )["weight"].iloc[0]
            else:
                best_rank_w = "no_safe_weight"

            # 2. Recall-Safe Optimum: delta_r1000 > TAU and delta_ndcg10 >= -EPSILON
            rec_qual = grp[(grp["delta_r1000"] > TAU) & (grp["delta_ndcg10"] >= -EPSILON)]
            if not rec_qual.empty:
                best_rec_w = rec_qual.sort_values(
                    by=["delta_r1000", "delta_ndcg10", "weight"],
                    ascending=[False, False, True]
                )["weight"].iloc[0]
            else:
                best_rec_w = "no_safe_weight"

            instance_results.append({
                "dataset": d,
                "qid": q,
                "candidate_term": t,
                "df_band": df_band,
                "idf_decile": idf_dec,
                "ranking_opt_weight": best_rank_w,
                "recall_opt_weight": best_rec_w,
            })

        df_inst = pd.DataFrame(instance_results)

        # Aggregate distributions by DF Band
        for band in sorted(df_inst["df_band"].unique()):
            b_inst = df_inst[df_inst["df_band"] == band]
            n_inst = len(b_inst)

            calib_row = {
                "dataset": ds,
                "partition_type": "df_band",
                "partition_bin": band,
                "instance_count": n_inst,
            }
            # Ranking-safe weight distribution
            for w in WEIGHTS:
                calib_row[f"ranking_safe_pct_w_{w}"] = (b_inst["ranking_opt_weight"] == w).mean() * 100.0
            calib_row["ranking_safe_pct_no_safe"] = (b_inst["ranking_opt_weight"] == "no_safe_weight").mean() * 100.0

            # Recall-safe weight distribution
            for w in WEIGHTS:
                calib_row[f"recall_safe_pct_w_{w}"] = (b_inst["recall_opt_weight"] == w).mean() * 100.0
            calib_row["recall_safe_pct_no_safe"] = (b_inst["recall_opt_weight"] == "no_safe_weight").mean() * 100.0

            calib_rows.append(calib_row)

        # Aggregate distributions by IDF Decile
        for dec in sorted(df_inst["idf_decile"].unique()):
            d_inst = df_inst[df_inst["idf_decile"] == dec]
            n_inst = len(d_inst)

            calib_row = {
                "dataset": ds,
                "partition_type": "idf_decile",
                "partition_bin": dec,
                "instance_count": n_inst,
            }
            for w in WEIGHTS:
                calib_row[f"ranking_safe_pct_w_{w}"] = (d_inst["ranking_opt_weight"] == w).mean() * 100.0
            calib_row["ranking_safe_pct_no_safe"] = (d_inst["ranking_opt_weight"] == "no_safe_weight").mean() * 100.0

            for w in WEIGHTS:
                calib_row[f"recall_safe_pct_w_{w}"] = (d_inst["recall_opt_weight"] == w).mean() * 100.0
            calib_row["recall_safe_pct_no_safe"] = (d_inst["recall_opt_weight"] == "no_safe_weight").mean() * 100.0

            calib_rows.append(calib_row)

    df_calib = pd.DataFrame(calib_rows)
    calib_csv = os.path.join(output_dir, "pool_weight_calibration.csv")
    df_calib.to_csv(calib_csv, index=False)
    print(f"Wrote {calib_csv}")

    # ----------------------------------------------------
    # 6. Metadata Manifest
    # ----------------------------------------------------
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    manifest = {
        "git_commit": git_hash,
        "datasets": df["dataset"].unique().tolist(),
        "total_executed_candidate_query_weight_variants": len(df),
        "total_queries": df.groupby("dataset")["qid"].nunique().to_dict(),
        "document_counts": DOC_COUNTS,
        "eligible_vocab_sizes": ELIGIBLE_VOCAB_SIZES,
        "weights": WEIGHTS,
        "fixed_capacities": FIXED_CAPACITIES,
        "percentage_capacities": PERCENTAGE_CAPACITIES,
        "policies": POLICIES,
        "tolerances": {
            "tau_zero_tolerance": TAU,
            "epsilon_recall_safety_ndcg_margin": EPSILON,
        },
        "metric_definitions": {
            "nDCG@10": "Official linear ir_measures.nDCG@10",
            "R@1000": "Official ir_measures.R@1000",
            "abstaining_oracle": "max(baseline, max_{t,mu} U_q(t,mu))",
            "ranking_safe_weight": f"delta_ndcg10 > {TAU} and delta_r1000 >= -{TAU}",
            "recall_safe_weight": f"delta_r1000 > {TAU} and delta_ndcg10 >= -{EPSILON}",
            "ranking_priority_joint_safe": f"max delta_ndcg10 s.t. delta_r1000 >= -{TAU}",
            "recall_priority_joint_safe": f"max delta_r1000 s.t. delta_ndcg10 >= -{EPSILON}",
        },
    }
    manifest_path = os.path.join(output_dir, "metadata_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Compile Stage 1 Oracle Tables")
    parser.add_argument("--audit-parquet", type=str, default="results/pool_isolation/pool_candidate_audit.parquet")
    parser.add_argument("--output-dir", type=str, default="results/pool_isolation")
    args = parser.parse_args()

    compile_oracle_tables(args.audit_parquet, args.output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        compile_oracle_tables("results/pool_isolation/pool_candidate_audit.parquet", "results/pool_isolation")
