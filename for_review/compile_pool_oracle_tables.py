#!/usr/bin/env python3
"""
Compile Stage 1 Vocabulary Pool Oracle & Capacity Isolation Tables.
Reads results/pool_isolation/pool_candidate_audit.parquet (or shards)
and outputs:
- pool_oracle_summary.csv
- pool_df_band_attribution.csv
- pool_capacity_knee_curve.csv
- metadata_manifest.json
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import numpy as np

CAPACITIES = [1000, 2500, 5000, 10000, 15000, 20000]
POLICIES = ["salience", "specificity", "hybrid", "stratified", "coverage"]
WEIGHTS = [0.05, 0.10, 0.30, 0.50, 1.00]


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

    # ----------------------------------------------------
    # 1. Macro Oracle Summary (Table 1)
    # ----------------------------------------------------
    summary_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        qids = sub["qid"].unique()
        n_q = len(qids)

        base_ndcg10 = sub.groupby("qid")["baseline_ndcg10"].first().mean()
        base_r100 = sub.groupby("qid")["baseline_r100"].first().mean()
        base_r1000 = sub.groupby("qid")["baseline_r1000"].first().mean()

        # Raw lexical ceiling (all candidates, multi-weight vs mu=0.10)
        raw_best_ndcg = sub.groupby("qid")["expanded_ndcg10"].max().mean()
        raw_w10_sub = sub[sub["weight"] == 0.10]
        raw_w10_ndcg = raw_w10_sub.groupby("qid")["expanded_ndcg10"].max().mean() if not raw_w10_sub.empty else raw_best_ndcg

        # Operational Eligible ceiling (is_raw_only == False)
        elig = sub[sub["is_raw_only"] == False]
        if not elig.empty:
            elig_best_ndcg = elig.groupby("qid")["expanded_ndcg10"].max().mean()
            elig_w10_sub = elig[elig["weight"] == 0.10]
            elig_w10_ndcg = elig_w10_sub.groupby("qid")["expanded_ndcg10"].max().mean() if not elig_w10_sub.empty else elig_best_ndcg
            elig_best_r1000 = elig.groupby("qid")["expanded_r1000"].max().mean()
        else:
            elig_best_ndcg = raw_best_ndcg
            elig_w10_ndcg = raw_w10_ndcg
            elig_best_r1000 = base_r1000

        row = {
            "dataset": ds,
            "num_queries": n_q,
            "baseline_ndcg10": base_ndcg10,
            "baseline_r100": base_r100,
            "baseline_r1000": base_r1000,
            "raw_ceiling_best_w_ndcg10": raw_best_ndcg,
            "raw_ceiling_best_w_delta": raw_best_ndcg - base_ndcg10,
            "raw_ceiling_mu010_ndcg10": raw_w10_ndcg,
            "raw_ceiling_mu010_delta": raw_w10_ndcg - base_ndcg10,
            "eligible_ceiling_best_w_ndcg10": elig_best_ndcg,
            "eligible_ceiling_best_w_delta": elig_best_ndcg - base_ndcg10,
            "eligible_ceiling_mu010_ndcg10": elig_w10_ndcg,
            "eligible_ceiling_mu010_delta": elig_w10_ndcg - base_ndcg10,
            "eligible_ceiling_best_w_r1000": elig_best_r1000,
            "eligible_ceiling_best_w_delta_r1000": elig_best_r1000 - base_r1000,
        }

        # Add 10k capacity best-performing deployable policy
        for pol in POLICIES:
            rank_col = f"{pol}_rank"
            p_sub = sub[(sub[rank_col] >= 0) & (sub[rank_col] < 10000)]
            if not p_sub.empty:
                p_max = p_sub.groupby("qid")["expanded_ndcg10"].max().mean()
                row[f"{pol}_10k_ndcg10"] = p_max
                row[f"{pol}_10k_delta"] = p_max - base_ndcg10
            else:
                row[f"{pol}_10k_ndcg10"] = base_ndcg10
                row[f"{pol}_10k_delta"] = 0.0

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "pool_oracle_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # ----------------------------------------------------
    # 2. DF Band Attribution Table (Table 2)
    # ----------------------------------------------------
    attribution_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sub["qid"].unique()
        n_q = len(all_qids)

        for band in sorted(sub["df_band"].unique()):
            b_sub = sub[sub["df_band"] == band]
            if b_sub.empty:
                continue

            # Per-query maximum delta nDCG@10 (if query has no candidates in band, delta is 0.0)
            max_delta_per_q = b_sub.groupby("qid")["delta_ndcg10"].max()
            q_deltas = pd.Series(0.0, index=all_qids)
            q_deltas.update(max_delta_per_q)

            # Per-query maximum delta R@1000
            max_delta_r1000_per_q = b_sub.groupby("qid")["delta_r1000"].max()
            q_r_deltas = pd.Series(0.0, index=all_qids)
            q_r_deltas.update(max_delta_r1000_per_q)

            # Useful and safe fractions
            queries_helped = (max_delta_per_q > 0.001).sum() / n_q
            queries_with_candidates = b_sub["qid"].nunique() / n_q

            # Overall term stats
            attribution_rows.append({
                "dataset": ds,
                "df_band": band,
                "term_count": len(b_sub["candidate_term"].unique()),
                "queries_with_candidates_pct": queries_with_candidates,
                "mean_max_delta_ndcg10": q_deltas.mean(),
                "mean_max_delta_r1000": q_r_deltas.mean(),
                "pct_queries_helped": queries_helped,
                "pct_terms_ranking_useful": b_sub["is_ranking_useful"].mean(),
                "pct_terms_ranking_safe": b_sub["is_ranking_safe"].mean(),
                "pct_terms_recall_safe": b_sub["is_recall_safe"].mean(),
            })

    df_attribution = pd.DataFrame(attribution_rows)
    attribution_csv = os.path.join(output_dir, "pool_df_band_attribution.csv")
    df_attribution.to_csv(attribution_csv, index=False)
    print(f"Wrote {attribution_csv}")

    # ----------------------------------------------------
    # 3. Capacity Knee Curve (Table 3)
    # ----------------------------------------------------
    knee_rows = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        all_qids = sub["qid"].unique()
        n_q = len(all_qids)

        base_ndcg = sub.groupby("qid")["baseline_ndcg10"].first().mean()
        base_r1000 = sub.groupby("qid")["baseline_r1000"].first().mean()

        # Eligible ceiling delta
        elig = sub[sub["is_raw_only"] == False]
        elig_best_delta = (elig.groupby("qid")["expanded_ndcg10"].max().mean() - base_ndcg) if not elig.empty else 0.001
        elig_best_delta = max(elig_best_delta, 1e-6)

        for pol in POLICIES:
            rank_col = f"{pol}_rank"
            for cap in CAPACITIES:
                cap_sub = sub[(sub[rank_col] >= 0) & (sub[rank_col] < cap)]
                if cap_sub.empty:
                    knee_rows.append({
                        "dataset": ds,
                        "policy": pol,
                        "capacity": cap,
                        "oracle_ndcg10": base_ndcg,
                        "delta_ndcg10": 0.0,
                        "oracle_r1000": base_r1000,
                        "delta_r1000": 0.0,
                        "pct_retained_vs_eligible": 0.0,
                        "recall_safe_query_rate": 0.0,
                        "ranking_safe_query_rate": 0.0,
                    })
                    continue

                # Max per query across all 5 weights
                q_max_ndcg = cap_sub.groupby("qid")["expanded_ndcg10"].max()
                q_ndcg = pd.Series(base_ndcg, index=all_qids)
                q_ndcg.update(q_max_ndcg)

                q_max_r1000 = cap_sub.groupby("qid")["expanded_r1000"].max()
                q_r = pd.Series(base_r1000, index=all_qids)
                q_r.update(q_max_r1000)

                # Predeclared primary score: macro recall-safe query rate at mu=0.10
                w10_cap_sub = cap_sub[cap_sub["weight"] == 0.10]
                if not w10_cap_sub.empty:
                    recall_safe_queries = w10_cap_sub.groupby("qid")["is_recall_safe"].any()
                    recall_safe_rate = recall_safe_queries.sum() / n_q
                    ranking_safe_queries = w10_cap_sub.groupby("qid")["is_ranking_safe"].any()
                    ranking_safe_rate = ranking_safe_queries.sum() / n_q
                else:
                    recall_safe_rate = 0.0
                    ranking_safe_rate = 0.0

                mean_delta_ndcg = q_ndcg.mean() - base_ndcg
                knee_rows.append({
                    "dataset": ds,
                    "policy": pol,
                    "capacity": cap,
                    "oracle_ndcg10": q_ndcg.mean(),
                    "delta_ndcg10": mean_delta_ndcg,
                    "oracle_r1000": q_r.mean(),
                    "delta_r1000": q_r.mean() - base_r1000,
                    "pct_retained_vs_eligible": (mean_delta_ndcg / elig_best_delta) * 100.0,
                    "recall_safe_query_rate": recall_safe_rate,
                    "ranking_safe_query_rate": ranking_safe_rate,
                })

    df_knee = pd.DataFrame(knee_rows)
    knee_csv = os.path.join(output_dir, "pool_capacity_knee_curve.csv")
    df_knee.to_csv(knee_csv, index=False)
    print(f"Wrote {knee_csv}")

    # ----------------------------------------------------
    # 4. Metadata Manifest
    # ----------------------------------------------------
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    manifest = {
        "git_commit": git_hash,
        "datasets": df["dataset"].unique().tolist(),
        "total_records": len(df),
        "total_queries": df.groupby("dataset")["qid"].nunique().to_dict(),
        "weights": WEIGHTS,
        "capacities": CAPACITIES,
        "policies": POLICIES,
        "metric_definitions": {
            "nDCG@10": "Official linear ir_measures.nDCG@10",
            "R@1000": "Official ir_measures.R@1000",
            "ranking_useful": "delta_ndcg10 > 1e-6",
            "ranking_safe": "delta_ndcg10 > 1e-6 and delta_r1000 >= -1e-6",
            "recall_safe": "delta_r1000 > 1e-6 and delta_ndcg10 >= -0.01",
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
