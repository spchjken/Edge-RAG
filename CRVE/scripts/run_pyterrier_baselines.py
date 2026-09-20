"""
scripts/run_pyterrier_baselines.py

CLI Runner for Standard Terrier Default Baselines:
  1. BM25_Default (Standard Terrier BM25)
  2. BM25_RM3_Terrier_Default (Native Java Relevance Model 3 PRF)
  3. BM25_Bo1_Terrier_Default (Native Java Bose-Einstein DFR PRF)
  4. DPH (Terrier Divergence From Randomness Model)
  5. DPH_Bo1_Terrier_Default (DPH with native Java Bose-Einstein Bo1 PRF)
  6. DPH_RM3_Terrier_Default (DPH with native Java Relevance Model 3 RM3 PRF)

Features:
- Full BEIR metric parity (official linear nDCG@10, MAP@100, MRR@10, P@10/100, Strict@10).
- Candidate-funnel diagnostics (R@10-1000, Completeness@100/500/1000, Strict@100/1000, Oracle-nDCG@10).
- Strict BRIGHT pre/post exclusion filtering preventing PRF leakage.
- Isolated PyTerrier single-query API latency benchmarking (P50/P90/P99) and resource tracking.
- Compressed Parquet candidate run persistence (snappy).
"""

import os
import sys
import argparse
import time
import pandas as pd
from typing import List, Optional
import subprocess
import psutil

# Ensure Edge-RAG root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.benchmark_loader import BenchmarkLoader, ALL_BRIGHT_DOMAINS
from evaluation.baselines.pyterrier_harness import PyTerrierIndexManager, PyTerrierBaselineHarness, init_pyterrier

ALL_BEIR_DATASETS = [
    "scifact",
    "nfcorpus",
    "fiqa",
    "arguana",
    "scidocs",
    "quora",
    "hotpotqa",
    "trec_covid",
    "webis_touche2020",
    "dbpedia_entity",
    "nq",
    "climate_fever",
    "fever",
]

ALL_BRIGHT_DATASETS = [f"bright_{d}" for d in ALL_BRIGHT_DOMAINS]

PIPELINE_SUITES = {
    "default4": [
        "BM25_Default",
        "BM25_RM3_Terrier_Default",
        "BM25_Bo1_Terrier_Default",
        "DPH",
    ],
    "dph_prf": [
        "DPH_Bo1_Terrier_Default",
        "DPH_RM3_Terrier_Default",
    ],
    "all6": [
        "BM25_Default",
        "BM25_RM3_Terrier_Default",
        "BM25_Bo1_Terrier_Default",
        "DPH",
        "DPH_Bo1_Terrier_Default",
        "DPH_RM3_Terrier_Default",
    ],
    "neural": [
        "BGE_Small_Dense",
        "SPLADE_v3_PISA",
    ],
    "all8": [
        "BM25_Default",
        "BM25_RM3_Terrier_Default",
        "BM25_Bo1_Terrier_Default",
        "DPH",
        "DPH_Bo1_Terrier_Default",
        "DPH_RM3_Terrier_Default",
        "BGE_Small_Dense",
        "SPLADE_v3_PISA",
    ],
}


def generate_summary_markdown(csv_path: str, output_dir: str):
    """Generates markdown summary tables from the results CSV."""
    if not os.path.exists(csv_path):
        return
    try:
        final_df = pd.read_csv(csv_path)
        summary_md_path = os.path.join(output_dir, "pyterrier_baselines_summary.md")
        with open(summary_md_path, "w") as f:
            f.write("# PyTerrier Baseline Evaluation Summary (Default Baselines Suite)\n\n")
            f.write(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("### 1. Headline Retrieval Quality (Linear nDCG@10, Supplemental Exp-nDCG, MRR@10, P@10)\n\n")
            q_cols = [c for c in ["dataset", "pipeline", "ndcg_10", "exp_ndcg_10", "ndcg_100", "map_100", "mrr_10", "p_10", "strict_10"] if c in final_df.columns]
            f.write(final_df[q_cols].to_markdown(index=False))
            
            f.write("\n\n### 2. Candidate Funnel Ceiling Diagnostics (Recall@K, Completeness@K, Oracle-nDCG@10)\n\n")
            fn_cols = [c for c in ["dataset", "pipeline", "recall_100", "recall_500", "recall_1000", "completeness_100", "completeness_500", "completeness_1000", "oracle_ndcg_10"] if c in final_df.columns]
            f.write(final_df[fn_cols].to_markdown(index=False))

            f.write("\n\n### 3. Retrieval Efficiency & Resource Footprint\n\n")
            eff_cols = [c for c in ["dataset", "pipeline", "retrieval_api_p50_ms", "retrieval_api_p99_ms", "batch_throughput_qps", "harness_per_query_ms", "index_disk_mb", "host_ram_peak_mb", "index_build_s", "cache_load_s"] if c in final_df.columns]
            f.write(final_df[eff_cols].to_markdown(index=False))
            f.write("\n")
        print(f"\n[Summary] Complete markdown summary written to {summary_md_path}")
    except Exception as e:
        print(f"[Warning] Failed generating summary markdown: {e}")


def run_dataset(
    dataset_name: str,
    index_mgr: PyTerrierIndexManager,
    output_dir: str,
    pipelines: Optional[List[str]] = None,
    max_queries: int = 0,
    overwrite_index: bool = False,
    chunk_size: int = 200,
    save_runs_dir: str = "data/cache/runs",
) -> pd.DataFrame:
    """Runs evaluation for a single dataset with zero in-memory corpus footprint."""
    print(f"\n=======================================================")
    print(f"Loading Dataset: {dataset_name}")
    print(f"=======================================================")

    # 0. Pre-flight Memory Safety Check
    avail_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
    print(f"[Pre-flight Memory] Host available RAM: {avail_mb:.0f} MiB")
    if avail_mb < 1500:
        print(f"[Pre-flight Memory Warning] Free RAM is below 1500 MiB ({avail_mb:.0f} MiB). Running gc...")
        import gc
        gc.collect()

    # 1. Load Queries and Metadata (zero in-memory corpus)
    queries, stats = BenchmarkLoader.load_queries(dataset_name)
    num_docs = BenchmarkLoader.get_corpus_doc_count(dataset_name)
    print(f"Loaded {len(queries):,} queries (corpus size: {num_docs:,} documents).")
    print(f"Dataset stats: {stats}")

    if max_queries > 0 and len(queries) > max_queries:
        print(f"[Sampling] Capping queries from {len(queries)} to {max_queries} for test run.")
        queries = queries[:max_queries]

    # 2. Evaluate Pipeline Groups Sequentially with Explicit Memory Reclamation
    all_res_dfs = []
    target_pipelines = list(pipelines) if pipelines else ["BM25_Default", "BM25_RM3_Terrier_Default", "BM25_Bo1_Terrier_Default", "DPH"]

    # Group A: Classical Terrier Pipelines
    terrier_pipelines = [p for p in target_pipelines if p not in ("BGE_Small_Dense", "SPLADE_v3_PISA")]
    if terrier_pipelines:
        print(f"\n[Group: Terrier] Evaluating {terrier_pipelines}...", flush=True)
        terrier_dict = index_mgr.build_or_load_indices(
            dataset_name=dataset_name,
            corpus_docs=None,
            overwrite=overwrite_index,
        )
        harness = PyTerrierBaselineHarness(terrier_dict)
        t_res = harness.run_all(
            queries,
            chunk_size=chunk_size,
            save_runs_dir=save_runs_dir,
            dataset_name=dataset_name,
            pipelines=terrier_pipelines,
        )
        all_res_dfs.append(t_res)
        del harness, terrier_dict
        import gc
        gc.collect()

    # Group B: BGE Dense Retrieval
    if "BGE_Small_Dense" in target_pipelines:
        print(f"\n[Group: BGE Dense] Evaluating BGE_Small_Dense...", flush=True)
        dense_dict = index_mgr.build_or_load_dense_index(
            dataset_name=dataset_name,
            overwrite=overwrite_index,
        )
        harness = PyTerrierBaselineHarness(dense_dict)
        d_res = harness.run_all(
            queries,
            chunk_size=chunk_size,
            save_runs_dir=save_runs_dir,
            dataset_name=dataset_name,
            pipelines=["BGE_Small_Dense"],
        )
        all_res_dfs.append(d_res)
        del harness, dense_dict
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Group C: SPLADE Neural Sparse Retrieval
    if "SPLADE_v3_PISA" in target_pipelines:
        print(f"\n[Group: SPLADE PISA] Evaluating SPLADE_v3_PISA...", flush=True)
        splade_dict = index_mgr.build_or_load_splade_index(
            dataset_name=dataset_name,
            overwrite=overwrite_index,
        )
        harness = PyTerrierBaselineHarness(splade_dict)
        s_res = harness.run_all(
            queries,
            chunk_size=chunk_size,
            save_runs_dir=save_runs_dir,
            dataset_name=dataset_name,
            pipelines=["SPLADE_v3_PISA"],
        )
        all_res_dfs.append(s_res)
        del harness, splade_dict
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    df_res = pd.concat(all_res_dfs, ignore_index=True) if all_res_dfs else pd.DataFrame()
    df_res["dataset"] = dataset_name
    df_res["num_docs"] = num_docs
    df_res["num_queries"] = len(queries)

    # Logical column ordering: Quality -> Candidate Funnel -> Latency & Throughput -> Resource Footprint
    quality_cols = ["ndcg_10", "exp_ndcg_10", "ndcg_50", "ndcg_100", "map_100", "mrr_10", "r_10", "r_50", "p_10", "p_100", "strict_10", "strict_50"]
    funnel_cols = ["recall_100", "recall_200", "recall_500", "recall_1000", "completeness_100", "completeness_500", "completeness_1000", "strict_100", "strict_1000", "oracle_ndcg_10"]
    efficiency_cols = ["retrieval_api_p50_ms", "retrieval_api_p90_ms", "retrieval_api_p99_ms", "retrieval_api_mean_ms", "batch_throughput_qps", "harness_per_query_ms"]
    efficiency_cols += ["index_disk_mb", "host_ram_peak_mb", "index_build_s", "cache_load_s"]
    meta_cols = ["dataset", "pipeline"]
    
    all_known = set(quality_cols + funnel_cols + efficiency_cols + meta_cols)
    extra_cols = [c for c in df_res.columns if c not in all_known]
    ordered_cols = [c for c in quality_cols + funnel_cols + efficiency_cols + meta_cols if c in df_res.columns] + extra_cols
    df_res = df_res[ordered_cols]

    # Display Results
    print(f"\n--- Results for {dataset_name} ---")
    display_cols = ["pipeline", "ndcg_10", "exp_ndcg_10", "recall_100", "recall_1000", "completeness_1000", "retrieval_api_p50_ms", "batch_throughput_qps"]
    available_display = [c for c in display_cols if c in df_res.columns]
    print(df_res[available_display].to_string(index=False))

    return df_res


def main():
    parser = argparse.ArgumentParser(description="PyTerrier Default Baseline Evaluation Harness")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset name to run")
    parser.add_argument("--all-beir", action="store_true", help="Run across all 13 canonical BEIR datasets")
    parser.add_argument("--all-bright", action="store_true", help="Run across all 12 BRIGHT reasoning datasets")
    parser.add_argument(
        "--pipelines",
        type=str,
        default="default4",
        choices=["default4", "dph_prf", "all6", "neural", "all8"],
        help="Pipelines suite to evaluate (default4: 4 standard baselines; dph_prf: DPH_Bo1 + DPH_RM3; all6: all 6 baselines; neural: BGE_Small_Dense + SPLADE_v3_PISA; all8: all 8 baselines)",
    )
    parser.add_argument("--cache-dir", type=str, default="data/cache/terrier_indices", help="Directory for Terrier indices")
    parser.add_argument("--runs-dir", type=str, default="data/cache/runs", help="Directory for candidate parquet runs")
    parser.add_argument("--output-dir", type=str, default="results/pyterrier_baselines", help="Directory for evaluation outputs")
    parser.add_argument("--max-queries", type=int, default=0, help="Cap queries for quick testing (0 for all)")
    parser.add_argument("--chunk-size", type=int, default=200, help="Query chunk size to prevent memory spikes (default: 200)")
    parser.add_argument("--mem", type=int, default=3072, help="JVM heap memory limit in MB (default: 3072)")
    parser.add_argument("--no-isolate", action="store_true", help="Disable subprocess isolation")
    parser.add_argument("--overwrite-index", action="store_true", help="Force rebuild indices")
    parser.add_argument("--overwrite-results", action="store_true", help="Force re-evaluation of already completed datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    target_datasets = []
    if args.all_beir and args.all_bright:
        target_datasets = ALL_BEIR_DATASETS + ALL_BRIGHT_DATASETS
    elif args.all_beir:
        target_datasets = ALL_BEIR_DATASETS
    elif args.all_bright:
        target_datasets = ALL_BRIGHT_DATASETS
    elif args.dataset:
        target_datasets = [args.dataset]
    else:
        print("Please specify --dataset <name>, --all-beir, or --all-bright.")
        sys.exit(1)

    selected_pipelines = PIPELINE_SUITES[args.pipelines]
    csv_path = os.path.join(args.output_dir, "pyterrier_baselines_results.csv")

    # =========================================================================
    # SUPERVISOR MODE: Spawn isolated subprocess per dataset to prevent memory accumulation
    # =========================================================================
    if len(target_datasets) > 1 and not args.no_isolate:
        print(f"\n=======================================================")
        print(f"[Supervisor] Launching isolated worker execution for {len(target_datasets)} datasets")
        print(f"[Supervisor] Pipelines: {selected_pipelines}")
        print(f"[Supervisor] JVM heap cap: {args.mem} MB | Target CSV: {csv_path}")
        print(f"=======================================================")

        for idx, d_name in enumerate(target_datasets, 1):
            # Check if dataset is already completed for all requested pipelines
            if os.path.exists(csv_path) and not args.overwrite_results:
                try:
                    existing_df = pd.read_csv(csv_path)
                    if "pipeline" in existing_df.columns:
                        needed_pipelines = set(selected_pipelines)
                        norm_d = d_name.lower().replace("-", "_").replace("_doc_level", "")
                        d_rows = existing_df[
                            existing_df["dataset"].astype(str).str.lower().str.replace("-", "_").str.replace("_doc_level", "") == norm_d
                        ]
                        done_pipelines = set(d_rows["pipeline"].dropna().unique())
                        if needed_pipelines.issubset(done_pipelines):
                            print(f"[{idx}/{len(target_datasets)}] [Supervisor Skip] Dataset '{d_name}' already evaluated for {list(needed_pipelines)}. Skipping...")
                            continue
                except Exception:
                    pass

            cmd = [
                sys.executable, "-u", os.path.abspath(__file__),
                "--dataset", d_name,
                "--pipelines", args.pipelines,
                "--cache-dir", args.cache_dir,
                "--runs-dir", args.runs_dir,
                "--output-dir", args.output_dir,
                "--mem", str(args.mem),
                "--chunk-size", str(args.chunk_size),
                "--no-isolate",
            ]
            if args.max_queries > 0:
                cmd.extend(["--max-queries", str(args.max_queries)])
            if args.overwrite_index:
                cmd.append("--overwrite-index")
            if args.overwrite_results:
                cmd.append("--overwrite-results")

            print(f"\n[{idx}/{len(target_datasets)}] >>> [Supervisor] Spawning worker for: {d_name}...")
            sub_res = subprocess.run(cmd)
            if sub_res.returncode != 0:
                print(f"[{idx}/{len(target_datasets)}] [Supervisor ERROR] Worker for '{d_name}' exited with code {sub_res.returncode}")

        generate_summary_markdown(csv_path, args.output_dir)
        print(f"\n[Supervisor] All {len(target_datasets)} dataset evaluations processed.")
        return

    # =========================================================================
    # WORKER MODE: Evaluate dataset(s) within current process
    # =========================================================================
    init_pyterrier(mem=args.mem)
    index_mgr = PyTerrierIndexManager(cache_dir=args.cache_dir)

    for d_name in target_datasets:
        try:
            res_df = run_dataset(
                dataset_name=d_name,
                index_mgr=index_mgr,
                output_dir=args.output_dir,
                pipelines=selected_pipelines,
                max_queries=args.max_queries,
                overwrite_index=args.overwrite_index,
                chunk_size=args.chunk_size,
                save_runs_dir=args.runs_dir,
            )

            # Non-destructive merge and deduplicate
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                comb_df = pd.concat([existing_df, res_df], ignore_index=True).drop_duplicates(
                    subset=["dataset", "pipeline"], keep="last"
                )
            else:
                comb_df = res_df

            comb_df.to_csv(csv_path, index=False)
            print(f"[Saved] Incremental results saved to {csv_path}")

        except Exception as e:
            print(f"[ERROR] Failed evaluating dataset {d_name}: {e}")
            import traceback
            traceback.print_exc()

    generate_summary_markdown(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
