"""
scripts/run_pyterrier_qe_baselines.py

Dedicated evaluation runner for sparse query-expansion baselines:
1. BGE_Vocab_QE (Frozen-encoder static vocabulary expansion)
2. LLM_Q2E_ZS (Local-LLM zero-shot keyword expansion via Qwen3.5:4b)

Execution Order Guarantee:
- BGE_Vocab_QE runs across all datasets first.
- LLM_Q2E_ZS runs last across all datasets (minimizing model load/unload overhead).
- LLM automatically unloaded from memory (keep_alive: 0) after generation.

Telemetry & Results:
- Appends to results/pyterrier_baselines/pyterrier_qe_results.csv (keeping main baseline CSV intact).
"""

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
import pyterrier as pt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.baselines.pyterrier_harness import (
    PyTerrierIndexManager,
    init_pyterrier,
    sanitize_default_query,
)
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import (
    TerrierQueryAnalyzer,
    BGEVocabSidecarManager,
    LLMQ2ESidecarManager,
    DEFAULT_QE_CONFIG,
)
from scripts.enrich_baseline_index_stats import (
    TERRIER_BUILD_TIMES,
    SIDECAR_PREPARE_TIMES,
    get_dir_size_mb,
)

ALL_25_DATASETS = [
    # Compact Tier (6)
    "scifact", "nfcorpus", "arguana", "bright_pony", "bright_theoremqa_theorems", "scidocs",
    # Medium Tier (6)
    "bright_economics", "bright_psychology", "bright_biology", "fiqa", "bright_sustainable_living", "bright_robotics",
    # Large Tier (8)
    "bright_stackoverflow", "bright_earth_science", "trec_covid", "bright_aops", "bright_theoremqa_questions",
    "webis_touche2020", "bright_leetcode", "quora",
    # Massive Tier (5)
    "nq", "dbpedia_entity", "hotpotqa", "climate_fever", "fever",
]

RESULTS_CSV_PATH = os.path.join(BASE_DIR, "results", "pyterrier_baselines", "pyterrier_qe_results.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "pyterrier_qe.yaml")


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return DEFAULT_QE_CONFIG


def is_already_done(csv_path: str, dataset: str, pipeline: str) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns or "pipeline" not in df.columns:
            return False
        norm_d = dataset.lower().replace("-", "_")
        sub = df[(df["dataset"].str.lower().str.replace("-", "_") == norm_d) & (df["pipeline"] == pipeline)]
        return len(sub) > 0
    except Exception:
        return False


def run_evaluation(
    datasets: List[str],
    pipelines: List[str],
    seed: int = 42,
    chunk_size: int = 200,
    resume: bool = True,
    preflight_only: bool = False,
):
    init_pyterrier(mem=3072)
    config = load_config()

    # Preflight Check: verify Ollama model availability if LLM_Q2E_ZS or DPH_LLM_Q2E_ZS is in pipelines
    if any(p in pipelines for p in ("LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS")):
        endpoint = config.get("llm_q2e_zs", {}).get("endpoint", "http://localhost:11434/api/generate")
        tag = config.get("llm_q2e_zs", {}).get("tag", "qwen3.5:4b")
        try:
            import urllib.request
            import json
            tags_url = endpoint.replace("/api/generate", "/api/tags")
            data = json.loads(urllib.request.urlopen(tags_url, timeout=5).read().decode())
            installed = [m["name"] for m in data.get("models", [])]
            if tag not in installed and f"{tag}:latest" not in installed:
                raise RuntimeError(f"Ollama model '{tag}' is not installed! Available: {installed}")
            print(f"[Preflight] Ollama verified: {tag} is installed and available.")
        except Exception as e:
            print(f"[Preflight ERROR] Ollama model check failed: {e}")
            if preflight_only:
                sys.exit(1)
            raise

    if preflight_only:
        print("[Preflight] Preflight checks passed successfully.")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    runs_dir = os.path.join(BASE_DIR, "data", "cache", "runs")

    # Enforce Execution Order: BGE QE first, then LLM QE last
    ordered_pipelines = []
    for p in ["BGE_Vocab_QE", "DPH_BGE_Vocab_QE", "LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS"]:
        if p in pipelines:
            ordered_pipelines.append(p)
    for p in pipelines:
        if p not in ordered_pipelines:
            ordered_pipelines.append(p)

    print("=" * 70)
    print(">>> PyTerrier Sparse Query-Expansion Evaluation Runner")
    print(f">>> Target Datasets ({len(datasets)}): {datasets}")
    print(f">>> Target Pipelines (in execution order): {ordered_pipelines}")
    print(f">>> Results CSV: {RESULTS_CSV_PATH}")
    print("=" * 70)

    sidecar_mgr = BGEVocabSidecarManager(config.get("bge_vocab_qe"))
    llm_rewriter = None
    if any(p in ordered_pipelines for p in ("LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS")):
        llm_rewriter = LLMQEKeywordRewriter(config.get("llm_q2e_zs"))

    for p_name in ordered_pipelines:
        print(f"\n{'='*70}")
        print(f">>> STARTING PIPELINE BATCH: {p_name}")
        print(f"{'='*70}")

        for idx, ds in enumerate(datasets, 1):
            print(f"\n[{idx}/{len(datasets)}] Evaluating {ds} | Pipeline: {p_name}")

            if resume and is_already_done(RESULTS_CSV_PATH, ds, p_name):
                print(f"  [SKIP] '{ds}' already evaluated for {p_name}.")
                continue

            # Load Terrier Index
            safe_ds = ds.lower().replace("-", "_")
            index_path = os.path.join(BASE_DIR, "data", "cache", "terrier_indices", f"{safe_ds}_default", "data.properties")
            if not os.path.exists(index_path):
                print(f"  [ERROR] Terrier index not found: {index_path}")
                continue

            index = pt.IndexFactory.of(index_path)

            # Load Queries & Qrels
            queries, stats = BenchmarkLoader.load_queries(ds)
            print(f"  Loaded {len(queries)} queries for {ds}.")

            harness = PyTerrierBaselineHarness(
                index_dict={"index_default": index}
            )

            # Setup QE Rewriter
            qe_meta = {}
            if p_name in ("BGE_Vocab_QE", "DPH_BGE_Vocab_QE"):
                sidecar = sidecar_mgr.build_or_load_sidecar(ds, index)
                rewriter = BGEVocabQERewriter(sidecar, config.get("bge_vocab_qe"))
                harness.bge_vocab_rewriter = rewriter
                qe_meta["qe_prepare_s"] = sidecar["timing"].get("qe_prepare_s", 0.0)
                qe_meta["qe_sidecar_mb"] = sidecar.get("sidecar_mb", 0.0)
                qe_meta["qe_cache_hit"] = sidecar["timing"].get("qe_cache_hit", False)

            elif p_name in ("LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS"):
                harness.llm_qe_rewriter = llm_rewriter

            # Format qrels for ir_measures
            import ir_measures
            qrels_ir = []
            gold_map = {}
            for q in queries:
                qid = str(q["query_id"])
                golds = q.get("qrels") or {str(did): 1.0 for did in q.get("gold_doc_ids", [])}
                gold_map[qid] = golds
                for did, rel in golds.items():
                    qrels_ir.append(ir_measures.Qrel(qid, str(did), int(rel)))

            # Evaluate
            metrics = harness.evaluate_pipeline(
                pipeline_name=p_name,
                queries=queries,
                qrels_ir=qrels_ir,
                gold_map=gold_map,
                chunk_size=chunk_size,
                save_runs_dir=runs_dir,
                dataset_name=ds,
            )

            # Record result with strict column parity matching pyterrier_baselines_results.csv
            num_docs = BenchmarkLoader.get_corpus_doc_count(ds)
            num_queries = len(queries)

            # Exact index stats
            t_dir = os.path.join(BASE_DIR, "data", "cache", "terrier_indices", f"{safe_ds}_default")
            t_disk_mb = get_dir_size_mb(t_dir)
            t_build_s = TERRIER_BUILD_TIMES.get(safe_ds, 0.0)

            v_dir = os.path.join(BASE_DIR, "data", "cache", "qe_vocab", safe_ds)
            v_disk_mb = get_dir_size_mb(v_dir)
            v_build_s = SIDECAR_PREPARE_TIMES.get(safe_ds, 0.0)

            if p_name in ("BGE_Vocab_QE", "DPH_BGE_Vocab_QE"):
                row_build_s = round(t_build_s + v_build_s, 2)
                row_disk_mb = round(t_disk_mb + v_disk_mb, 2)
            else:
                row_build_s = round(t_build_s, 2)
                row_disk_mb = round(t_disk_mb, 2)

            res_row = {
                "dataset": ds,
                "pipeline": p_name,
                "ndcg_10": round(metrics.get("ndcg_10", 0.0), 4),
                "exp_ndcg_10": round(metrics.get("exp_ndcg_10", 0.0), 4),
                "ndcg_100": round(metrics.get("ndcg_100", 0.0), 4),
                "map_100": round(metrics.get("map_100", 0.0), 4),
                "mrr_10": round(metrics.get("mrr_10", 0.0), 4),
                "p_10": round(metrics.get("p_10", 0.0), 4),
                "p_100": round(metrics.get("p_100", 0.0), 4),
                "strict_10": round(metrics.get("strict_10", 0.0), 4),
                "recall_10": round(metrics.get("recall_10", 0.0), 4),
                "recall_50": round(metrics.get("recall_50", 0.0), 4),
                "recall_100": round(metrics.get("recall_100", 0.0), 4),
                "recall_200": round(metrics.get("recall_200", 0.0), 4),
                "recall_500": round(metrics.get("recall_500", 0.0), 4),
                "recall_1000": round(metrics.get("recall_1000", 0.0), 4),
                "completeness_100": round(metrics.get("completeness_100", 0.0), 4),
                "completeness_500": round(metrics.get("completeness_500", 0.0), 4),
                "completeness_1000": round(metrics.get("completeness_1000", 0.0), 4),
                "strict_100": round(metrics.get("strict_100", 0.0), 4),
                "strict_1000": round(metrics.get("strict_1000", 0.0), 4),
                "oracle_ndcg_10": round(metrics.get("oracle_ndcg_10", 0.0), 4),
                "retrieval_api_p50_ms": round(metrics.get("retrieval_api_p50_ms", 0.0), 2),
                "retrieval_api_p90_ms": round(metrics.get("retrieval_api_p90_ms", 0.0), 2),
                "retrieval_api_p99_ms": round(metrics.get("retrieval_api_p99_ms", 0.0), 2),
                "retrieval_api_mean_ms": round(metrics.get("retrieval_api_mean_ms", 0.0), 2),
                "batch_throughput_qps": round(metrics.get("batch_throughput_qps", 0.0), 2),
                "harness_per_query_ms": round(metrics.get("harness_per_query_ms", 0.0), 2),
                "num_docs": num_docs,
                "num_queries": num_queries,
                "index_disk_mb": row_disk_mb,
                "host_ram_peak_mb": round(metrics.get("host_ram_peak_mb", 0.0), 2),
                "index_build_s": row_build_s,
                "cache_load_s": round(metrics.get("cache_load_s", 0.0), 2),
                "ndcg_50": round(metrics.get("ndcg_50", 0.0), 4),
                "strict_50": round(metrics.get("strict_50", 0.0), 4),
            }

            # Append to dedicated CSV with strict schema parity
            csv_columns = list(res_row.keys())
            df_res = pd.DataFrame([res_row])[csv_columns]
            header = not os.path.exists(RESULTS_CSV_PATH)
            df_res.to_csv(RESULTS_CSV_PATH, mode="a", index=False, header=header)
            print(f"  [Saved] Appended result for {ds} | {p_name} to {RESULTS_CSV_PATH}")
            print(f"  --> nDCG@10: {res_row['ndcg_10']:.4f} | Recall@1000: {res_row['recall_1000']:.4f} | Online E2E P50: {res_row['retrieval_api_p50_ms']:.2f}ms")
            if p_name in ("LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS") and llm_rewriter is not None:
                g_stats = llm_rewriter.get_generation_stats(queries)
                if g_stats.get("llm_p50_ms", 0) > 0:
                    print(f"  --> [LLM Telemetry] LLM Gen P50: {g_stats['llm_p50_ms']:.2f}ms (Mean: {g_stats['llm_mean_ms']:.2f}ms) | Total Online E2E P50: {res_row['retrieval_api_p50_ms']:.2f}ms (cached: {g_stats['llm_cached_count']}/{len(queries)})")

        # Cleanup after pipeline batch
        if p_name in ("BGE_Vocab_QE", "DPH_BGE_Vocab_QE"):
            try:
                import torch, gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"[{p_name}] PyTorch VRAM and cache collected.", flush=True)
            except Exception:
                pass
        if p_name in ("LLM_Q2E_ZS", "DPH_LLM_Q2E_ZS") and p_name == ordered_pipelines[-1]:
            if llm_rewriter is not None:
                llm_rewriter.unload_model()

    print("\n" + "=" * 70)
    print(">>> Sparse Query-Expansion Evaluation Completed Successfully! <<<")
    print(f">>> Results saved at: {RESULTS_CSV_PATH}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PyTerrier Sparse QE Baseline Runner")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "bright_pony"], help="Specific datasets to evaluate")
    parser.add_argument("--all-datasets", action="store_true", help="Evaluate all 25 benchmark datasets")
    parser.add_argument("--pipelines", nargs="+", default=["BGE_Vocab_QE", "LLM_Q2E_ZS", "DPH_BGE_Vocab_QE", "DPH_LLM_Q2E_ZS"], help="Pipelines to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=200, help="Query chunk size")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip already evaluated datasets")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight environment checks and exit")
    args = parser.parse_args()

    target_datasets = ALL_25_DATASETS if args.all_datasets else args.datasets
    run_evaluation(
        datasets=target_datasets,
        pipelines=args.pipelines,
        seed=args.seed,
        chunk_size=args.chunk_size,
        resume=args.resume,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
