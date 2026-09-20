"""
scripts/run_gate1_oracle_evaluation.py

Phase 2 Gate 1 Candidate Selection Under Uncertainty Evaluation Harness.
Executes counterfactual retrieval for reference universe R_q across 8 corpora,
evaluates proposal channels (Whole-Query BGE, Anchor BGE, Lexical PPMI, RRF Hybrid),
and generates parent candidate audit & sparse transition Parquets.
"""

import os
import sys
import time
import math
import json
import random
import hashlib
import argparse
import subprocess
import gc
import glob
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import torch
import pyarrow.parquet as pq
import pyterrier as pt
import ir_measures
from ir_measures import nDCG, R

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.baselines.pyterrier_harness import init_pyterrier
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer, BGEVocabSidecarManager
from crve.selection.gate1_proposers import (
    WholeQueryBGEProposer,
    AnchorBGEProposer,
    LexicalPPMIProposer,
    RRFHybridProposer,
    BaseProposer,
)
from crve.selection.gate1_metrics import (
    TAU,
    EPSILON,
    DEFAULT_DELTA,
    DEFAULT_RHO,
    TRACKED_CUTOFFS,
    compute_safe_ranking_gain,
    compute_safe_recall_gain,
    classify_document_transition,
)

DEV_DATASETS = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]
EXT_DATASETS = ["fiqa", "scidocs", "arguana", "bright_stackoverflow"]
ALL_DATASETS = DEV_DATASETS + EXT_DATASETS


def get_process_tree_rss_bytes() -> int:
    """Calculates total RSS memory consumed by this process and all children (e.g. JVM)."""
    try:
        parent = psutil.Process()
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total
    except Exception:
        return 0


def get_shard_paths(shards_dir: str, dataset: str, qid: str) -> Tuple[str, str]:
    """Generates shard paths for parent candidate audit and child transitions."""
    q_hash = hashlib.md5(qid.encode("utf-8")).hexdigest()[:12]
    parent_path = os.path.join(shards_dir, f"{dataset}_{q_hash}.parquet")
    child_path = os.path.join(shards_dir, f"{dataset}_{q_hash}_transitions.parquet")
    return parent_path, child_path


def compute_ir_metrics(
    retrieved_df: pd.DataFrame,
    qrels_dict: Dict[str, float],
    qid: str,
    exclusions: Optional[Set[str]] = None,
    k_eval: int = 1000,
) -> Dict[str, float]:
    """Computes linear nDCG@10 and Recall at cutoffs (100, 200, 500, 1000) with exact ir_measures identity."""
    if retrieved_df.empty:
        return {"ndcg10": 0.0, "r100": 0.0, "r200": 0.0, "r500": 0.0, "r1000": 0.0}

    docs = retrieved_df["docno"].values
    if exclusions:
        docs = [str(d) for d in docs if str(d) not in exclusions][:k_eval]
    else:
        docs = [str(d) for d in docs][:k_eval]

    n_rel = sum(1 for r in qrels_dict.values() if r >= 1)
    if n_rel == 0:
        return {"ndcg10": 0.0, "r100": 0.0, "r200": 0.0, "r500": 0.0, "r1000": 0.0}

    ideal_rels = sorted([r for r in qrels_dict.values() if r >= 1], reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    dcg = sum(qrels_dict.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(docs[:10]))
    ndcg10 = float(dcg / idcg) if idcg > 0.0 else 0.0

    r100 = sum(1 for d in docs[:100] if qrels_dict.get(d, 0) >= 1) / n_rel
    r200 = sum(1 for d in docs[:200] if qrels_dict.get(d, 0) >= 1) / n_rel
    r500 = sum(1 for d in docs[:500] if qrels_dict.get(d, 0) >= 1) / n_rel
    r1000 = sum(1 for d in docs[:1000] if qrels_dict.get(d, 0) >= 1) / n_rel

    return {
        "ndcg10": ndcg10,
        "r100": float(r100),
        "r200": float(r200),
        "r500": float(r500),
        "r1000": float(r1000),
    }


def verify_baseline_parity(
    dataset: str,
    sampled_queries: List[Dict[str, Any]],
    analyzer,
    bm25,
    num_to_check: int = 5,
) -> Dict[str, Any]:
    """Asserts exact executable parity between query string and zero-expansion query_toks."""
    checked = 0
    passed = 0
    max_score_diff = 0.0
    queries_checked = min(len(sampled_queries), num_to_check)

    for i in range(queries_checked):
        q = sampled_queries[i]
        qid = str(q["query_id"])
        q_text = str(q.get("question", q.get("query", "")))
        terms, _ = analyzer.analyze(q_text)
        orig_query_toks = analyzer.to_query_toks(terms)
        query_analyzed_str = " ".join(terms)

        df_str = pd.DataFrame([{"qid": qid, "query": query_analyzed_str}])
        res_str = bm25.transform(df_str)

        df_toks = pd.DataFrame([{"qid": qid, "query_toks": orig_query_toks}])
        res_toks = bm25.transform(df_toks)

        checked += 1
        s_docs = list(res_str["docno"]) if not res_str.empty else []
        t_docs = list(res_toks["docno"]) if not res_toks.empty else []
        s_scores = list(res_str["score"]) if not res_str.empty else []
        t_scores = list(res_toks["score"]) if not res_toks.empty else []

        doc_match = (s_docs == t_docs)
        if doc_match and s_scores:
            diff = float(np.max(np.abs(np.array(s_scores) - np.array(t_scores))))
            max_score_diff = max(max_score_diff, diff)
            if diff <= 1e-5:
                passed += 1
        elif doc_match and not s_scores:
            passed += 1

    is_ok = (checked > 0) and (passed == checked)
    status = "PASS" if is_ok else "FAIL"
    print(f"  [Baseline Parity Check] {dataset}: {passed}/{checked} queries matched (max score diff: {max_score_diff:.6e}) -> {status}")
    return {
        "dataset": dataset,
        "checked": checked,
        "passed": passed,
        "max_score_diff": max_score_diff,
        "status": status,
    }


def run_ppmi_preflight_gate(
    dataset: str,
    lex_proposer: LexicalPPMIProposer,
    queries: List[Dict[str, Any]],
    num_queries: int = 5,
) -> Dict[str, Any]:
    """
    Stratified Preflight Gate:
    Evaluates PPMI proposal latency on sample queries. Verifies warm p95 < 20 ms.
    """
    print(f"  [Preflight Gate] Benchmarking LexicalPPMI latency on {num_queries} queries ({dataset})...")
    latencies = []
    # Warmup
    if queries:
        lex_proposer.propose(queries[0], top_k=500)

    for q in queries[:num_queries]:
        t0 = time.perf_counter()
        _ = lex_proposer.propose(q, top_k=500)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    status = "PASS" if p95 < 20.0 else "WARNING_SLOW"
    print(f"  [Preflight Gate] {dataset} LexicalPPMI: p50={p50:.2f}ms, p95={p95:.2f}ms -> {status}")
    return {"dataset": dataset, "p50_ms": p50, "p95_ms": p95, "status": status}


def evaluate_query_gate1(
    dataset: str,
    query_obj: Dict[str, Any],
    index,
    analyzer,
    bm25,
    proposers: Dict[str, BaseProposer],
    rrf_proposer: RRFHybridProposer,
    phase1_cands_for_q: Set[str],
    weights: List[float],
    config_hash: str,
    dataset_hash: str,
    qrels_hash: str,
    chunk_size: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all candidates in R_q for a single query across tested weights:
    Returns (parent_df, transitions_df, query_meta).
    """
    qid = str(query_obj["query_id"])
    q_text = str(query_obj.get("question", query_obj.get("query", "")))
    qrels = query_obj.get("qrels") or {str(d): 1.0 for d in query_obj.get("gold_doc_ids", [])}
    gold_dids = [str(d) for d, r in qrels.items() if r >= 1]
    exclusions = set(str(e) for e in query_obj.get("exclusions", []))

    # 1. Analyze query terms and canonical surfaces for exclusion
    orig_terms, orig_surfs = analyzer.analyze(q_text)
    orig_query_toks = analyzer.to_query_toks(orig_terms)
    query_excluded_terms = set(orig_terms)
    for s in orig_surfs:
        query_excluded_terms.add(str(s).lower())
        query_excluded_terms.add(str(s))

    # 2. Baseline retrieval
    df_base = pd.DataFrame([{"qid": qid, "query_toks": orig_query_toks}])
    base_run = bm25.transform(df_base)
    if exclusions:
        base_run = base_run[~base_run["docno"].astype(str).isin(exclusions)].copy()
    base_run = base_run.sort_values(by="score", ascending=False).reset_index(drop=True)
    base_docs = list(base_run["docno"].astype(str))[:1000]
    base_ranks = {docno: r for r, docno in enumerate(base_docs, 1)}
    base_metrics = compute_ir_metrics(base_run, qrels, qid, exclusions=exclusions)

    # 3. Channel proposals (each top-500)
    channel_proposals = {}
    channel_latencies = {}
    for name, prop in proposers.items():
        t0 = time.perf_counter()
        props = prop.propose(query_obj, top_k=500)
        channel_latencies[name] = (time.perf_counter() - t0) * 1000.0
        # Ensure no candidate lies in query_excluded_terms
        props_clean = [(t, s) for t, s in props if t not in query_excluded_terms]
        channel_proposals[name] = props_clean

    # Proposer ranks map
    proposer_ranks_map = defaultdict(dict)
    for ch_name, ranking in channel_proposals.items():
        for r, (t, _) in enumerate(ranking, 1):
            proposer_ranks_map[t][f"{ch_name}_rank"] = r

    # 4. Form Core Reference Universe R_q^core
    # R_q^core = (T_Phase1 n P_q) U U_m C_{500}^m
    clean_phase1_cands = {t for t in phase1_cands_for_q if t not in query_excluded_terms}
    r_core = set(clean_phase1_cands)
    for ranking in channel_proposals.values():
        r_core.update([t for t, _ in ranking])

    if not r_core:
        return pd.DataFrame(), pd.DataFrame(), {}

    candidates_list = sorted(list(r_core))

    # 5. Build counterfactual variants for PyTerrier execution across weights
    variants_to_eval = []
    for cand in candidates_list:
        for w in weights:
            var_id = f"{qid}___{cand}___{w}"
            q_toks = dict(orig_query_toks)
            q_toks[cand] = q_toks.get(cand, 0.0) + w
            variants_to_eval.append({
                "var_id": var_id,
                "candidate": cand,
                "weight": w,
                "query_toks": q_toks,
            })

    # 6. Batch retrieval execution
    results_by_var = defaultdict(list)
    for i in range(0, len(variants_to_eval), chunk_size):
        chunk = variants_to_eval[i:i + chunk_size]
        df_chunk = pd.DataFrame([{"qid": item["var_id"], "query_toks": item["query_toks"]} for item in chunk])
        res_chunk = bm25.transform(df_chunk)
        if not res_chunk.empty:
            q_arr = res_chunk["qid"].values
            d_arr = res_chunk["docno"].values
            for q_id, doc in zip(q_arr, d_arr):
                results_by_var[q_id].append(str(doc))

    # Precompute ideal DCG@10 and total relevant docs once for this query
    n_rel = sum(1 for r in qrels.values() if r >= 1)
    ideal_rels = sorted([r for r in qrels.values() if r >= 1], reverse=True)[:10]
    idcg10 = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    # 7. Evaluate metrics and log sparse document transitions
    parent_records = []
    t_dataset, t_qid, t_cand, t_weight = [], [], [], []
    t_docid, t_rel, t_conf, t_dhash = [], [], [], []
    t_b_rank, t_e_rank, t_r_delta, t_type = [], [], [], []
    t_ck100, t_ck200, t_ck500, t_ck1000 = [], [], [], []
    t_dk100, t_dk200, t_dk500, t_dk1000 = [], [], [], []

    for item in variants_to_eval:
        var_id = item["var_id"]
        cand = item["candidate"]
        w = item["weight"]

        raw_docs = results_by_var.get(var_id, [])
        if exclusions:
            exp_docs = [d for d in raw_docs if d not in exclusions][:1000]
        else:
            exp_docs = raw_docs[:1000]
        exp_ranks = {docno: r for r, docno in enumerate(exp_docs, 1)}

        # Direct fast calculation of metrics
        dcg = sum(qrels.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(exp_docs[:10]))
        ndcg10_val = float(dcg / idcg10) if idcg10 > 0.0 else 0.0
        r100_val = (sum(1 for d in exp_docs[:100] if qrels.get(d, 0) >= 1) / n_rel) if n_rel > 0 else 0.0
        r200_val = (sum(1 for d in exp_docs[:200] if qrels.get(d, 0) >= 1) / n_rel) if n_rel > 0 else 0.0
        r500_val = (sum(1 for d in exp_docs[:500] if qrels.get(d, 0) >= 1) / n_rel) if n_rel > 0 else 0.0
        r1000_val = (sum(1 for d in exp_docs[:1000] if qrels.get(d, 0) >= 1) / n_rel) if n_rel > 0 else 0.0

        d_ndcg10 = ndcg10_val - base_metrics["ndcg10"]
        d_r100 = r100_val - base_metrics["r100"]
        d_r200 = r200_val - base_metrics["r200"]
        d_r500 = r500_val - base_metrics["r500"]
        d_r1000 = r1000_val - base_metrics["r1000"]

        # Calculate cutoff entry / leaving / net counts
        cutoff_net = {}
        for k in TRACKED_CUTOFFS:
            ent = sum(1 for d in gold_dids if base_ranks.get(d, 9999) > k and exp_ranks.get(d, 9999) <= k)
            lea = sum(1 for d in gold_dids if base_ranks.get(d, 9999) <= k and exp_ranks.get(d, 9999) > k)
            cutoff_net[f"entering_rel_docs_k{k}"] = ent
            cutoff_net[f"leaving_rel_docs_k{k}"] = lea
            cutoff_net[f"net_rel_docs_k{k}"] = ent - lea

        # Sparse transition rows for judged relevant documents
        for did in gold_dids:
            b_rank = base_ranks.get(did, None)
            e_rank = exp_ranks.get(did, None)
            trans = classify_document_transition(b_rank, e_rank, cutoffs=TRACKED_CUTOFFS)
            if trans is not None:
                # Filter out micro-jitter deep in tail (>100) that crosses zero cutoffs to keep memory safe
                if trans["transition_type"] == "moved_within_top1000":
                    any_cut = any(trans.get(f"crossed_k{k}", False) for k in TRACKED_CUTOFFS)
                    if not any_cut and min(trans["baseline_rank"], trans["expanded_rank"]) > 100 and abs(trans["rank_delta"]) < 5:
                        continue

                t_dataset.append(dataset)
                t_qid.append(qid)
                t_cand.append(cand)
                t_weight.append(w)
                t_docid.append(did)
                t_rel.append(int(qrels.get(did, 1)))
                t_conf.append(config_hash)
                t_dhash.append(dataset_hash)
                t_b_rank.append(trans["baseline_rank"])
                t_e_rank.append(trans["expanded_rank"])
                t_r_delta.append(trans["rank_delta"])
                t_type.append(trans["transition_type"])
                t_ck100.append(trans["crossed_k100"])
                t_ck200.append(trans["crossed_k200"])
                t_ck500.append(trans["crossed_k500"])
                t_ck1000.append(trans["crossed_k1000"])
                t_dk100.append(trans["direction_k100"])
                t_dk200.append(trans["direction_k200"])
                t_dk500.append(trans["direction_k500"])
                t_dk1000.append(trans["direction_k1000"])

        p_row = {
            "dataset": dataset,
            "qid": qid,
            "candidate_term": cand,
            "weight": w,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "qrels_hash": qrels_hash,
            "baseline_ndcg10": base_metrics["ndcg10"],
            "expanded_ndcg10": ndcg10_val,
            "delta_ndcg10": d_ndcg10,
            "baseline_r1000": base_metrics["r1000"],
            "expanded_r1000": r1000_val,
            "delta_r100": d_r100,
            "delta_r200": d_r200,
            "delta_r500": d_r500,
            "delta_r1000": d_r1000,
            "is_ranking_helpful_action": (d_ndcg10 > TAU and d_r1000 >= -TAU),
            "is_recall_helpful_k100": (cutoff_net["net_rel_docs_k100"] >= 1 and d_ndcg10 >= -EPSILON),
            "is_recall_helpful_k200": (cutoff_net["net_rel_docs_k200"] >= 1 and d_ndcg10 >= -EPSILON),
            "is_recall_helpful_k500": (cutoff_net["net_rel_docs_k500"] >= 1 and d_ndcg10 >= -EPSILON),
            "is_recall_helpful_k1000": (cutoff_net["net_rel_docs_k1000"] >= 1 and d_ndcg10 >= -EPSILON),
            "in_phase1": cand in clean_phase1_cands,
        }
        p_row.update(cutoff_net)
        # Add proposer rank columns
        p_ranks = proposer_ranks_map.get(cand, {})
        p_row["wq_rank"] = p_ranks.get("WholeQueryBGE_rank", None)
        p_row["anchor_filt_rank"] = p_ranks.get("AnchorBGEFiltered_rank", None)
        p_row["anchor_all_rank"] = p_ranks.get("AnchorBGEAll_rank", None)
        p_row["lex_rank"] = p_ranks.get("LexicalPPMI_rank", None)
        parent_records.append(p_row)

    parent_df = pd.DataFrame(parent_records)
    if t_dataset:
        trans_df = pd.DataFrame({
            "dataset": t_dataset,
            "qid": t_qid,
            "candidate_term": t_cand,
            "weight": t_weight,
            "docid": t_docid,
            "relevance": t_rel,
            "config_hash": t_conf,
            "dataset_hash": t_dhash,
            "baseline_rank": t_b_rank,
            "expanded_rank": t_e_rank,
            "rank_delta": t_r_delta,
            "transition_type": t_type,
            "crossed_k100": t_ck100,
            "crossed_k200": t_ck200,
            "crossed_k500": t_ck500,
            "crossed_k1000": t_ck1000,
            "direction_k100": t_dk100,
            "direction_k200": t_dk200,
            "direction_k500": t_dk500,
            "direction_k1000": t_dk1000,
        })
    else:
        trans_df = pd.DataFrame()

    n_trans = len(t_dataset)
    del t_dataset, t_qid, t_cand, t_weight, t_docid, t_rel, t_conf, t_dhash
    del t_b_rank, t_e_rank, t_r_delta, t_type, t_ck100, t_ck200, t_ck500, t_ck1000
    del t_dk100, t_dk200, t_dk500, t_dk1000

    query_meta = {
        "qid": qid,
        "universe_size": len(candidates_list),
        "total_variants": len(variants_to_eval),
        "transition_count": n_trans,
        "channel_latencies": channel_latencies,
    }
    return parent_df, trans_df, query_meta


def run_dataset_gate1_evaluation(
    dataset: str,
    output_dir: str,
    weights: List[float],
    sample_size: int = 50,
    seed: int = 42,
    chunk_size: int = 100,
    resume: bool = True,
    phase1_audit_df: Optional[pd.DataFrame] = None,
    encoder=None,
    device: str = "cpu",
    config_hash: str = "core_dev_v1",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Runs complete Gate 1 counterfactual evaluation for a single dataset."""
    print(f"\n=======================================================")
    print(f"Starting Gate 1 Evaluation: {dataset} (seed={seed}, sample_size={sample_size})")
    print(f"=======================================================")

    safe_ds = dataset.lower().replace("-", "_")
    idx_path = os.path.abspath(f"data/cache/terrier_indices/{safe_ds}_default/data.properties")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Terrier index not found: {idx_path}")

    index = pt.IndexFactory.of(idx_path)
    analyzer = get_terrier_analyzer()
    num_docs = int(index.getCollectionStatistics().getNumberOfDocuments())

    # Load queries
    all_queries, _ = BenchmarkLoader.load_queries(dataset)
    rng = random.Random(seed)
    if len(all_queries) <= sample_size:
        sampled_queries = list(all_queries)
    else:
        sampled_queries = rng.sample(all_queries, sample_size)

    max_ex = max((len(q.get("exclusions", [])) for q in all_queries), default=0)
    k_fetch = min(num_docs, 1000 + max_ex)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=k_fetch)

    # Parity check
    parity_res = verify_baseline_parity(dataset, sampled_queries, analyzer, bm25, num_to_check=5)

    # Load BGE Sidecar vocabulary & embeddings
    sidecar_mgr = BGEVocabSidecarManager()
    sidecar = sidecar_mgr.build_or_load_sidecar(safe_ds, index)
    retrieval_terms = sidecar["retrieval_terms"]
    display_surfaces = sidecar["display_surfaces"]
    embeddings = sidecar["embeddings"]
    df_vals = sidecar["df"]
    cf_vals = sidecar["cf"]

    # Shared candidate pool P (top 10,000 terms)
    pool_size = min(10000, len(retrieval_terms))
    pool_terms = retrieval_terms[:pool_size]
    pool_embeddings = torch.tensor(embeddings[:pool_size], dtype=torch.float32, device=device)
    surf_to_idx = {str(s).lower(): i for i, s in enumerate(display_surfaces[:pool_size])}

    df_map = {t: int(d) for t, d in zip(retrieval_terms, df_vals)}
    cf_map = {t: int(c) for t, c in zip(retrieval_terms, cf_vals)}
    idf_map = {t: max(0.0, math.log(1.0 + (num_docs - d + 0.5) / (d + 0.5))) for t, d in df_map.items()}

    # Compute dataset fingerprints & hashes
    dataset_hash = hashlib.sha256(f"{dataset}:{num_docs}:{pool_size}".encode("utf-8")).hexdigest()[:12]
    qrels_hash = hashlib.sha256(f"{dataset}:qrels:{len(sampled_queries)}".encode("utf-8")).hexdigest()[:12]

    # Instantiate Proposers
    print(f"  [Proposers] Initializing proposal channels for {dataset}...")
    wq_proposer = WholeQueryBGEProposer(pool_terms, pool_embeddings, encoder, device=device)
    anchor_filt_proposer = AnchorBGEProposer(
        pool_terms, pool_embeddings, surf_to_idx, idf_map, df_map, num_docs, encoder, filter_anchors=True, device=device
    )
    anchor_all_proposer = AnchorBGEProposer(
        pool_terms, pool_embeddings, surf_to_idx, idf_map, df_map, num_docs, encoder, filter_anchors=False, device=device
    )
    lex_proposer = LexicalPPMIProposer(index, pool_terms, idf_map, df_map, num_docs)
    rrf_proposer = RRFHybridProposer(k=60)

    # Preflight gate on Lexical PPMI
    preflight_res = run_ppmi_preflight_gate(dataset, lex_proposer, sampled_queries, num_queries=5)

    proposers = {
        "WholeQueryBGE": wq_proposer,
        "AnchorBGEFiltered": anchor_filt_proposer,
        "AnchorBGEAll": anchor_all_proposer,
        "LexicalPPMI": lex_proposer,
    }

    # Extract Phase 1 candidate terms lookup
    phase1_map = defaultdict(set)
    if phase1_audit_df is not None and not phase1_audit_df.empty:
        ds_phase1 = phase1_audit_df[phase1_audit_df["dataset"] == dataset]
        for _, row in ds_phase1.iterrows():
            phase1_map[str(row["qid"])].add(str(row["candidate_term"]))
        print(f"  [Phase 1 Pool] Loaded {len(phase1_map)} query term sets from Phase 1 audit.")

    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    total_variants = 0
    total_transitions = 0
    t0_retrieval = time.perf_counter()

    for idx_q, q in enumerate(sampled_queries, 1):
        qid = str(q["query_id"])
        parent_shard, trans_shard = get_shard_paths(shards_dir, dataset, qid)

        if resume and os.path.exists(parent_shard) and os.path.exists(trans_shard):
            try:
                p_rows = pq.read_metadata(parent_shard).num_rows
                t_rows = pq.read_metadata(trans_shard).num_rows
                total_variants += p_rows
                total_transitions += t_rows
                print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Resumed from shard ({p_rows} variants, {t_rows} transitions)")
                continue
            except Exception:
                pass

        t0_q = time.perf_counter()
        df_p, df_t, q_meta = evaluate_query_gate1(
            dataset=dataset,
            query_obj=q,
            index=index,
            analyzer=analyzer,
            bm25=bm25,
            proposers=proposers,
            rrf_proposer=rrf_proposer,
            phase1_cands_for_q=phase1_map.get(qid, set()),
            weights=weights,
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            qrels_hash=qrels_hash,
            chunk_size=chunk_size,
        )
        t_q = time.perf_counter() - t0_q

        if not df_p.empty:
            df_p.to_parquet(parent_shard, index=False)
            df_t.to_parquet(trans_shard, index=False)
            total_variants += len(df_p)
            total_transitions += len(df_t)
            v_rate = len(df_p) / max(t_q, 0.001)
            print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Evaluated {len(df_p)} variants ({len(df_t)} trans) in {t_q:.2f}s ({v_rate:.1f} var/s)")
        else:
            print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: No candidates in {t_q:.2f}s")

        del df_p, df_t
        gc.collect()

    total_time = time.perf_counter() - t0_retrieval
    print(f"Completed {dataset}: {total_variants} variants, {total_transitions} transitions in {total_time:.2f}s")

    meta = {
        "dataset": dataset,
        "num_docs": num_docs,
        "pool_size": pool_size,
        "num_queries": len(sampled_queries),
        "total_variants": total_variants,
        "total_transitions": total_transitions,
        "retrieval_time_sec": total_time,
        "baseline_parity": parity_res,
        "ppmi_preflight": preflight_res,
        "dataset_hash": dataset_hash,
        "qrels_hash": qrels_hash,
    }
    return meta


def assemble_shards_to_master(shards_dir: str, is_transitions: bool, output_path: str):
    """
    Streams individual query Parquet shards into a single master Parquet file
    using pyarrow.ParquetWriter. Memory usage is strictly O(1 query shard) <= 10 MB.
    """
    import glob
    import pyarrow as pa
    import pyarrow.parquet as pq

    all_files = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    if is_transitions:
        target_files = [f for f in all_files if f.endswith("_transitions.parquet")]
    else:
        target_files = [f for f in all_files if not f.endswith("_transitions.parquet")]
    if not target_files:
        print(f"No target files found in {shards_dir} (is_transitions={is_transitions})")
        return
    print(f"Streaming {len(target_files)} shards to {output_path} via pyarrow.ParquetWriter...")
    writer = None
    total_rows = 0
    unified_schema = None
    for fpath in target_files:
        try:
            table = pq.read_table(fpath)
            if table.num_rows == 0:
                del table
                continue
            if is_transitions:
                for col_name in ["baseline_rank", "expanded_rank", "rank_delta"]:
                    if col_name in table.column_names:
                        idx = table.column_names.index(col_name)
                        table = table.set_column(idx, col_name, table[col_name].cast(pa.float64()))
            if writer is None:
                unified_schema = table.schema
                writer = pq.ParquetWriter(output_path, unified_schema, compression="zstd")
            elif table.schema != unified_schema:
                table = table.cast(unified_schema)
            writer.write_table(table)
            total_rows += table.num_rows
            del table
        except Exception as e:
            print(f"Warning: error reading {fpath}: {e}")
    if writer is not None:
        writer.close()
    print(f"Successfully assembled {total_rows:,} rows -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Gate 1 Candidate Selection Evaluation Harness")
    parser.add_argument("--datasets", type=str, default=",".join(DEV_DATASETS), help="Comma-separated datasets")
    parser.add_argument("--sample-size", type=int, default=50, help="Query sample size per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--weights", type=str, default="0.05,0.10,0.30,0.50,1.00", help="Weights grid")
    parser.add_argument("--chunk-size", type=int, default=100, help="Retrieval batch chunk size")
    parser.add_argument("--output-dir", type=str, default="results/gate1_selection", help="Output directory")
    parser.add_argument("--phase1-audit", type=str, default="results/pool_isolation_combined/pool_candidate_audit.parquet")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resumption")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight latency checks only")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5", help="BGE model name")
    parser.add_argument("--config-hash", type=str, default="gate1_core_dev_v1", help="Configuration hash")
    return parser.parse_args()


def main():
    args = parse_args()
    init_pyterrier()

    # Lock seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize device & sentence transformer encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder '{args.bge_model}' on {device}...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.bge_model, device=device)

    # Load Phase 1 audit DataFrame if available
    phase1_audit_df = None
    if os.path.exists(args.phase1_audit):
        print(f"Loading Phase 1 candidate terms from {args.phase1_audit}...")
        try:
            phase1_audit_df = pd.read_parquet(args.phase1_audit, columns=["dataset", "qid", "candidate_term"])
        except Exception as e:
            print(f"Warning: could not load Phase 1 parquet: {e}")

    meta_summaries = []
    peak_rss = get_process_tree_rss_bytes()

    for ds in datasets:
        summary = run_dataset_gate1_evaluation(
            dataset=ds,
            output_dir=args.output_dir,
            weights=weights,
            sample_size=args.sample_size,
            seed=args.seed,
            chunk_size=args.chunk_size,
            resume=not args.no_resume,
            phase1_audit_df=phase1_audit_df,
            encoder=encoder,
            device=device,
            config_hash=args.config_hash,
        )
        meta_summaries.append(summary)
        peak_rss = max(peak_rss, get_process_tree_rss_bytes())
        gc.collect()

    # Get git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    peak_gib = peak_rss / (1024 ** 3)

    # Save run manifest
    manifest = {
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": args.config_hash,
        "datasets": datasets,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "weights": weights,
        "peak_rss_gib": round(peak_gib, 3),
        "total_variants": sum(s["total_variants"] for s in meta_summaries),
        "total_transitions": sum(s["total_transitions"] for s in meta_summaries),
        "total_retrieval_time_sec": round(sum(s["retrieval_time_sec"] for s in meta_summaries), 2),
        "dataset_summaries": meta_summaries,
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved Gate 1 run manifest -> {manifest_path}")

    master_parent = os.path.join(args.output_dir, "gate1_candidate_audit.parquet")
    master_trans = os.path.join(args.output_dir, "gate1_sparse_transitions.parquet")

    shards_dir = os.path.join(args.output_dir, "shards")
    assemble_shards_to_master(shards_dir, is_transitions=False, output_path=master_parent)
    assemble_shards_to_master(shards_dir, is_transitions=True, output_path=master_trans)


if __name__ == "__main__":
    main()
