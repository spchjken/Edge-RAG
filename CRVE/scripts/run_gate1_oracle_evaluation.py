"""
scripts/run_gate1_oracle_evaluation.py

Phase 2 Gate 1 Candidate Selection Under Uncertainty Evaluation Harness.
Executes counterfactual retrieval for reference universe R_q across datasets,
evaluates proposal channels (Whole-Query BGE, Anchor BGE, Lexical PPMI, RRF Hybrid),
and generates candidate audit, cutoff entries, and query status Parquets.
"""

import os
import sys
import time
import math
import json
import random
import hashlib
import argparse
import gc
import glob
import shutil
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyterrier as pt
import yaml

# Ensure project root and CRVE/src are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from evaluation.baselines.pyterrier_harness import init_pyterrier
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
from crve.selection.gate1_sidecars import Gate1SidecarManager
from crve.selection.gate1_proposers import (
    WholeQueryBGEProposer,
    AnchorBGEProposer,
    LexicalPPMIProposer,
    PPMISidecarProposer,
    SparseLexicalContextProposer,
    AcronymDefinitionRescueProposer,
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
)

DEV_DATASETS = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]


def compute_rbo(s: List[str], t: List[str], p: float = 0.98, k: int = 500) -> float:
    """Computes Rank-Biased Overlap (RBO) between two ranked candidate lists."""
    if not s or not t:
        return 0.0
    s_set, t_set = set(), set()
    rbo_sum = 0.0
    max_depth = min(max(len(s), len(t)), k)
    for d in range(1, max_depth + 1):
        if d <= len(s):
            s_set.add(s[d - 1])
        if d <= len(t):
            t_set.add(t[d - 1])
        overlap = len(s_set.intersection(t_set))
        agreement = overlap / float(d)
        rbo_sum += (p ** (d - 1)) * agreement
    return (1.0 - p) * rbo_sum


class MemorySafetyError(RuntimeError):
    """Raised when process-tree memory exceeds the hard safety cap."""
    pass


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


def check_memory_watchdog(warn_rss_gib: float = 8.0, abort_rss_gib: float = 12.0):
    """Fail-closed memory watchdog: triggers gc at warn threshold, aborts at hard cap."""
    rss_gib = get_process_tree_rss_bytes() / (1024 ** 3)
    if rss_gib >= abort_rss_gib:
        raise MemorySafetyError(
            f"FATAL: Process-tree RSS ({rss_gib:.2f} GiB) exceeded hard cap of {abort_rss_gib} GiB! Aborting to prevent OOM."
        )
    elif rss_gib >= warn_rss_gib:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_shard_paths(shards_dir: str, dataset: str, qid: str) -> Tuple[str, str, str]:
    """Generates shard paths for parent candidate audit, cutoff entries, and query status."""
    q_hash = hashlib.md5(qid.encode("utf-8")).hexdigest()[:12]
    parent_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}.parquet")
    cutoff_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_cutoff_entries.parquet")
    status_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_status.parquet")
    return parent_shard, cutoff_shard, status_shard


def compute_ir_metrics(
    doc_scores: List[Tuple[str, float]],
    qrels: Dict[str, float],
    qid: str,
    exclusions: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Computes baseline nDCG@10 and R@K for unexpanded query."""
    if exclusions:
        filtered = [d for d, _ in doc_scores if d not in exclusions]
    else:
        filtered = [d for d, _ in doc_scores]

    ideal_rels = sorted([r for r in qrels.values() if r >= 1], reverse=True)[:10]
    idcg10 = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    dcg10 = sum(qrels.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(filtered[:10]))
    ndcg10 = float(dcg10 / idcg10) if idcg10 > 0.0 else 0.0

    n_rel = sum(1 for r in qrels.values() if r >= 1)
    if n_rel == 0:
        return {"ndcg10": ndcg10, "r100": 0.0, "r200": 0.0, "r500": 0.0, "r1000": 0.0}

    r100 = float(sum(1 for d in filtered[:100] if qrels.get(d, 0) >= 1) / n_rel)
    r200 = float(sum(1 for d in filtered[:200] if qrels.get(d, 0) >= 1) / n_rel)
    r500 = float(sum(1 for d in filtered[:500] if qrels.get(d, 0) >= 1) / n_rel)
    r1000 = float(sum(1 for d in filtered[:1000] if qrels.get(d, 0) >= 1) / n_rel)
    return {"ndcg10": ndcg10, "r100": r100, "r200": r200, "r500": r500, "r1000": r1000}


def evaluate_query_gate1(
    dataset: str,
    query_obj: Dict[str, Any],
    index,
    analyzer,
    bm25,
    proposers: Dict[str, BaseProposer],
    rrf_proposer: RRFHybridProposer,
    pool_set: Set[str],
    phase1_cands_for_q: Set[str],
    weights: List[float],
    thresholds: Dict[str, float],
    config_hash: str = "core_dev_v1",
    dataset_hash: str = "",
    qrels_hash: str = "",
    pool_hash: str = "",
    chunk_size: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all candidate actions for a single query.
    Emits parent candidate audit, cutoff entries, and query status.
    """
    check_memory_watchdog()
    qid = str(query_obj.get("query_id", query_obj.get("qid", "")))
    q_text = str(query_obj.get("question", query_obj.get("query", "")))
    qrels = {str(k): float(v) for k, v in query_obj.get("qrels", {}).items()}
    exclusions = set(str(x) for x in query_obj.get("exclusions", []))
    gold_dids = {d for d, r in qrels.items() if r >= 1 and d not in exclusions}

    # 1. Baseline retrieval
    terms, _ = analyzer.analyze(q_text)
    orig_query_toks = defaultdict(float)
    for t in terms:
        orig_query_toks[t] += 1.0

    df_base = pd.DataFrame([{"qid": qid, "query_toks": dict(orig_query_toks)}])
    base_res = bm25.transform(df_base)
    base_docs = []
    if not base_res.empty:
        raw_docs = base_res["docno"].tolist()
        base_docs = [d for d in raw_docs if d not in exclusions][:1000]

    base_ranks = {docno: r for r, docno in enumerate(base_docs, 1)}
    base_metrics = compute_ir_metrics([(d, 1.0) for d in base_docs], qrels, qid, exclusions=exclusions)

    # Analyzed query terms to exclude
    query_excluded_terms = set(terms)

    # 2. Channel proposals (each top-500)
    channel_proposals = {}
    channel_latencies = {}
    for name, prop in proposers.items():
        t0 = time.perf_counter()
        props = prop.propose(query_obj, top_k=500)
        channel_latencies[name] = (time.perf_counter() - t0) * 1000.0
        # Ensure candidate is in canonical pool P and not in query_excluded_terms
        props_clean = [(t, s) for t, s in props if t in pool_set and t not in query_excluded_terms]
        channel_proposals[name] = props_clean

    # Compute RRF Fusions
    # RRF-Core3: WholeQueryBGE + AnchorBGEFiltered + PPMISidecar
    core3_rankings = {
        "WholeQueryBGE": channel_proposals.get("WholeQueryBGE", []),
        "AnchorBGEFiltered": channel_proposals.get("AnchorBGEFiltered", []),
        "PPMISidecar": channel_proposals.get("PPMISidecar", []),
    }
    channel_proposals["RRF_Core3"] = rrf_proposer.fuse(core3_rankings, top_l=500)

    # RRF-Extended: Core3 + SparseLexicalContextProfiles + AcronymDefinitionRescue
    ext_rankings = dict(core3_rankings)
    if "SparseLexicalContextProfiles" in channel_proposals:
        ext_rankings["SparseLexicalContextProfiles"] = channel_proposals["SparseLexicalContextProfiles"]
    if "AcronymDefinitionRescue" in channel_proposals:
        ext_rankings["AcronymDefinitionRescue"] = channel_proposals["AcronymDefinitionRescue"]
    channel_proposals["RRF_Extended"] = rrf_proposer.fuse(ext_rankings, top_l=500)

    # Proposer ranks map
    proposer_ranks_map = defaultdict(dict)
    for ch_name, ranking in channel_proposals.items():
        for r, (t, _) in enumerate(ranking, 1):
            proposer_ranks_map[t][f"{ch_name}_rank"] = r

    # 3. Form Core Reference Universe R_q: (T_Phase1 n P_q) U U_m C_{500}^m
    clean_phase1_cands = {t for t in phase1_cands_for_q if t in pool_set and t not in query_excluded_terms}
    r_core = set(clean_phase1_cands)
    for ranking in channel_proposals.values():
        r_core.update([t for t, _ in ranking])

    # Compute PPMI fidelity if both sidecar and live PPMI are present
    live_ppmi_terms = [t for t, _ in channel_proposals.get("LivePPMI", [])]
    sidecar_ppmi_terms = [t for t, _ in channel_proposals.get("PPMISidecar", [])]
    if live_ppmi_terms:
        live_set = set(live_ppmi_terms)
        sidecar_set = set(sidecar_ppmi_terms)
        ppmi_recall500 = len(live_set.intersection(sidecar_set)) / float(len(live_set)) if live_set else 1.0
        ppmi_rbo = compute_rbo(sidecar_ppmi_terms, live_ppmi_terms, p=0.98, k=500)
    else:
        ppmi_recall500 = 1.0
        ppmi_rbo = 1.0

    status_meta = {
        "dataset": dataset,
        "qid": qid,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "qrels_hash": qrels_hash,
        "pool_hash": pool_hash,
        "sampled": True,
        "num_candidates": len(r_core),
        "num_variants": len(r_core) * len(weights),
        "num_cutoff_entries": 0,
        "is_empty_reference": len(r_core) == 0,
        "ppmi_recall500": float(ppmi_recall500),
        "ppmi_rbo": float(ppmi_rbo),
        "channel_latencies_json": json.dumps(channel_latencies),
        "status": "SUCCESS",
        "error_msg": "",
    }


    if not r_core:
        return pd.DataFrame(), pd.DataFrame(), status_meta

    candidates_list = sorted(list(r_core))

    # 4. Build counterfactual variants for PyTerrier execution across weights
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

    # 5. Batch retrieval execution
    results_by_var = defaultdict(list)
    for i in range(0, len(variants_to_eval), chunk_size):
        check_memory_watchdog()
        chunk = variants_to_eval[i:i + chunk_size]
        df_chunk = pd.DataFrame([{"qid": item["var_id"], "query_toks": item["query_toks"]} for item in chunk])
        res_chunk = bm25.transform(df_chunk)
        if not res_chunk.empty:
            q_arr = res_chunk["qid"].values
            d_arr = res_chunk["docno"].values
            for q_id, doc in zip(q_arr, d_arr):
                results_by_var[q_id].append(str(doc))

    # Precompute ideal DCG@10 and total relevant docs
    n_rel = sum(1 for r in qrels.values() if r >= 1)
    ideal_rels = sorted([r for r in qrels.values() if r >= 1], reverse=True)[:10]
    idcg10 = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    delta_thresh = thresholds.get("delta", DEFAULT_DELTA)
    epsilon_thresh = thresholds.get("epsilon", EPSILON)
    tau_thresh = thresholds.get("tau", TAU)

    # 6. Evaluate metrics & cutoff entries
    parent_records = []
    cutoff_records = []

    # Map candidate -> list of variant outcomes to compute deduplicated cutoff entries
    cand_variant_outcomes = defaultdict(list)

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
            cutoff_net[f"net_rel_docs_k{k}"] = ent - lea

        net_k1000 = cutoff_net["net_rel_docs_k1000"]

        # Exhaustive Mutually Exclusive Action Partition:
        # Helpful: d_ndcg10 >= delta and net_rel_docs_k1000 >= 0
        # Harmful: d_ndcg10 < -epsilon or net_rel_docs_k1000 < 0
        # Neutral: otherwise
        is_helpful_act = (d_ndcg10 >= delta_thresh) and (net_k1000 >= 0)
        is_harmful_act = (d_ndcg10 < -epsilon_thresh) or (net_k1000 < 0)
        is_neutral_act = (not is_helpful_act) and (not is_harmful_act)
        is_waste_act = (d_ndcg10 <= 0.0) and (d_r1000 <= 0.0)

        p_row = {
            "dataset": dataset,
            "qid": qid,
            "candidate_term": cand,
            "weight": float(w),
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "qrels_hash": qrels_hash,
            "baseline_ndcg10": float(base_metrics["ndcg10"]),
            "expanded_ndcg10": float(ndcg10_val),
            "delta_ndcg10": float(d_ndcg10),
            "baseline_r1000": float(base_metrics["r1000"]),
            "expanded_r1000": float(r1000_val),
            "delta_r100": float(d_r100),
            "delta_r200": float(d_r200),
            "delta_r500": float(d_r500),
            "delta_r1000": float(d_r1000),
            "net_rel_docs_k10": int(cutoff_net.get("net_rel_docs_k10", 0)),
            "net_rel_docs_k100": int(cutoff_net["net_rel_docs_k100"]),
            "net_rel_docs_k200": int(cutoff_net["net_rel_docs_k200"]),
            "net_rel_docs_k500": int(cutoff_net["net_rel_docs_k500"]),
            "net_rel_docs_k1000": int(cutoff_net["net_rel_docs_k1000"]),
            "is_ranking_helpful_action": bool(is_helpful_act),
            "is_recall_helpful_k10": bool(cutoff_net.get("net_rel_docs_k10", 0) >= 1 and d_ndcg10 >= -epsilon_thresh),
            "is_recall_helpful_k100": bool(cutoff_net["net_rel_docs_k100"] >= 1 and d_ndcg10 >= -epsilon_thresh),
            "is_recall_helpful_k200": bool(cutoff_net["net_rel_docs_k200"] >= 1 and d_ndcg10 >= -epsilon_thresh),
            "is_recall_helpful_k500": bool(cutoff_net["net_rel_docs_k500"] >= 1 and d_ndcg10 >= -epsilon_thresh),
            "is_recall_helpful_k1000": bool(cutoff_net["net_rel_docs_k1000"] >= 1 and d_ndcg10 >= -epsilon_thresh),
            "action_helpful": bool(is_helpful_act),
            "action_harmful": bool(is_harmful_act),
            "action_neutral": bool(is_neutral_act),
            "action_waste": bool(is_waste_act),
            "in_phase1": bool(cand in clean_phase1_cands),
        }
        p_ranks = proposer_ranks_map.get(cand, {})
        p_row["wq_rank"] = float(p_ranks.get("WholeQueryBGE_rank", np.nan))
        p_row["anchor_filt_rank"] = float(p_ranks.get("AnchorBGEFiltered_rank", np.nan))
        p_row["anchor_all_rank"] = float(p_ranks.get("AnchorBGEAll_rank", np.nan))
        p_row["ppmi_sidecar_rank"] = float(p_ranks.get("PPMISidecar_rank", np.nan))
        p_row["live_ppmi_rank"] = float(p_ranks.get("LivePPMI_rank", np.nan))
        p_row["lex_rank"] = float(p_ranks.get("LivePPMI_rank", p_ranks.get("PPMISidecar_rank", np.nan)))
        p_row["sparse_lex_rank"] = float(p_ranks.get("SparseLexicalContextProfiles_rank", np.nan))
        p_row["acronym_rank"] = float(p_ranks.get("AcronymDefinitionRescue_rank", np.nan))
        p_row["rrf_core3_rank"] = float(p_ranks.get("RRF_Core3_rank", np.nan))
        p_row["rrf_ext_rank"] = float(p_ranks.get("RRF_Extended_rank", np.nan))
        parent_records.append(p_row)

        # Store variant outcome for cutoff entries
        cand_variant_outcomes[cand].append({
            "weight": w,
            "exp_ranks": exp_ranks,
            "d_ndcg10": d_ndcg10,
            "cutoff_net": cutoff_net,
        })

    # Deduplicate cutoff entries over weights for each candidate term
    for cand, v_list in cand_variant_outcomes.items():
        for k in TRACKED_CUTOFFS:
            for did in gold_dids:
                b_r = base_ranks.get(did, 9999)
                if b_r > k:  # Document was strictly outside baseline top-K
                    raw_in = False
                    safe_in = False
                    for v in v_list:
                        e_r = v["exp_ranks"].get(did, 9999)
                        if e_r <= k:
                            raw_in = True
                            # Recall-safe entry: d_ndcg10 >= -epsilon and net_rel_docs_k >= 1
                            if (v["d_ndcg10"] >= -epsilon_thresh) and (v["cutoff_net"][f"net_rel_docs_k{k}"] >= 1):
                                safe_in = True
                    if raw_in:
                        cutoff_records.append({
                            "dataset": dataset,
                            "qid": qid,
                            "candidate_term": cand,
                            "cutoff": int(k),
                            "docid": str(did),
                            "raw_entry": bool(raw_in),
                            "recall_safe_entry": bool(safe_in),
                        })

    status_meta["num_cutoff_entries"] = len(cutoff_records)

    parent_df = pd.DataFrame(parent_records)
    cutoff_df = pd.DataFrame(cutoff_records) if cutoff_records else pd.DataFrame(columns=[
        "dataset", "qid", "candidate_term", "cutoff", "docid", "raw_entry", "recall_safe_entry"
    ])

    return parent_df, cutoff_df, status_meta


def run_dataset_gate1_evaluation(
    dataset: str,
    output_dir: str,
    weights: List[float],
    sample_size: int = 50,
    seed: int = 42,
    chunk_size: int = 100,
    resume: bool = True,
    phase1_audit_df: Optional[pd.DataFrame] = None,
    frozen_config: Optional[Dict[str, Any]] = None,
    encoder=None,
    device: str = "cpu",
    config_hash: str = "core_dev_v1",
    measure_fidelity: bool = False,
) -> Dict[str, Any]:
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

    # 1. Load canonical pool artifact & verify hash directly from terms
    pool_path = os.path.abspath(f"data/cache/canonical_pools/{safe_ds}_canonical_pool.json")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Canonical pool artifact not found at {pool_path}!")

    with open(pool_path, "r", encoding="utf-8") as f:
        pool_data = json.load(f)

    pool_terms = pool_data["terms"]
    pool_size = len(pool_terms)
    
    # Direct raw-term SHA-256 computation
    computed_pool_hash = hashlib.sha256("\n".join(pool_terms).encode("utf-8")).hexdigest()
    assert computed_pool_hash == pool_data.get("sha256"), f"Corrupt pool artifact {pool_path}: computed {computed_pool_hash} != stored {pool_data.get('sha256')}"
    pool_hash = computed_pool_hash

    # Verify against frozen config if present
    if frozen_config:
        expected_size = frozen_config["pool_spec"]["sizes"].get(dataset)
        expected_hash = frozen_config["pool_spec"]["hashes"].get(dataset)
        assert pool_size == expected_size, f"Pool size mismatch for {dataset}: {pool_size} != {expected_size}"
        assert pool_hash == expected_hash, f"Pool hash mismatch for {dataset}: {pool_hash} != {expected_hash}"

    pool_set = set(pool_terms)
    print(f"  [Canonical Pool] Loaded {pool_size:,} terms (SHA-256: {pool_hash[:12]}...)")

    # Compute dataset and qrels content hashes
    q_file, qrels_file = BenchmarkLoader.get_query_qrels_paths(dataset)
    with open(q_file, "rb") as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()
    with open(qrels_file, "rb") as f:
        qrels_hash = hashlib.sha256(f.read()).hexdigest()

    # 2. Build full-lexicon statistics for query anchors
    lex = index.getLexicon()
    full_df_map = {}
    full_idf_map = {}
    for entry in lex:
        t = str(entry.getKey())
        df = int(entry.getValue().getDocumentFrequency())
        idf = max(math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5)), 0.0)
        full_df_map[t] = df
        full_idf_map[t] = idf

    # 3. Load queries (using frozen manifest if available)
    all_queries, _ = BenchmarkLoader.load_queries(dataset)
    q_by_id = {str(q.get("query_id", q.get("qid", ""))): q for q in all_queries}

    frozen_qids = None
    if frozen_config and "query_manifests" in frozen_config:
        q_manifest = frozen_config["query_manifests"]
        if sample_size == 10 and "probe_40_qids" in q_manifest:
            frozen_qids = q_manifest["probe_40_qids"].get(dataset)
        elif sample_size == 50 and "dev_200_qids" in q_manifest:
            frozen_qids = q_manifest["dev_200_qids"].get(dataset)

    if frozen_qids:
        if len(frozen_qids) != len(set(frozen_qids)):
            raise RuntimeError(f"FATAL: Duplicate QIDs detected in frozen manifest for dataset '{dataset}'!")
        missing_qids = set(frozen_qids) - set(q_by_id.keys())
        if missing_qids:
            raise RuntimeError(f"FATAL: Dataset '{dataset}' is missing {len(missing_qids)} QIDs from frozen manifest: {list(missing_qids)[:5]}")
        sampled_queries = [q_by_id[qid] for qid in frozen_qids]
        print(f"  [Queries] Loaded {len(sampled_queries)} queries from frozen manifest for {dataset}.")
    else:
        rng = random.Random(seed)
        if len(all_queries) <= sample_size:
            sampled_queries = list(all_queries)
        else:
            sampled_queries = rng.sample(all_queries, sample_size)
        print(f"  [Queries] Sampled {len(sampled_queries)} queries using seed={seed}.")

    max_ex = max((len(q.get("exclusions", [])) for q in all_queries), default=0)
    k_fetch = min(num_docs, 1000 + max_ex)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=k_fetch)

    # 4. Load all 4 Gate 1 Sidecars with frozen config wiring
    sidecar_mgr = Gate1SidecarManager(config=frozen_config, bge_model_name=bge_model)
    bge_sidecar = sidecar_mgr.build_or_load_bge_sidecar(dataset, pool_terms, num_docs=num_docs, encoder=encoder, device=device)
    pool_embeddings = bge_sidecar["pool_embeddings"]
    surf_to_idx = bge_sidecar["surf_to_idx"]

    ppmi_top_m = frozen_config.get("ppmi", {}).get("top_m", 600) if frozen_config else 600
    ppmi_sidecar = sidecar_mgr.build_or_load_bounded_ppmi_sidecar(dataset, index, pool_terms, num_docs=num_docs, top_m=ppmi_top_m)
    anchor_ppmi = ppmi_sidecar["anchor_ppmi"]

    acronym_sidecar = sidecar_mgr.build_or_load_acronym_rescue_sidecar(dataset, pool_terms, num_docs=num_docs)
    acronym_to_pool = acronym_sidecar["acronym_to_pool"]

    lex_sidecar = sidecar_mgr.build_or_load_sparse_lexical_sidecar(dataset, pool_terms, num_docs=num_docs)
    aux_retriever = lex_sidecar["retriever"]

    # Proposers initialization
    wq_proposer = WholeQueryBGEProposer(pool_terms, pool_embeddings, encoder, device=device)
    anchor_filt_proposer = AnchorBGEProposer(
        encoder=encoder,
        pool_terms=pool_terms,
        pool_embeddings_tensor=pool_embeddings,
        surf_to_pool_idx=surf_to_idx,
        idf_map=full_idf_map,
        df_map=full_df_map,
        num_docs=num_docs,
        filter_anchors=True,
        device=device,
    )
    anchor_all_proposer = AnchorBGEProposer(
        encoder=encoder,
        pool_terms=pool_terms,
        pool_embeddings_tensor=pool_embeddings,
        surf_to_pool_idx=surf_to_idx,
        idf_map=full_idf_map,
        df_map=full_df_map,
        num_docs=num_docs,
        filter_anchors=False,
        device=device,
    )
    ppmi_sidecar_proposer = PPMISidecarProposer(anchor_ppmi, full_idf_map, full_df_map, num_docs)
    sparse_lex_proposer = SparseLexicalContextProposer(aux_retriever, pool_terms)
    acronym_proposer = AcronymDefinitionRescueProposer(acronym_to_pool, pool_terms)
    
    rrf_k = frozen_config.get("rrf", {}).get("k", 60) if frozen_config else 60
    rrf_proposer = RRFHybridProposer(k=rrf_k)

    proposers = {
        "WholeQueryBGE": wq_proposer,
        "AnchorBGEFiltered": anchor_filt_proposer,
        "AnchorBGEAll": anchor_all_proposer,
        "PPMISidecar": ppmi_sidecar_proposer,
        "SparseLexicalContextProfiles": sparse_lex_proposer,
        "AcronymDefinitionRescue": acronym_proposer,
    }

    # Only include LivePPMI for diagnostic comparison when measure_fidelity is True
    if measure_fidelity:
        live_ppmi_proposer = LexicalPPMIProposer(index, pool_terms, full_idf_map, full_df_map, num_docs)
        proposers["LivePPMI"] = live_ppmi_proposer

    # Phase 1 lookup (fail-closed if missing)
    phase1_map = defaultdict(set)
    if phase1_audit_df is None or phase1_audit_df.empty:
        raise RuntimeError(f"FATAL: Phase 1 candidate audit is required but missing/empty!")
    ds_phase1 = phase1_audit_df[phase1_audit_df["dataset"] == dataset]
    if len(ds_phase1) == 0:
        raise RuntimeError(f"FATAL: Phase 1 candidate audit has 0 rows for dataset '{dataset}'!")
    for _, row in ds_phase1.iterrows():
        phase1_map[str(row["qid"])].add(str(row["candidate_term"]))

    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    thresholds = frozen_config.get("thresholds", {}) if frozen_config else {
        "delta": DEFAULT_DELTA, "epsilon": EPSILON, "tau": TAU, "rho": DEFAULT_RHO
    }

    total_variants = 0
    total_cutoff_entries = 0
    t0_retrieval = time.perf_counter()

    for idx_q, q in enumerate(sampled_queries, 1):
        qid = str(q["query_id"])
        parent_shard, cutoff_shard, status_shard = get_shard_paths(shards_dir, dataset, qid)

        if resume and os.path.exists(parent_shard) and os.path.exists(cutoff_shard) and os.path.exists(status_shard):
            try:
                st_data = pq.read_table(status_shard).to_pydict()
                st_cfg = st_data.get("config_hash", [""])[0]
                st_ds = st_data.get("dataset_hash", [""])[0]
                st_qrels = st_data.get("qrels_hash", [""])[0]
                st_pool = st_data.get("pool_hash", [""])[0]
                if st_cfg != config_hash or st_ds != dataset_hash or st_qrels != qrels_hash or st_pool != pool_hash:
                    raise RuntimeError(
                        f"FATAL: Stale shard for QID {qid} has mismatched hashes: "
                        f"config={st_cfg[:8]} vs {config_hash[:8]}, qrels={st_qrels[:8]} vs {qrels_hash[:8]}, pool={st_pool[:8]} vs {pool_hash[:8]}. Clean output directory."
                    )
                p_rows = pq.read_metadata(parent_shard).num_rows
                c_rows = pq.read_metadata(cutoff_shard).num_rows
                total_variants += p_rows
                total_cutoff_entries += c_rows
                print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Resumed from shard ({p_rows} variants, {c_rows} cutoff entries)")
                continue
            except Exception as e:
                if "FATAL" in str(e):
                    raise
                pass

        t0_q = time.perf_counter()
        df_p, df_c, q_status = evaluate_query_gate1(
            dataset=dataset,
            query_obj=q,
            index=index,
            analyzer=analyzer,
            bm25=bm25,
            proposers=proposers,
            rrf_proposer=rrf_proposer,
            pool_set=pool_set,
            phase1_cands_for_q=phase1_map.get(qid, set()),
            weights=weights,
            thresholds=thresholds,
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            qrels_hash=qrels_hash,
            pool_hash=pool_hash,
            chunk_size=chunk_size,
        )

        t_q = time.perf_counter() - t0_q

        # Save shards
        df_p.to_parquet(parent_shard, index=False)
        df_c.to_parquet(cutoff_shard, index=False)
        pd.DataFrame([q_status]).to_parquet(status_shard, index=False)

        total_variants += len(df_p)
        total_cutoff_entries += len(df_c)
        v_rate = len(df_p) / max(t_q, 0.001)
        print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Evaluated {len(df_p)} variants ({len(df_c)} cutoff entries) in {t_q:.2f}s ({v_rate:.1f} var/s)")

        del df_p, df_c
        gc.collect()

    total_time = time.perf_counter() - t0_retrieval
    print(f"Completed {dataset}: {total_variants} variants, {total_cutoff_entries} cutoff entries in {total_time:.2f}s")

    # Fidelity and performance benchmarking
    fidelity_meta = {}
    if measure_fidelity:
        print(f"\n  --- [Fidelity & Performance Probe Benchmark: {dataset}] ---")
        ppmi_lookup_latencies = []
        bge_scoring_latencies = []
        bge_full_latencies = []

        for q in sampled_queries:
            # 1. Warmup
            _ = ppmi_sidecar_proposer.propose(q, top_k=500)
            _ = wq_proposer.propose(q, top_k=500)
            # 2. Repeated timed lookups (5 iterations)
            for _ in range(5):
                t0 = time.perf_counter()
                _ = ppmi_sidecar_proposer.propose(q, top_k=500)
                ppmi_lookup_latencies.append((time.perf_counter() - t0) * 1000.0)

            # 3. BGE scoring alone (dot product) vs full propose
            q_text = str(q.get("question", q.get("query", "")))
            if hasattr(encoder, "encode"):
                q_emb = encoder.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]
            else:
                q_emb = encoder.encode_queries([q_text])[0]
            q_tensor = torch.tensor(q_emb, device=device).unsqueeze(0)
            for _ in range(5):
                t0 = time.perf_counter()
                _ = torch.matmul(q_tensor, pool_embeddings.T).squeeze(0)
                if device == "cuda":
                    torch.cuda.synchronize()
                bge_scoring_latencies.append((time.perf_counter() - t0) * 1000.0)

            for _ in range(5):
                t0 = time.perf_counter()
                _ = wq_proposer.propose(q, top_k=500)
                bge_full_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Collect status metrics across sampled queries
        recalls = []
        rbos = []
        for q in sampled_queries:
            qid = str(q["query_id"])
            _, _, s_shard = get_shard_paths(shards_dir, dataset, qid)
            if os.path.exists(s_shard):
                df_s = pd.read_parquet(s_shard)
                if "ppmi_recall500" in df_s.columns:
                    recalls.append(float(df_s["ppmi_recall500"].iloc[0]))
                if "ppmi_rbo" in df_s.columns:
                    rbos.append(float(df_s["ppmi_rbo"].iloc[0]))

        fidelity_meta = {
            "ppmi_recall500_mean": float(np.mean(recalls)) if recalls else 1.0,
            "ppmi_rbo_mean": float(np.mean(rbos)) if rbos else 1.0,
            "ppmi_lookup_p50_ms": float(np.percentile(ppmi_lookup_latencies, 50)) if ppmi_lookup_latencies else 0.0,
            "ppmi_lookup_p95_ms": float(np.percentile(ppmi_lookup_latencies, 95)) if ppmi_lookup_latencies else 0.0,
            "bge_scoring_p50_ms": float(np.percentile(bge_scoring_latencies, 50)) if bge_scoring_latencies else 0.0,
            "bge_scoring_p95_ms": float(np.percentile(bge_scoring_latencies, 95)) if bge_scoring_latencies else 0.0,
            "bge_full_p50_ms": float(np.percentile(bge_full_latencies, 50)) if bge_full_latencies else 0.0,
            "bge_full_p95_ms": float(np.percentile(bge_full_latencies, 95)) if bge_full_latencies else 0.0,
            "ppmi_disk_mb": round(os.path.getsize(os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bounded_ppmi.json")) / (1024 ** 2), 2),
            "bge_disk_mb": round(os.path.getsize(os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bge_sidecar.pt")) / (1024 ** 2), 2),
        }
        print(f"  PPMI Fidelity: Recall@500 = {fidelity_meta['ppmi_recall500_mean']:.4f}, RBO = {fidelity_meta['ppmi_rbo_mean']:.4f}")
        print(f"  PPMI Latency: p50 = {fidelity_meta['ppmi_lookup_p50_ms']:.2f}ms, p95 = {fidelity_meta['ppmi_lookup_p95_ms']:.2f}ms")
        print(f"  BGE Scoring: p50 = {fidelity_meta['bge_scoring_p50_ms']:.2f}ms, p95 = {fidelity_meta['bge_scoring_p95_ms']:.2f}ms (Full: p95 = {fidelity_meta['bge_full_p95_ms']:.2f}ms)")
        print(f"  Disk Footprint: PPMI = {fidelity_meta['ppmi_disk_mb']} MB, BGE = {fidelity_meta['bge_disk_mb']} MB")

    return {
        "dataset": dataset,
        "num_docs": num_docs,
        "pool_size": pool_size,
        "pool_sha256": pool_hash,
        "num_queries": len(sampled_queries),
        "total_variants": total_variants,
        "total_cutoff_entries": total_cutoff_entries,
        "retrieval_time_sec": total_time,
        "fidelity_meta": fidelity_meta,
    }


def assemble_shards_to_master(shards_dir: str, shard_type: str, output_path: str, expected_qids: Optional[Set[str]] = None):
    """
    Streams individual query Parquet shards into a single master Parquet file
    using pyarrow.ParquetWriter with fail-closed schema and QID verification.
    shard_type: 'audit', 'cutoff', or 'status'
    """
    if shard_type == "cutoff":
        all_files = sorted(glob.glob(os.path.join(shards_dir, "*_cutoff_entries.parquet")))
    elif shard_type == "status":
        all_files = sorted(glob.glob(os.path.join(shards_dir, "*_status.parquet")))
    elif shard_type == "audit":
        all_files = sorted([
            f for f in glob.glob(os.path.join(shards_dir, "*.parquet"))
            if not f.endswith("_cutoff_entries.parquet") and not f.endswith("_status.parquet")
        ])
    else:
        raise ValueError(f"Unknown shard_type: {shard_type}")

    if not all_files:
        print(f"No target files found in {shards_dir} for shard_type={shard_type}")
        return

    print(f"Streaming {len(all_files)} {shard_type} shards to {output_path} via pyarrow.ParquetWriter...")
    writer = None
    total_rows = 0
    unified_schema = None
    assembled_qids = set()

    for fpath in all_files:
        try:
            table = pq.read_table(fpath)
            if unified_schema is None:
                unified_schema = table.schema
            if table.num_rows == 0:
                del table
                continue
            if writer is None:
                writer = pq.ParquetWriter(output_path, unified_schema, compression="zstd")
            elif table.schema != unified_schema:
                # Cast to unified schema if strictly compatible
                table = table.cast(unified_schema)
            writer.write_table(table)
            total_rows += table.num_rows
            if "qid" in table.column_names:
                for qid_val in table["qid"].to_pylist():
                    assembled_qids.add(str(qid_val))
            del table
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to assemble shard {fpath}: {e}")

    if writer is not None:
        writer.close()
    elif unified_schema is not None:
        # All shards had 0 rows (e.g. no cutoff entries in sample)
        empty_table = pa.Table.from_batches([], schema=unified_schema)
        pq.write_table(empty_table, output_path, compression="zstd")

    print(f"Successfully assembled {total_rows:,} rows ({len(assembled_qids)} QIDs) -> {output_path}")

    if expected_qids is not None:
        missing = expected_qids - assembled_qids
        if missing:
            raise RuntimeError(f"FATAL: Shard assembly dropped {len(missing)} expected QIDs: {list(missing)[:5]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Gate 1 Candidate Selection Evaluation Harness")
    parser.add_argument("--datasets", type=str, default=",".join(DEV_DATASETS), help="Comma-separated datasets")
    parser.add_argument("--sample-size", type=int, default=50, help="Query sample size per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--weights", type=str, default="0.05,0.10,0.30,0.50,1.00", help="Weights grid")
    parser.add_argument("--chunk-size", type=int, default=100, help="Retrieval batch chunk size")
    parser.add_argument("--output-dir", type=str, default="results/gate1_selection/dev_corrected", help="Output directory")
    parser.add_argument("--frozen-config-path", type=str, default=None, help="Path to authoritative frozen YAML configuration")
    parser.add_argument("--phase1-audit", type=str, default="for_review/pool_phase/pool_candidate_audit.parquet", help="Path to Phase 1 candidate audit")
    parser.add_argument("--clean-output", action="store_true", help="Clean output directory if it already exists")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resumption")
    parser.add_argument("--measure-fidelity", action="store_true", help="Measure PPMI sidecar fidelity vs live PPMI")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5", help="BGE model name")
    parser.add_argument("--config-hash", type=str, default="gate1_core_dev_v1", help="Configuration hash")
    return parser.parse_args()


def main():
    args = parse_args()

    # Fail-closed output directory safety
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.clean_output:
            print(f"Cleaning existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
        elif not args.no_resume:
            print(f"Output directory {args.output_dir} exists and will be resumed.")
        else:
            raise FileExistsError(
                f"FATAL: Output directory {args.output_dir} already exists and is not empty! Specify a unique directory or use --clean-output."
            )
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize PyTerrier with hard-capped JVM heap
    if not pt.started():
        pt.java.set_memory_limit(4096)
        pt.init()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load frozen configuration and compute its exact SHA-256
    frozen_config = None
    if args.frozen_config_path:
        if not os.path.exists(args.frozen_config_path):
            raise FileNotFoundError(f"Frozen config not found at {args.frozen_config_path}")
        with open(args.frozen_config_path, "rb") as f:
            raw_bytes = f.read()
            config_hash = hashlib.sha256(raw_bytes).hexdigest()
        frozen_config = yaml.safe_load(raw_bytes.decode("utf-8"))
        print(f"Loaded frozen configuration from {args.frozen_config_path} (SHA-256: {config_hash[:12]}...)")
        
        # Enforce frozen parameters
        if "weights" in frozen_config:
            weights = [float(w) for w in frozen_config["weights"]]
        else:
            weights = [float(w.strip()) for w in args.weights.split(",")]
        if "seed" in frozen_config:
            seed = int(frozen_config["seed"])
        else:
            seed = args.seed
    else:
        config_hash = args.config_hash
        weights = [float(w.strip()) for w in args.weights.split(",")]
        seed = args.seed

    # Reseed global RNG with frozen seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load Phase 1 candidate audit if available
    phase1_audit_df = None
    if args.phase1_audit and os.path.exists(args.phase1_audit):
        print(f"Loading Phase 1 candidate audit from {args.phase1_audit}...")
        phase1_audit_df = pd.read_parquet(args.phase1_audit)
        print(f"Loaded {len(phase1_audit_df):,} Phase 1 audit rows.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Initialize encoder once on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder {args.bge_model} on {device}...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.bge_model, device=device)

    meta_summaries = []
    for ds in datasets:
        ds_meta = run_dataset_gate1_evaluation(
            dataset=ds,
            output_dir=args.output_dir,
            weights=weights,
            sample_size=args.sample_size,
            seed=seed,
            chunk_size=args.chunk_size,
            resume=not args.no_resume,
            phase1_audit_df=phase1_audit_df,
            frozen_config=frozen_config,
            encoder=encoder,
            device=device,
            config_hash=config_hash,
            measure_fidelity=args.measure_fidelity,
        )
        meta_summaries.append(ds_meta)

    # Assemble master artifacts with fail-closed QID verification
    expected_qids_all = set()
    if frozen_config and "query_manifests" in frozen_config:
        q_manifest = frozen_config["query_manifests"]
        for ds in datasets:
            if args.sample_size == 10 and "probe_40_qids" in q_manifest:
                expected_qids_all.update(q_manifest["probe_40_qids"].get(ds, []))
            elif args.sample_size == 50 and "dev_200_qids" in q_manifest:
                expected_qids_all.update(q_manifest["dev_200_qids"].get(ds, []))

    shards_dir = os.path.join(args.output_dir, "shards")
    master_parent = os.path.join(args.output_dir, "gate1_candidate_audit.parquet")
    master_cutoff = os.path.join(args.output_dir, "gate1_cutoff_entries.parquet")
    master_status = os.path.join(args.output_dir, "query_status.parquet")

    assemble_shards_to_master(shards_dir, shard_type="audit", output_path=master_parent, expected_qids=expected_qids_all if expected_qids_all else None)
    assemble_shards_to_master(shards_dir, shard_type="cutoff", output_path=master_cutoff)
    assemble_shards_to_master(shards_dir, shard_type="status", output_path=master_status, expected_qids=expected_qids_all if expected_qids_all else None)

    import platform
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    driver_ver = "N/A"
    try:
        with open("/proc/driver/nvidia/version", "r") as f:
            driver_ver = f.readline().strip().split()[7]
    except Exception:
        pass

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash,
        "datasets": datasets,
        "sample_size": args.sample_size,
        "seed": seed,
        "weights": weights,
        "peak_rss_gib": round(get_process_tree_rss_bytes() / (1024 ** 3), 3),
        "total_variants": sum(s["total_variants"] for s in meta_summaries),
        "total_cutoff_entries": sum(s["total_cutoff_entries"] for s in meta_summaries),
        "total_retrieval_time_sec": round(sum(s["retrieval_time_sec"] for s in meta_summaries), 2),
        "environment": {
            "cpu_model": platform.processor() or "x86_64",
            "gpu_name": gpu_name,
            "driver_version": driver_ver,
            "cuda_version": cuda_ver,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "pyterrier_version": getattr(pt, "__version__", "5.11"),
        },
        "dataset_summaries": meta_summaries,
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved Gate 1 run manifest -> {manifest_path}")



if __name__ == "__main__":
    main()
