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
import subprocess
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
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from evaluation.baselines.pyterrier_harness import init_pyterrier
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
from crve.selection.gate1_sidecars import (
    Gate1SidecarManager,
    validate_gate1_config,
    compute_pool_sha256,
    compute_corpus_source_hash,
)
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


def load_canonical_pool_terms(dataset: str) -> List[str]:
    """Loads terms list from canonical pool JSON artifact."""
    safe_ds = dataset.lower().replace("-", "_")
    pool_path = os.path.abspath(f"data/cache/canonical_pools/{safe_ds}_canonical_pool.json")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Canonical pool artifact not found at {pool_path}!")
    with open(pool_path, "r", encoding="utf-8") as f:
        pool_data = json.load(f)
    return pool_data["terms"]


def compute_index_hash(dataset: str) -> str:
    """Computes SHA-256 hash of index_manifest.json for dataset."""
    safe_ds = dataset.lower().replace("-", "_")
    idx_path = os.path.abspath(f"data/cache/terrier_indices/{safe_ds}_default/data.properties")
    idx_manifest_path = os.path.join(os.path.dirname(idx_path), "index_manifest.json")
    if not os.path.exists(idx_manifest_path):
        raise FileNotFoundError(f"FATAL: index_manifest.json not found at {idx_manifest_path}!")
    manifest_hasher = hashlib.sha256()
    with open(idx_manifest_path, "rb") as f:
        while chunk := f.read(65536):
            manifest_hasher.update(chunk)
    return manifest_hasher.hexdigest()


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


PEAK_PROCESS_TREE_RSS_BYTES: int = 0


def get_git_info() -> Dict[str, Any]:
    """Retrieves current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return {"git_commit": commit, "git_dirty": bool(status)}
    except Exception:
        return {"git_commit": "UNKNOWN", "git_dirty": False}


def get_process_tree_rss_bytes() -> int:
    """Calculates total RSS memory consumed by this process and all children (e.g. JVM)."""
    global PEAK_PROCESS_TREE_RSS_BYTES
    try:
        parent = psutil.Process()
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if total > PEAK_PROCESS_TREE_RSS_BYTES:
            PEAK_PROCESS_TREE_RSS_BYTES = total
        return total
    except Exception:
        return 0


def check_memory_watchdog(config: Optional[Dict[str, Any]] = None):
    """Fail-closed memory watchdog: triggers gc at warn threshold, aborts at hard cap."""
    if config and "memory_watchdog" in config:
        warn_rss_gib = float(config["memory_watchdog"]["warn_rss_gib"])
        abort_rss_gib = float(config["memory_watchdog"]["hard_abort_rss_gib"])
    else:
        warn_rss_gib = 8.0
        abort_rss_gib = 12.0
    rss_gib = get_process_tree_rss_bytes() / (1024 ** 3)
    if rss_gib >= abort_rss_gib:
        raise MemorySafetyError(
            f"FATAL: Process-tree RSS ({rss_gib:.2f} GiB) exceeded hard cap of {abort_rss_gib} GiB! Aborting to prevent OOM."
        )
    elif rss_gib >= warn_rss_gib:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_manifest_qids(manifest_qids: List[str], dataset: str) -> None:
    """Validates frozen manifest QIDs: non-empty, no duplicates."""
    if not manifest_qids:
        raise ValueError(f"FATAL: Manifest QIDs for dataset '{dataset}' cannot be empty!")
    if len(manifest_qids) != len(set(manifest_qids)):
        raise RuntimeError(f"FATAL: Duplicate QIDs detected in frozen manifest for dataset '{dataset}'!")


def validate_shard_hashes(status_dict: Dict[str, Any], expected_hashes: Dict[str, str], qid: str) -> None:
    """Validates shard hashes match expected config, dataset, qrels, and pool hashes."""
    st_cfg = status_dict.get("config_hash", "")
    st_ds = status_dict.get("dataset_hash", "")
    st_qrels = status_dict.get("qrels_hash", "")
    st_pool = status_dict.get("pool_hash", "")

    exp_cfg = expected_hashes.get("config_hash", "")
    exp_ds = expected_hashes.get("dataset_hash", "")
    exp_qrels = expected_hashes.get("qrels_hash", "")
    exp_pool = expected_hashes.get("pool_hash", "")

    if st_cfg != exp_cfg or st_ds != exp_ds or st_qrels != exp_qrels or st_pool != exp_pool:
        raise RuntimeError(
            f"FATAL: Stale shard for QID {qid} has mismatched hashes: "
            f"config={st_cfg[:8]} vs {exp_cfg[:8]}, dataset={st_ds[:8]} vs {exp_ds[:8]}, "
            f"qrels={st_qrels[:8]} vs {exp_qrels[:8]}, pool={st_pool[:8]} vs {exp_pool[:8]}. Clean output directory."
        )


def validate_action_coverage(audit_df: pd.DataFrame, universe_df: pd.DataFrame, weights: List[float]) -> None:
    """Validates exact Cartesian set equality between actual audit actions and reference universe x weights."""
    expected_triples = set()
    for _, row in universe_df.iterrows():
        qid = str(row["qid"])
        t = str(row["candidate_term"])
        for w in weights:
            expected_triples.add((qid, t, round(float(w), 4)))

    actual_triples = set()
    for _, row in audit_df.iterrows():
        qid = str(row["qid"])
        t = str(row["candidate_term"])
        w = round(float(row["weight"]), 4)
        actual_triples.add((qid, t, w))

    if len(audit_df) != len(actual_triples):
        raise RuntimeError(
            f"FATAL: Duplicate action evaluations found in audit: {len(audit_df)} rows vs {len(actual_triples)} unique triples!"
        )

    missing = expected_triples - actual_triples
    extra = actual_triples - expected_triples
    if missing or extra:
        raise RuntimeError(
            f"FATAL: Action coverage set equality failed! Missing triples: {len(missing)}, Extra triples: {len(extra)}."
        )


def get_shard_paths(shards_dir: str, dataset: str, qid: str) -> Tuple[str, str, str, str, str]:
    """Generates shard paths for parent candidate audit, cutoff entries, query status, reference universe, and diagnostic universe."""
    q_hash = hashlib.md5(qid.encode("utf-8")).hexdigest()[:12]
    parent_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}.parquet")
    cutoff_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_cutoff_entries.parquet")
    status_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_status.parquet")
    universe_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_universe.parquet")
    diag_universe_shard = os.path.join(shards_dir, f"{dataset}_{q_hash}_diag_universe.parquet")
    return parent_shard, cutoff_shard, status_shard, universe_shard, diag_universe_shard


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
    rrf_core3_proposer: RRFHybridProposer,
    rrf_ext_proposer: RRFHybridProposer,
    pool_set: Set[str],
    phase1_cands_for_q: Set[str],
    weights: List[float],
    thresholds: Dict[str, float],
    config_hash: str = "core_dev_v1",
    dataset_hash: str = "",
    qrels_hash: str = "",
    pool_hash: str = "",
    chunk_size: int = 100,
    frozen_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Evaluates all candidate actions for a single query.
    Emits parent candidate audit, cutoff entries, and query status.
    """
    check_memory_watchdog(frozen_config)
    qid = str(query_obj.get("query_id", query_obj.get("qid", "")))
    q_text = str(query_obj.get("question", query_obj.get("query", "")))
    qrels = {str(k): float(v) for k, v in query_obj.get("qrels", {}).items()}
    exclusions = set(str(x) for x in query_obj.get("excluded_doc_ids", query_obj.get("exclusions", [])))
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

    # Compute RRF Fusions with separate policy proposers
    # RRF-Core3: WholeQueryBGE + AnchorBGEFiltered + PPMISidecar
    core3_rankings = {
        "WholeQueryBGE": channel_proposals.get("WholeQueryBGE", []),
        "AnchorBGEFiltered": channel_proposals.get("AnchorBGEFiltered", []),
        "PPMISidecar": channel_proposals.get("PPMISidecar", []),
    }
    channel_proposals["RRF_Core3"] = rrf_core3_proposer.fuse(core3_rankings, top_l=500)

    # RRF-Extended: Core3 + SparseLexicalContextProfiles + AcronymDefinitionRescue
    ext_rankings = dict(core3_rankings)
    if "SparseLexicalContextProfiles" in channel_proposals:
        ext_rankings["SparseLexicalContextProfiles"] = channel_proposals["SparseLexicalContextProfiles"]
    if "AcronymDefinitionRescue" in channel_proposals:
        ext_rankings["AcronymDefinitionRescue"] = channel_proposals["AcronymDefinitionRescue"]
    channel_proposals["RRF_Extended"] = rrf_ext_proposer.fuse(ext_rankings, top_l=500)

    # Proposer ranks map
    proposer_ranks_map = defaultdict(dict)
    for ch_name, ranking in channel_proposals.items():
        for r, (t, _) in enumerate(ranking, 1):
            proposer_ranks_map[t][f"{ch_name}_rank"] = r

    OPERATIONAL_CHANNELS = [
        "WholeQueryBGE",
        "AnchorBGEFiltered",
        "AnchorBGEAll",
        "PPMISidecar",
        "SparseLexicalContextProfiles",
        "AcronymDefinitionRescue",
        "RRF_Core3",
        "RRF_Extended",
    ]

    # 3. Form Core Reference Universe R_q: (T_Phase1 n P_q) U U_{m in OPERATIONAL} C_{500}^m
    clean_phase1_cands = {t for t in phase1_cands_for_q if t in pool_set and t not in query_excluded_terms}
    r_core = set(clean_phase1_cands)
    candidate_sources = defaultdict(set)
    for t in clean_phase1_cands:
        candidate_sources[t].add("phase1")

    for ch_name in OPERATIONAL_CHANNELS:
        if ch_name in channel_proposals:
            for t, _ in channel_proposals[ch_name]:
                r_core.add(t)
                candidate_sources[t].add(ch_name)

    universe_rows = [
        {
            "dataset": dataset,
            "qid": qid,
            "candidate_term": t,
            "is_phase1": bool("phase1" in candidate_sources[t]),
            "proposer_sources": ",".join(sorted(candidate_sources[t])),
        }
        for t in sorted(r_core)
    ]
    df_universe = pd.DataFrame(universe_rows)

    # If LivePPMI is present (diagnostic mode), form separate Diagnostic Reference Universe R_q^diag
    df_diag_universe = None
    if "LivePPMI" in channel_proposals:
        r_diag = set(r_core)
        diag_sources = defaultdict(set)
        for t, s in candidate_sources.items():
            diag_sources[t] = set(s)
        for t, _ in channel_proposals["LivePPMI"]:
            r_diag.add(t)
            diag_sources[t].add("LivePPMI")

        diag_universe_rows = [
            {
                "dataset": dataset,
                "qid": qid,
                "candidate_term": t,
                "is_phase1": bool("phase1" in diag_sources[t]),
                "proposer_sources": ",".join(sorted(diag_sources[t])),
            }
            for t in sorted(r_diag)
        ]
        df_diag_universe = pd.DataFrame(diag_universe_rows)
        candidates_list = sorted(list(r_diag))
    else:
        candidates_list = sorted(list(r_core))

    # Compute PPMI fidelity if both sidecar and live PPMI are present (diagnostic only)
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

    sorted_r_core = sorted(list(r_core))
    r_core_sha = hashlib.sha256("\n".join(sorted_r_core).encode("utf-8")).hexdigest()
    status_meta = {
        "dataset": dataset,
        "qid": qid,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "qrels_hash": qrels_hash,
        "pool_hash": pool_hash,
        "sampled": True,
        "num_candidates": len(candidates_list),
        "num_variants": len(candidates_list) * len(weights),
        "num_cutoff_entries": 0,
        "is_empty_reference": len(r_core) == 0,
        "reference_universe_size": len(r_core),
        "reference_universe_sha256": r_core_sha,
        "diagnostic_universe_size": len(candidates_list) if df_diag_universe is not None else 0,
        "expected_action_count": len(candidates_list) * len(weights),
        "ppmi_recall500": float(ppmi_recall500),
        "ppmi_rbo": float(ppmi_rbo),
        "channel_latencies_json": json.dumps(channel_latencies),
        "status": "SUCCESS",
        "error_msg": "",
    }

    if not candidates_list:
        return pd.DataFrame(), pd.DataFrame(), status_meta, df_universe, df_diag_universe

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

    # Precompute ideal DCG@10 and total relevant docs
    n_rel = sum(1 for r in qrels.values() if r >= 1)
    ideal_rels = sorted([r for r in qrels.values() if r >= 1], reverse=True)[:10]
    idcg10 = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    delta_thresh = thresholds.get("delta", DEFAULT_DELTA)
    epsilon_thresh = thresholds.get("epsilon", EPSILON)
    tau_thresh = thresholds.get("tau", TAU)

    # 5. Stream batch retrieval execution and evaluate metrics chunk-by-chunk
    parent_records = []
    cutoff_records = []
    cand_variant_outcomes = defaultdict(list)

    for i in range(0, len(variants_to_eval), chunk_size):
        check_memory_watchdog(frozen_config)
        chunk = variants_to_eval[i:i + chunk_size]
        df_chunk = pd.DataFrame([{"qid": item["var_id"], "query_toks": item["query_toks"]} for item in chunk])
        res_chunk = bm25.transform(df_chunk)

        # Map var_id to filtered top-1000 document list for this chunk only
        chunk_results = defaultdict(list)
        if not res_chunk.empty:
            q_arr = res_chunk["qid"].values
            d_arr = res_chunk["docno"].values
            for q_id, doc in zip(q_arr, d_arr):
                chunk_results[q_id].append(str(doc))

        # Evaluate each variant in this chunk immediately
        for item in chunk:
            var_id = item["var_id"]
            cand = item["candidate"]
            w = item["weight"]

            raw_docs = chunk_results.get(var_id, [])
            if exclusions:
                exp_docs = [d for d in raw_docs if d not in exclusions][:1000]
            else:
                exp_docs = raw_docs[:1000]
            exp_ranks = {docno: r for r, docno in enumerate(exp_docs, 1)}

            dcg = sum(qrels.get(d, 0.0) / math.log2(idx + 2) for idx, d in enumerate(exp_docs[:10]))
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

            # Store variant outcome for cutoff entries (only track ranks of gold relevant docs for memory efficiency)
            gold_ranks = {did: exp_ranks.get(did, 9999) for did in gold_dids}
            cand_variant_outcomes[cand].append({
                "weight": w,
                "gold_ranks": gold_ranks,
                "d_ndcg10": d_ndcg10,
                "cutoff_net": cutoff_net,
            })

        del chunk_results, res_chunk, df_chunk


    # Deduplicate cutoff entries over weights for each candidate term
    for cand, v_list in cand_variant_outcomes.items():
        for k in TRACKED_CUTOFFS:
            for did in gold_dids:
                b_r = base_ranks.get(did, 9999)
                if b_r > k:  # Document was strictly outside baseline top-K
                    raw_in = False
                    safe_in = False
                    for v in v_list:
                        e_r = v.get("gold_ranks", v.get("exp_ranks", {})).get(did, 9999)
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

    return parent_df, cutoff_df, status_meta, df_universe, df_diag_universe


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
    bge_model: str = "BAAI/bge-small-en-v1.5",
    qids_manifest: Optional[str] = None,
    force_rebuild_sidecars: bool = False,
    explicit_qids: Optional[List[str]] = None,
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

    # Compute dataset and qrels content hashes from consumed input paths
    consumed_inputs = BenchmarkLoader.get_consumed_input_paths(dataset)
    input_hashes = {}
    for name, path in consumed_inputs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"FATAL: Required input file missing: {path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        input_hashes[name] = hasher.hexdigest()

    if "examples" in input_hashes:
        dataset_hash = input_hashes["examples"]
        qrels_hash = input_hashes["examples"]
    else:
        dataset_hash = input_hashes.get("queries", "")
        qrels_hash = input_hashes.get("qrels", "")

    # Recompute and validate recorded index files at runtime using streaming chunks
    idx_manifest_path = os.path.join(os.path.dirname(idx_path), "index_manifest.json")
    if not os.path.exists(idx_manifest_path):
        raise FileNotFoundError(f"FATAL: index_manifest.json not found at {idx_manifest_path}!")

    manifest_hasher = hashlib.sha256()
    with open(idx_manifest_path, "rb") as f:
        while chunk := f.read(65536):
            manifest_hasher.update(chunk)
    index_manifest_file_sha256 = manifest_hasher.hexdigest()

    with open(idx_manifest_path, "r", encoding="utf-8") as f:
        idx_manifest = json.load(f)

    for fname, exp_hash in idx_manifest.get("index_files", {}).items():
        fpath = os.path.join(os.path.dirname(idx_path), fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"FATAL: Required index file missing: {fpath}")
        hasher = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        if actual_hash != exp_hash:
            raise RuntimeError(f"FATAL: Index file {fname} has changed! Actual {actual_hash} != recorded {exp_hash}")

    index_manifest_hash = idx_manifest.get("manifest_sha256", "")

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
    if explicit_qids is not None:
        frozen_qids = explicit_qids
    elif frozen_config and "query_manifests" in frozen_config:
        q_manifest = frozen_config["query_manifests"]
        if qids_manifest and qids_manifest in q_manifest:
            frozen_qids = q_manifest[qids_manifest].get(dataset)
        elif sample_size == 1 and "micro_smoke_qids" in q_manifest:
            frozen_qids = q_manifest["micro_smoke_qids"].get(dataset)
        elif sample_size == 10 and "probe_40_qids" in q_manifest:
            frozen_qids = q_manifest["probe_40_qids"].get(dataset)
        elif sample_size == 50 and "dev_200_qids" in q_manifest:
            frozen_qids = q_manifest["dev_200_qids"].get(dataset)

    if frozen_qids:
        validate_manifest_qids(frozen_qids, dataset)
        missing_qids = set(frozen_qids) - set(q_by_id.keys())
        if missing_qids:
            raise RuntimeError(f"FATAL: Dataset '{dataset}' is missing {len(missing_qids)} QIDs from frozen manifest: {list(missing_qids)[:5]}")
        sampled_queries = [q_by_id[qid] for qid in frozen_qids]
        print(f"  [Queries] Loaded {len(sampled_queries)} queries from frozen manifest for {dataset}.")
        if dataset == "bright_aops" and qids_manifest == "micro_smoke_qids":
            q_ex = sampled_queries[0].get("excluded_doc_ids", sampled_queries[0].get("exclusions", []))
            assert len(q_ex) > 0, f"FATAL: Micro-smoke BRIGHT query {frozen_qids[0]} has 0 exclusions!"
    else:
        rng = random.Random(seed)
        if len(all_queries) <= sample_size:
            sampled_queries = list(all_queries)
        else:
            sampled_queries = rng.sample(all_queries, sample_size)
        print(f"  [Queries] Sampled {len(sampled_queries)} queries using seed={seed}.")

    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=1000)

    # 4. Load all 4 Gate 1 Sidecars with frozen config wiring
    sidecar_mgr = Gate1SidecarManager(config=frozen_config, bge_model_name=bge_model)
    bge_sidecar = sidecar_mgr.build_or_load_bge_sidecar(
        dataset, pool_terms, num_docs=num_docs, encoder=encoder, device=device, force_rebuild=force_rebuild_sidecars
    )
    pool_embeddings = bge_sidecar["pool_embeddings"]
    surf_to_idx = bge_sidecar["surf_to_idx"]

    ppmi_top_m = frozen_config.get("sidecars", {}).get("ppmi", {}).get("top_m", 600) if frozen_config else 600
    ppmi_sidecar = sidecar_mgr.build_or_load_bounded_ppmi_sidecar(
        dataset, index, pool_terms, num_docs=num_docs, top_m=ppmi_top_m, force_rebuild=force_rebuild_sidecars
    )
    anchor_ppmi = ppmi_sidecar["anchor_ppmi"]

    acronym_sidecar = sidecar_mgr.build_or_load_acronym_rescue_sidecar(
        dataset, pool_terms, num_docs=num_docs, force_rebuild=force_rebuild_sidecars
    )
    acronym_to_pool = acronym_sidecar["acronym_to_pool"]

    lex_sidecar = sidecar_mgr.build_or_load_sparse_lexical_sidecar(
        dataset, pool_terms, num_docs=num_docs, force_rebuild=force_rebuild_sidecars
    )
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
    
    rrf_core3_k = int(frozen_config["rrf_policies"]["rrf_core3"]["k"]) if frozen_config else 60
    rrf_ext_k = int(frozen_config["rrf_policies"]["rrf_extended"]["k"]) if frozen_config else 60
    rrf_core3_proposer = RRFHybridProposer(k=rrf_core3_k)
    rrf_ext_proposer = RRFHybridProposer(k=rrf_ext_k)

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
        parent_shard, cutoff_shard, status_shard, universe_shard, diag_universe_shard = get_shard_paths(shards_dir, dataset, qid)

        if resume and os.path.exists(parent_shard) and os.path.exists(cutoff_shard) and os.path.exists(status_shard) and os.path.exists(universe_shard):
            if measure_fidelity and not os.path.exists(diag_universe_shard):
                pass
            else:
                try:
                    st_data = pq.read_table(status_shard).to_pydict()
                    st_status = {k: v[0] for k, v in st_data.items()}
                    expected_hashes = {
                        "config_hash": config_hash,
                        "dataset_hash": dataset_hash,
                        "qrels_hash": qrels_hash,
                        "pool_hash": pool_hash,
                    }
                    validate_shard_hashes(st_status, expected_hashes, qid)
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
        q_exclusions = set(str(x) for x in q.get("excluded_doc_ids", q.get("exclusions", [])))
        if q_exclusions:
            k_fetch_q = min(num_docs, 1000 + len(q_exclusions))
            bm25_q = pt.terrier.Retriever(index, wmodel="BM25", num_results=k_fetch_q)
        else:
            bm25_q = bm25

        df_p, df_c, q_status, df_u, df_diag_u = evaluate_query_gate1(
            dataset=dataset,
            query_obj=q,
            index=index,
            analyzer=analyzer,
            bm25=bm25_q,
            proposers=proposers,
            rrf_core3_proposer=rrf_core3_proposer,
            rrf_ext_proposer=rrf_ext_proposer,
            pool_set=pool_set,
            phase1_cands_for_q=phase1_map.get(qid, set()),
            weights=weights,
            thresholds=thresholds,
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            qrels_hash=qrels_hash,
            pool_hash=pool_hash,
            chunk_size=chunk_size,
            frozen_config=frozen_config,
        )

        t_q = time.perf_counter() - t0_q

        # Save shards
        df_p.to_parquet(parent_shard, index=False)
        df_c.to_parquet(cutoff_shard, index=False)
        pd.DataFrame([q_status]).to_parquet(status_shard, index=False)
        df_u.to_parquet(universe_shard, index=False)
        if df_diag_u is not None:
            df_diag_u.to_parquet(diag_universe_shard, index=False)

        total_variants += len(df_p)
        total_cutoff_entries += len(df_c)
        v_rate = len(df_p) / max(t_q, 0.001)
        print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Evaluated {len(df_p)} variants ({len(df_c)} cutoff entries) in {t_q:.2f}s ({v_rate:.1f} var/s)")

        del df_p, df_c, df_u, df_diag_u
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
            q_tensor = torch.tensor(q_emb, device=device, dtype=pool_embeddings.dtype).unsqueeze(0)
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
            _, _, s_shard, _, _ = get_shard_paths(shards_dir, dataset, qid)
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
            "ppmi_disk_mb": round(os.path.getsize(
                os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bounded_ppmi.parquet")
                if os.path.exists(os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bounded_ppmi.parquet"))
                else os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bounded_ppmi.json")
            ) / (1024 ** 2), 2),
            "bge_disk_mb": round(os.path.getsize(os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_bge_sidecar.pt")) / (1024 ** 2), 2),
            "lexical_disk_mb": round(os.path.getsize(
                os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_lexical_profiles.parquet")
            ) / (1024 ** 2), 2) if os.path.exists(os.path.join(sidecar_mgr.cache_dir, f"{safe_ds}_lexical_profiles.parquet")) else 0.0,
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
        "input_hashes": input_hashes,
        "index_manifest_sha256": index_manifest_file_sha256,
        "sidecars_provenance": {
            "bge": bge_sidecar.get("metadata", {}),
            "ppmi": ppmi_sidecar.get("metadata", {}),
            "acronym": acronym_sidecar.get("metadata", {}),
            "lexical": lex_sidecar.get("metadata", {}),
        },
    }


def assemble_shards_to_master(
    shards_dir: str,
    shard_type: str,
    output_path: str,
    expected_qids: Optional[Set[str]] = None,
    filter_to_universe_df: Optional[pd.DataFrame] = None,
    filter_qids: Optional[Set[str]] = None,
    clear_diagnostic_cols: bool = False,
):
    """
    Streams individual query Parquet shards into a single master Parquet file
    using pyarrow.ParquetWriter with fail-closed schema and QID verification.
    shard_type: 'audit', 'cutoff', 'status', 'universe', or 'diag_universe'
    """
    if shard_type == "cutoff":
        all_files = sorted(glob.glob(os.path.join(shards_dir, "*_cutoff_entries.parquet")))
    elif shard_type == "status":
        all_files = sorted(glob.glob(os.path.join(shards_dir, "*_status.parquet")))
    elif shard_type == "universe":
        all_files = sorted([
            f for f in glob.glob(os.path.join(shards_dir, "*_universe.parquet"))
            if not f.endswith("_diag_universe.parquet")
        ])
    elif shard_type == "diag_universe":
        all_files = sorted(glob.glob(os.path.join(shards_dir, "*_diag_universe.parquet")))
    elif shard_type == "audit":
        all_files = sorted([
            f for f in glob.glob(os.path.join(shards_dir, "*.parquet"))
            if not f.endswith("_cutoff_entries.parquet") and not f.endswith("_status.parquet") and not f.endswith("_universe.parquet") and not f.endswith("_diag_universe.parquet")
        ])
    else:
        raise ValueError(f"Unknown shard_type: {shard_type}")

    if not all_files:
        print(f"No shards found for type '{shard_type}' in {shards_dir}.")
        return

    valid_pairs = None
    if filter_to_universe_df is not None:
        valid_pairs = set(zip(
            filter_to_universe_df["qid"].astype(str),
            filter_to_universe_df["candidate_term"].astype(str),
        ))

    print(f"Streaming {len(all_files)} {shard_type} shards to {output_path} via pyarrow.ParquetWriter...")
    writer = None
    total_rows = 0
    unified_schema = None
    assembled_qids = set()

    for fpath in all_files:
        try:
            table = pq.read_table(fpath)
            if table.num_rows == 0:
                del table
                continue
            if "qid" in table.column_names:
                shard_qid = str(table["qid"][0].as_py())
                if filter_qids is not None and shard_qid not in filter_qids:
                    del table
                    continue

            if (valid_pairs is not None or clear_diagnostic_cols) and "qid" in table.column_names and "candidate_term" in table.column_names:
                df_shard = table.to_pandas()
                if valid_pairs is not None:
                    mask = [
                        (str(q), str(c)) in valid_pairs
                        for q, c in zip(df_shard["qid"], df_shard["candidate_term"])
                    ]
                    df_shard = df_shard[mask]
                    if df_shard.empty:
                        del table
                        continue
                if clear_diagnostic_cols and "live_ppmi_rank" in df_shard.columns:
                    df_shard["live_ppmi_rank"] = np.nan
                df_shard.reset_index(drop=True, inplace=True)
                table = pa.Table.from_pandas(df_shard, schema=unified_schema if unified_schema else None, preserve_index=False)

            if unified_schema is None:
                unified_schema = table.schema
            if writer is None:
                writer = pq.ParquetWriter(output_path, unified_schema, compression="zstd")
            elif table.schema != unified_schema:
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
    parser.add_argument("--force-rebuild-sidecars", action="store_true", help="Force rebuild all sidecars before running evaluation")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5", help="BGE model name")
    parser.add_argument("--config-hash", type=str, default="gate1_core_dev_v1", help="Configuration hash")
    parser.add_argument("--qids-manifest", type=str, default=None, help="Name of query manifest to use from frozen config (e.g. micro_smoke_qids, probe_40_qids, dev_200_qids)")
    parser.add_argument("--probe-loss-gate-path", type=str, default="results/gate1_selection/probe_40_eval/checkpoint_b_loss_gate.json", help="Path to probe Checkpoint B loss gate artifact")
    parser.add_argument("--probe-manifest-path", type=str, default="results/gate1_selection/probe_40_eval/run_manifest.json", help="Path to probe run manifest artifact")
    parser.add_argument("--skip-probe-gate", action="store_true", help="Bypass probe gate preflight verification (for diagnostic testing only)")
    parser.add_argument("--staged-evaluation", action="store_true", help="Run 40-query probe first, enforce Checkpoint B gate in-process, then evaluate remaining 160 queries if passed.")
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
        validate_gate1_config(frozen_config)
        print(f"Loaded and validated frozen configuration from {args.frozen_config_path} (SHA-256: {config_hash[:12]}...)")
        
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

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Check if staged evaluation should be performed
    is_staged_run = (
        args.staged_evaluation
        or (args.qids_manifest == "dev_200_qids" and not args.skip_probe_gate and not os.path.exists(args.probe_loss_gate_path))
    )

    if is_staged_run:
        print("\n" + "=" * 70)
        print(">>> [STAGED EVALUATION] Engaged: 40-Query Probe -> In-Process Checkpoint B -> 160-Query Completion")
        print("=" * 70)

        if not frozen_config or "query_manifests" not in frozen_config:
            raise ValueError("FATAL: Staged evaluation requires a frozen config containing query_manifests with 'probe_40_qids' and 'dev_200_qids'!")
        q_manifest = frozen_config["query_manifests"]
        if "probe_40_qids" not in q_manifest or "dev_200_qids" not in q_manifest:
            raise ValueError("FATAL: query_manifests must contain both 'probe_40_qids' and 'dev_200_qids' for staged evaluation!")

        # Preflight: Verify pool hashes, index hashes, corpus source hashes, and sidecars on disk
        sidecar_cache_dir = os.path.abspath("data/cache/canonical_pools")
        for ds in datasets:
            safe_ds = ds.lower().replace("-", "_")
            cur_pool_terms = load_canonical_pool_terms(ds)
            cur_pool_sha = compute_pool_sha256(cur_pool_terms)
            cur_idx_hash = compute_index_hash(ds)
            cur_src_hash = compute_corpus_source_hash(ds)

            bge_path = os.path.join(sidecar_cache_dir, f"{safe_ds}_bge_sidecar.pt")
            ppmi_parquet = os.path.join(sidecar_cache_dir, f"{safe_ds}_bounded_ppmi.parquet")
            ppmi_json = os.path.join(sidecar_cache_dir, f"{safe_ds}_bounded_ppmi.json")
            ppmi_path = ppmi_parquet if os.path.exists(ppmi_parquet) else ppmi_json
            acronym_path = os.path.join(sidecar_cache_dir, f"{safe_ds}_acronym_rescue.json")
            lex_parquet = os.path.join(sidecar_cache_dir, f"{safe_ds}_lexical_profiles.parquet")
            lex_idx = os.path.join(sidecar_cache_dir, f"{safe_ds}_lexical_profiles_idx")
            lex_path = lex_parquet if os.path.exists(lex_parquet) else lex_idx

            for sidecar_name, sidecar_file in [
                ("bge", bge_path),
                ("ppmi", ppmi_path),
                ("acronym", acronym_path),
                ("lexical", lex_path),
            ]:
                if not os.path.exists(sidecar_file):
                    raise FileNotFoundError(
                        f"FATAL: Staged run preflight failed: Required {sidecar_name} sidecar artifact not found on disk at '{sidecar_file}'!"
                    )

        # Force rebuild sidecars if requested
        if args.force_rebuild_sidecars:
            print("\n[ForceRebuild] Rebuilding sidecars for all datasets before evaluation...")
            sidecar_mgr = Gate1SidecarManager(config=frozen_config)
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            enc_temp = SentenceTransformer(args.bge_model, device=device)
            for ds in datasets:
                safe_ds = ds.lower().replace("-", "_")
                pool_path = f"data/cache/canonical_pools/{safe_ds}_canonical_pool.json"
                if not os.path.exists(pool_path):
                    raise FileNotFoundError(f"Canonical pool missing for {ds}: {pool_path}")
                with open(pool_path, "r", encoding="utf-8") as f:
                    pool_terms = json.load(f)["terms"]
                idx_path = f"data/cache/terrier_indices/{safe_ds}_default/data.properties"
                idx_temp = pt.IndexFactory.of(os.path.abspath(idx_path))
                num_docs = idx_temp.getCollectionStatistics().getNumberOfDocuments()
                sidecar_mgr.build_or_load_bge_sidecar(ds, pool_terms, num_docs=num_docs, encoder=enc_temp, device=device, force_rebuild=True)
                top_m = int(frozen_config.get("sidecars", {}).get("ppmi", {}).get("top_m", 600)) if frozen_config else 600
                sidecar_mgr.build_or_load_bounded_ppmi_sidecar(ds, idx_temp, pool_terms, num_docs=num_docs, top_m=top_m, force_rebuild=True)
                sidecar_mgr.build_or_load_acronym_rescue_sidecar(ds, pool_terms, num_docs=num_docs, force_rebuild=True)
                sidecar_mgr.build_or_load_sparse_lexical_sidecar(ds, pool_terms, num_docs=num_docs, force_rebuild=True)
                del idx_temp
            del enc_temp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[ForceRebuild] Completed sidecar rebuilds successfully.\n")

        # Load Phase 1 candidate audit if available
        phase1_audit_df = None
        if args.phase1_audit and os.path.exists(args.phase1_audit):
            print(f"Loading Phase 1 candidate audit from {args.phase1_audit}...")
            phase1_audit_df = pd.read_parquet(args.phase1_audit)
            print(f"Loaded {len(phase1_audit_df):,} Phase 1 audit rows.")

        # Initialize encoder once on device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading encoder {args.bge_model} on {device}...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.bge_model, device=device)

        # -------------------------------------------------------------
        # STAGE 1: 40 Probe Queries with Diagnostic Channel (LivePPMI)
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print(">>> [STAGE 1/2] Evaluating 40 Probe Queries (Diagnostic Fidelity Mode)")
        print("=" * 70)

        probe_summaries = []
        all_probe_qids = set()
        for ds in datasets:
            probe_qids = q_manifest["probe_40_qids"].get(ds, [])
            all_probe_qids.update(probe_qids)
            print(f"\n[Stage 1] Running {len(probe_qids)} probe queries for {ds} with measure_fidelity=True...")
            ds_meta = run_dataset_gate1_evaluation(
                dataset=ds,
                output_dir=args.output_dir,
                weights=weights,
                sample_size=len(probe_qids),
                seed=seed,
                chunk_size=args.chunk_size,
                resume=not args.no_resume,
                phase1_audit_df=phase1_audit_df,
                frozen_config=frozen_config,
                encoder=encoder,
                device=device,
                config_hash=config_hash,
                measure_fidelity=True,
                bge_model=args.bge_model,
                qids_manifest="probe_40_qids",
                force_rebuild_sidecars=args.force_rebuild_sidecars,
                explicit_qids=probe_qids,
            )
            probe_summaries.append(ds_meta)

        # Assemble Stage 1 Probe artifacts
        shards_dir = os.path.join(args.output_dir, "shards")
        probe_audit_path = os.path.join(args.output_dir, "diagnostic_candidate_audit.parquet")
        probe_cutoff_path = os.path.join(args.output_dir, "diagnostic_cutoff_entries.parquet")
        probe_cutoff_legacy_path = os.path.join(args.output_dir, "probe_cutoff_entries.parquet")
        probe_status_path = os.path.join(args.output_dir, "probe_query_status.parquet")
        probe_universe_path = os.path.join(args.output_dir, "probe_reference_universe.parquet")
        probe_diag_universe_path = os.path.join(args.output_dir, "diagnostic_universe.parquet")

        print("\nAssembling Stage 1 probe shards for Checkpoint B verification...")
        assemble_shards_to_master(shards_dir, "audit", probe_audit_path, expected_qids=all_probe_qids, filter_qids=all_probe_qids)
        assemble_shards_to_master(shards_dir, "cutoff", probe_cutoff_path, filter_qids=all_probe_qids)
        assemble_shards_to_master(shards_dir, "cutoff", probe_cutoff_legacy_path, filter_qids=all_probe_qids)
        assemble_shards_to_master(shards_dir, "status", probe_status_path, expected_qids=all_probe_qids, filter_qids=all_probe_qids)
        assemble_shards_to_master(shards_dir, "universe", probe_universe_path, expected_qids=all_probe_qids, filter_qids=all_probe_qids)
        assemble_shards_to_master(shards_dir, "diag_universe", probe_diag_universe_path, expected_qids=all_probe_qids, filter_qids=all_probe_qids)

        # Verify Stage 1 action coverage against diagnostic universe
        df_probe_audit = pd.read_parquet(probe_audit_path)
        df_probe_diag_univ = pd.read_parquet(probe_diag_universe_path)
        print("\nValidating probe action coverage against diagnostic reference universe...")
        validate_action_coverage(df_probe_audit, df_probe_diag_univ, weights)
        print("Probe action coverage PASSED (100% Cartesian set equality).")

        # -------------------------------------------------------------
        # IN-PROCESS CHECKPOINT B LOSS GATE ENFORCEMENT
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print(">>> [CHECKPOINT B] Enforcing Operational Loss Gate In-Process")
        print("=" * 70)
        from compile_gate1_research_tables import Gate1TableCompiler
        probe_compiler = Gate1TableCompiler(
            audit_parquet_path=probe_audit_path,
            cutoff_parquet_path=probe_cutoff_path,
            output_dir=os.path.join(args.output_dir, "probe_artifacts"),
            universe_parquet_path=probe_universe_path,
            diag_universe_parquet_path=probe_diag_universe_path,
            frozen_config_path=args.frozen_config_path,
        )
        loss_gate_results = probe_compiler.enforce_operational_loss_gate(budget_l=200)
        loss_gate_path = os.path.join(args.output_dir, "checkpoint_b_loss_gate.json")
        with open(loss_gate_path, "w", encoding="utf-8") as f:
            json.dump(loss_gate_results, f, indent=2)
        print(f"Saved Checkpoint B Loss Gate artifact -> {loss_gate_path}")

        if not loss_gate_results.get("gate_passed", False):
            print("\n" + "!" * 70)
            print("FATAL: Checkpoint B operational-loss gate FAILED!")
            print(f"Corpus-macro nDCG loss: {loss_gate_results.get('corpus_macro_delta_ndcg10_loss')} (threshold: {loss_gate_results.get('max_oracle_loss_corpus_macro_threshold')})")
            print(f"Corpus-macro doc loss:  {loss_gate_results.get('corpus_macro_raw_doc_opp_recall1000_loss')} (threshold: {loss_gate_results.get('max_doc_opp_recall_loss_corpus_macro_threshold')})")
            for ds_name, p_res in loss_gate_results.get("per_corpus_results", {}).items():
                print(f"  - {ds_name}: nDCG loss={p_res['mean_delta_ndcg10_loss']} (passed={p_res['ndcg_loss_passed']}), doc loss={p_res['mean_raw_doc_opp_recall1000_loss']} (passed={p_res['doc_loss_passed']})")
            print("Stage 2 is BLOCKED. Halting execution before spending compute on the remaining 160 queries.")
            print("!" * 70 + "\n")
            raise RuntimeError("FATAL: Checkpoint B operational-loss gate failed during Stage 1 probe evaluation; full run is blocked.")

        print("\n>>> Checkpoint B operational-loss gate PASSED! Proceeding to Stage 2.\n")

        # -------------------------------------------------------------
        # STAGE 2: Remaining 160 Queries (Operational Channels Only)
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print(">>> [STAGE 2/2] Evaluating Remaining 160 Queries (Operational Channels Only)")
        print("=" * 70)

        all_200_qids = set()
        stage2_summaries = []
        for ds in datasets:
            full_qids = q_manifest["dev_200_qids"].get(ds, [])
            all_200_qids.update(full_qids)
            probe_qids_ds = set(q_manifest["probe_40_qids"].get(ds, []))
            remaining_qids = [q for q in full_qids if q not in probe_qids_ds]

            print(f"\n[Stage 2] Running {len(remaining_qids)} remaining queries for {ds} (measure_fidelity=False)...")
            ds_meta = run_dataset_gate1_evaluation(
                dataset=ds,
                output_dir=args.output_dir,
                weights=weights,
                sample_size=len(remaining_qids),
                seed=seed,
                chunk_size=args.chunk_size,
                resume=not args.no_resume,
                phase1_audit_df=phase1_audit_df,
                frozen_config=frozen_config,
                encoder=encoder,
                device=device,
                config_hash=config_hash,
                measure_fidelity=False,
                bge_model=args.bge_model,
                qids_manifest=None,
                force_rebuild_sidecars=False,
                explicit_qids=remaining_qids,
            )
            stage2_summaries.append(ds_meta)

        # -------------------------------------------------------------
        # MASTER ASSEMBLY: Full 200 Queries Operational Audit
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print(">>> Assembling Master 200-Query Operational Artifacts")
        print("=" * 70)

        master_parent = os.path.join(args.output_dir, "gate1_candidate_audit.parquet")
        master_cutoff = os.path.join(args.output_dir, "gate1_cutoff_entries.parquet")
        master_status = os.path.join(args.output_dir, "query_status.parquet")
        master_universe = os.path.join(args.output_dir, "reference_universe.parquet")

        assemble_shards_to_master(shards_dir, shard_type="universe", output_path=master_universe, expected_qids=all_200_qids)
        df_master_univ = pd.read_parquet(master_universe)

        assemble_shards_to_master(
            shards_dir,
            shard_type="audit",
            output_path=master_parent,
            expected_qids=all_200_qids,
            filter_to_universe_df=df_master_univ,
            clear_diagnostic_cols=True,
        )
        assemble_shards_to_master(
            shards_dir,
            shard_type="cutoff",
            output_path=master_cutoff,
            filter_to_universe_df=df_master_univ,
        )
        assemble_shards_to_master(shards_dir, shard_type="status", output_path=master_status, expected_qids=all_200_qids)

        print("\nValidating action coverage between 200-query candidate audit and reference universe...")
        df_audit_master = pd.read_parquet(master_parent)
        validate_action_coverage(df_audit_master, df_master_univ, weights)
        print("Action coverage verification PASSED (exact Cartesian set equality across all 200 queries).")

        # Combine dataset summaries
        meta_summaries = []
        probe_summary_map = {s["dataset"]: s for s in probe_summaries}
        stage2_summary_map = {s["dataset"]: s for s in stage2_summaries}
        for ds in datasets:
            p_s = probe_summary_map.get(ds, {})
            s2_s = stage2_summary_map.get(ds, {})
            combined_s = {
                "dataset": ds,
                "num_docs": p_s.get("num_docs", s2_s.get("num_docs", 0)),
                "pool_size": p_s.get("pool_size", s2_s.get("pool_size", 0)),
                "pool_sha256": p_s.get("pool_sha256", s2_s.get("pool_sha256", "")),
                "num_queries": p_s.get("num_queries", 0) + s2_s.get("num_queries", 0),
                "total_variants": p_s.get("total_variants", 0) + s2_s.get("total_variants", 0),
                "total_cutoff_entries": p_s.get("total_cutoff_entries", 0) + s2_s.get("total_cutoff_entries", 0),
                "retrieval_time_sec": round(p_s.get("retrieval_time_sec", 0.0) + s2_s.get("retrieval_time_sec", 0.0), 2),
                "fidelity_meta": p_s.get("fidelity_meta", {}),
                "input_hashes": p_s.get("input_hashes", s2_s.get("input_hashes", {})),
                "index_manifest_sha256": p_s.get("index_manifest_sha256", s2_s.get("index_manifest_sha256", "")),
                "sidecars_provenance": p_s.get("sidecars_provenance", s2_s.get("sidecars_provenance", {})),
            }
            meta_summaries.append(combined_s)

    else:
        # Full-run preflight verification: Checkpoint B gate must be passed and provenance must match
        is_full_run = (args.sample_size > 10 or args.qids_manifest == "dev_200_qids")
        if is_full_run and not args.skip_probe_gate:
            print("\n[Preflight] Verifying Checkpoint B operational loss gate before launching full evaluation run...")
            if not os.path.exists(args.probe_loss_gate_path):
                raise FileNotFoundError(
                    f"FATAL: Full run preflight failed: Checkpoint B loss gate artifact not found at '{args.probe_loss_gate_path}'. "
                    f"The 40-query probe evaluation and table compilation must be completed first."
                )
            with open(args.probe_loss_gate_path, "r", encoding="utf-8") as f:
                probe_gate_data = json.load(f)
            if not probe_gate_data.get("gate_passed", False):
                raise RuntimeError(
                    f"FATAL: Full run preflight failed: Checkpoint B operational loss gate in '{args.probe_loss_gate_path}' is FAILED! "
                    f"The full evaluation cannot proceed until the probe gate passes."
                )

            # Verify matching provenance from probe run manifest
            if not os.path.exists(args.probe_manifest_path):
                raise FileNotFoundError(
                    f"FATAL: Full run preflight failed: Probe run manifest not found at '{args.probe_manifest_path}'."
                )
            with open(args.probe_manifest_path, "r", encoding="utf-8") as f:
                probe_manifest = json.load(f)

            # 1. Config hash check
            if probe_manifest.get("config_hash") != config_hash:
                raise ValueError(
                    f"FATAL: Full run preflight failed: Config hash mismatch! "
                    f"Probe run had config_hash='{probe_manifest.get('config_hash')}', but current run has config_hash='{config_hash}'."
                )

            # Build dataset map from probe manifest
            probe_ds_map = {
                s["dataset"]: s for s in probe_manifest.get("dataset_summaries", [])
            }
            if not probe_ds_map:
                raise ValueError(
                    f"FATAL: Full run preflight failed: No dataset summaries found in probe run manifest '{args.probe_manifest_path}'!"
                )

            sidecar_cache_dir = os.path.abspath("data/cache/canonical_pools")

            for ds in datasets:
                if ds not in probe_ds_map:
                    raise ValueError(
                        f"FATAL: Full run preflight failed: Dataset '{ds}' missing from probe run manifest dataset_summaries!"
                    )
                ds_summary = probe_ds_map[ds]
                safe_ds = ds.lower().replace("-", "_")

                # 2. Pool hash check
                cur_pool_terms = load_canonical_pool_terms(ds)
                cur_pool_sha = compute_pool_sha256(cur_pool_terms)
                probe_pool_sha = ds_summary.get("pool_sha256")
                if not probe_pool_sha:
                    raise ValueError(f"FATAL: Full run preflight failed: Missing pool_sha256 for '{ds}' in probe manifest!")
                if probe_pool_sha != cur_pool_sha:
                    raise ValueError(
                        f"FATAL: Full run preflight failed: Pool hash mismatch for {ds}! "
                        f"Probe had '{probe_pool_sha}', current is '{cur_pool_sha}'."
                    )

                # 3. Index hash check
                cur_idx_hash = compute_index_hash(ds)
                probe_idx_hash = ds_summary.get("index_manifest_sha256")
                if not probe_idx_hash:
                    raise ValueError(f"FATAL: Full run preflight failed: Missing index_manifest_sha256 for '{ds}' in probe manifest!")
                if probe_idx_hash != cur_idx_hash:
                    raise ValueError(
                        f"FATAL: Full run preflight failed: Index hash mismatch for {ds}! "
                        f"Probe had '{probe_idx_hash}', current is '{cur_idx_hash}'."
                    )

                # 4. Corpus source hash check
                cur_src_hash = compute_corpus_source_hash(ds)
                probe_sidecars = ds_summary.get("sidecars_provenance", {})
                probe_src_hash = probe_sidecars.get("bge", {}).get("corpus_source_hash")
                if not probe_src_hash:
                    raise ValueError(f"FATAL: Full run preflight failed: Missing corpus_source_hash for '{ds}' in probe manifest!")
                if probe_src_hash != cur_src_hash:
                    raise ValueError(
                        f"FATAL: Full run preflight failed: Corpus source hash mismatch for {ds}! "
                        f"Probe had '{probe_src_hash}', current is '{cur_src_hash}'."
                    )

                # 5. Sidecar provenance check (BGE, PPMI, Acronym, Lexical)
                bge_path = os.path.join(sidecar_cache_dir, f"{safe_ds}_bge_sidecar.pt")
                ppmi_parquet = os.path.join(sidecar_cache_dir, f"{safe_ds}_bounded_ppmi.parquet")
                ppmi_json = os.path.join(sidecar_cache_dir, f"{safe_ds}_bounded_ppmi.json")
                ppmi_path = ppmi_parquet if os.path.exists(ppmi_parquet) else ppmi_json

                acronym_path = os.path.join(sidecar_cache_dir, f"{safe_ds}_acronym_rescue.json")
                lex_parquet = os.path.join(sidecar_cache_dir, f"{safe_ds}_lexical_profiles.parquet")
                lex_idx = os.path.join(sidecar_cache_dir, f"{safe_ds}_lexical_profiles_idx")
                lex_path = lex_parquet if os.path.exists(lex_parquet) else lex_idx

                for sidecar_name, sidecar_file in [
                    ("bge", bge_path),
                    ("ppmi", ppmi_path),
                    ("acronym", acronym_path),
                    ("lexical", lex_path),
                ]:
                    if not os.path.exists(sidecar_file):
                        raise FileNotFoundError(
                            f"FATAL: Full run preflight failed: Required {sidecar_name} sidecar artifact not found on disk at '{sidecar_file}'!"
                        )

                # Validate BGE sidecar metadata
                bge_meta = torch.load(bge_path, map_location="cpu")
                if isinstance(bge_meta, dict) and "metadata" in bge_meta:
                    bge_meta = bge_meta["metadata"]
                if bge_meta.get("pool_sha256") != cur_pool_sha:
                    raise ValueError(f"FATAL: Full run preflight failed: BGE sidecar pool_sha256 mismatch for {ds}!")
                if bge_meta.get("corpus_source_hash") != cur_src_hash:
                    raise ValueError(f"FATAL: Full run preflight failed: BGE sidecar corpus_source_hash mismatch for {ds}!")

                # Validate PPMI sidecar metadata (supports both Parquet and JSON)
                if ppmi_path.endswith(".parquet"):
                    ppmi_tbl = pq.read_table(ppmi_path)
                    ppmi_sm = ppmi_tbl.schema.metadata or {}
                    ppmi_meta = json.loads(ppmi_sm.get(b"sidecar_metadata", b"{}").decode("utf-8"))
                else:
                    with open(ppmi_path, "r", encoding="utf-8") as f:
                        ppmi_data = json.load(f)
                    ppmi_meta = ppmi_data.get("metadata", {})

                if ppmi_meta.get("pool_sha256") != cur_pool_sha:
                    raise ValueError(f"FATAL: Full run preflight failed: PPMI sidecar pool_sha256 mismatch for {ds}!")
                if ppmi_meta.get("corpus_source_hash") != cur_src_hash:
                    raise ValueError(f"FATAL: Full run preflight failed: PPMI sidecar corpus_source_hash mismatch for {ds}!")
                expected_top_m = int(frozen_config.get("sidecars", {}).get("ppmi", {}).get("top_m", 600)) if frozen_config else 600
                if ppmi_meta.get("top_m") != expected_top_m:
                    raise ValueError(
                        f"FATAL: Full run preflight failed: PPMI sidecar top_m mismatch for {ds}! "
                        f"Sidecar has {ppmi_meta.get('top_m')}, config expects {expected_top_m}."
                    )
                if probe_sidecars.get("ppmi", {}).get("top_m") != ppmi_meta.get("top_m"):
                    raise ValueError(
                        f"FATAL: Full run preflight failed: PPMI sidecar top_m mismatch with probe! "
                        f"Probe had {probe_sidecars.get('ppmi', {}).get('top_m')}, sidecar on disk has {ppmi_meta.get('top_m')}."
                    )

            print("[Preflight] Checkpoint B operational loss gate PASSED and provenance verified. Proceeding to full evaluation run.\n")

        # Force rebuild sidecars if requested
        if args.force_rebuild_sidecars:
            print("\n[ForceRebuild] Rebuilding sidecars for all datasets before evaluation...")
            sidecar_mgr = Gate1SidecarManager(config=frozen_config)
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            enc_temp = SentenceTransformer(args.bge_model, device=device)
            for ds in datasets:
                safe_ds = ds.lower().replace("-", "_")
                pool_path = f"data/cache/canonical_pools/{safe_ds}_canonical_pool.json"
                if not os.path.exists(pool_path):
                    raise FileNotFoundError(f"Canonical pool missing for {ds}: {pool_path}")
                with open(pool_path, "r", encoding="utf-8") as f:
                    pool_terms = json.load(f)["terms"]
                idx_path = f"data/cache/terrier_indices/{safe_ds}_default/data.properties"
                idx_temp = pt.IndexFactory.of(os.path.abspath(idx_path))
                num_docs = idx_temp.getCollectionStatistics().getNumberOfDocuments()
                sidecar_mgr.build_or_load_bge_sidecar(ds, pool_terms, num_docs=num_docs, encoder=enc_temp, device=device, force_rebuild=True)
                top_m = int(frozen_config.get("sidecars", {}).get("ppmi", {}).get("top_m", 600)) if frozen_config else 600
                sidecar_mgr.build_or_load_bounded_ppmi_sidecar(ds, idx_temp, pool_terms, num_docs=num_docs, top_m=top_m, force_rebuild=True)
                sidecar_mgr.build_or_load_acronym_rescue_sidecar(ds, pool_terms, num_docs=num_docs, force_rebuild=True)
                sidecar_mgr.build_or_load_sparse_lexical_sidecar(ds, pool_terms, num_docs=num_docs, force_rebuild=True)
                del idx_temp
            del enc_temp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[ForceRebuild] Completed sidecar rebuilds successfully.\n")

        # Load Phase 1 candidate audit if available
        phase1_audit_df = None
        if args.phase1_audit and os.path.exists(args.phase1_audit):
            print(f"Loading Phase 1 candidate audit from {args.phase1_audit}...")
            phase1_audit_df = pd.read_parquet(args.phase1_audit)
            print(f"Loaded {len(phase1_audit_df):,} Phase 1 audit rows.")

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
                bge_model=args.bge_model,
                qids_manifest=args.qids_manifest,
                force_rebuild_sidecars=args.force_rebuild_sidecars,
            )
            meta_summaries.append(ds_meta)

        # Assemble master artifacts with fail-closed QID verification
        expected_qids_all = set()
        if frozen_config and "query_manifests" in frozen_config:
            q_manifest = frozen_config["query_manifests"]
            for ds in datasets:
                if args.qids_manifest and args.qids_manifest in q_manifest:
                    expected_qids_all.update(q_manifest[args.qids_manifest].get(ds, []))
                elif args.sample_size == 1 and "micro_smoke_qids" in q_manifest:
                    expected_qids_all.update(q_manifest["micro_smoke_qids"].get(ds, []))
                elif args.sample_size == 10 and "probe_40_qids" in q_manifest:
                    expected_qids_all.update(q_manifest["probe_40_qids"].get(ds, []))
                elif args.sample_size == 50 and "dev_200_qids" in q_manifest:
                    expected_qids_all.update(q_manifest["dev_200_qids"].get(ds, []))

        shards_dir = os.path.join(args.output_dir, "shards")
        master_parent = os.path.join(args.output_dir, "gate1_candidate_audit.parquet")
        master_cutoff = os.path.join(args.output_dir, "gate1_cutoff_entries.parquet")
        master_status = os.path.join(args.output_dir, "query_status.parquet")
        master_universe = os.path.join(args.output_dir, "reference_universe.parquet")
        master_diag_universe = os.path.join(args.output_dir, "diagnostic_universe.parquet")

        assemble_shards_to_master(shards_dir, shard_type="audit", output_path=master_parent, expected_qids=expected_qids_all if expected_qids_all else None)
        assemble_shards_to_master(shards_dir, shard_type="cutoff", output_path=master_cutoff)
        assemble_shards_to_master(shards_dir, shard_type="status", output_path=master_status, expected_qids=expected_qids_all if expected_qids_all else None)
        assemble_shards_to_master(shards_dir, shard_type="universe", output_path=master_universe, expected_qids=expected_qids_all if expected_qids_all else None)
        if args.measure_fidelity:
            assemble_shards_to_master(shards_dir, shard_type="diag_universe", output_path=master_diag_universe, expected_qids=expected_qids_all if expected_qids_all else None)

        print("\nValidating action coverage between candidate audit and reference universe...")
        df_audit_master = pd.read_parquet(master_parent)
        if args.measure_fidelity and os.path.exists(master_diag_universe):
            df_target_univ = pd.read_parquet(master_diag_universe)
            print("Validating against diagnostic universe (measure_fidelity=True)...")
        else:
            df_target_univ = pd.read_parquet(master_universe)
            print("Validating against operational reference universe...")
        validate_action_coverage(df_audit_master, df_target_univ, weights)
        print("Action coverage verification PASSED (exact Cartesian set equality).")

    import platform
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    driver_ver = "N/A"
    try:
        with open("/proc/driver/nvidia/version", "r") as f:
            driver_ver = f.readline().strip().split()[7]
    except Exception:
        pass

    git_info = get_git_info()
    peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash,
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "command_args": sys.argv,
        "datasets": datasets,
        "sample_size": args.sample_size,
        "seed": seed,
        "weights": weights,
        "peak_rss_gib": round(PEAK_PROCESS_TREE_RSS_BYTES / (1024 ** 3), 3),
        "peak_cuda_vram_mib": peak_vram_mib,
        "total_variants": sum(s["total_variants"] for s in meta_summaries),
        "total_cutoff_entries": sum(s["total_cutoff_entries"] for s in meta_summaries),
        "total_retrieval_time_sec": round(sum(s["retrieval_time_sec"] for s in meta_summaries), 2),
        "pool_hashes": {s["dataset"]: s["pool_sha256"] for s in meta_summaries},
        "index_hashes": {s["dataset"]: s["index_manifest_sha256"] for s in meta_summaries},
        "dataset_hashes": {s["dataset"]: s.get("sidecars_provenance", {}).get("bge", {}).get("corpus_source_hash", "") for s in meta_summaries},
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
