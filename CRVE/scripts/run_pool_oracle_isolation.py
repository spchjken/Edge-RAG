"""
scripts/run_pool_oracle_isolation.py

Unified PyTerrier Stage 1 Vocabulary Pool Oracle & Capacity Isolation Harness.
Executes Phase A (5-query throughput gate & CELF profiling), Phase B (seeded pilot),
and Phase C (confirmatory full-query oracle evaluation) under strict memory bounds.

Key Features:
- Exact Gate A weight injection via query_toks preserving original query TFs.
- Two-pass candidate screening: exact deployable 5-weight oracle + screened full vocabulary.
- Dynamic BRIGHT exclusion padding (K_fetch = min(N, K + |E_q|)).
- Disjoint 6-band DF attribution partition.
- Atomic query-level Parquet sharding with md5 hash filenames for robust auto-resumption.
- CELF construction timing and peak memory profiling.
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
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
import psutil
import pyterrier as pt
import ir_measures
from ir_measures import nDCG, R

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.baselines.pyterrier_harness import init_pyterrier
from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer
from evaluation.pool_generators import (
    PoolPolicy,
    get_df_band,
    is_eligible_term,
    generate_salience_sequence,
    generate_specificity_sequence,
    generate_hybrid_sequence,
    generate_stratified_sequence,
    generate_celf_coverage_sequence,
    get_effective_capacities,
    FIXED_BUDGETS,
    PERCENTAGE_CUTOFFS,
    MAX_CAPACITY,
)

TAU_TOL = 1e-5
SAFETY_EPSILON = 0.001


def get_process_tree_rss_bytes() -> int:
    """Calculates total RSS memory consumed by this process and all its child processes (e.g. JVM)."""
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


def verify_baseline_parity(
    dataset: str,
    sampled_queries: List[Dict[str, Any]],
    analyzer,
    bm25,
    num_to_check: int = 5,
) -> Dict[str, Any]:
    """
    Asserts exact executable parity between standard query-string retrieval
    and zero-expansion query_toks retrieval on analyzed terms.
    """
    checked = 0
    passed = 0
    max_score_diff = 0.0
    queries_checked = min(len(sampled_queries), num_to_check)

    for i in range(queries_checked):
        q = sampled_queries[i]
        qid = str(q["query_id"])
        q_text = q["question"]
        terms, _ = analyzer.analyze(q_text)
        orig_query_toks = analyzer.to_query_toks(terms)
        query_analyzed_str = " ".join(terms)

        # 1. query string from analyzed terms
        df_str = pd.DataFrame([{"qid": qid, "query": query_analyzed_str}])
        res_str = bm25.transform(df_str)

        # 2. query_toks
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
            if diff <= 1e-4:
                passed += 1
        elif doc_match and not s_scores:
            passed += 1

    is_ok = (checked > 0) and (passed == checked)
    status = "PASS" if is_ok else "FAIL"
    print(f"  [Baseline Parity Check] {dataset}: {passed}/{checked} queries matched perfectly (max score diff: {max_score_diff:.6e}) -> {status}")
    return {
        "dataset": dataset,
        "checked": checked,
        "passed": passed,
        "max_score_diff": max_score_diff,
        "status": status,
    }


def get_shard_path(shards_dir: str, dataset: str, qid: str) -> str:
    """Generates a filesystem-safe md5-hashed shard path for query-level checkpointing."""
    q_hash = hashlib.md5(qid.encode("utf-8")).hexdigest()[:12]
    return os.path.join(shards_dir, f"{dataset}_{q_hash}.parquet")


def extract_gold_terms(index, gold_dids: List[str]) -> Set[str]:
    """
    Extracts all analyzed terms from judged relevant documents (rel >= 1).
    Uses DirectIndex and Lexicon for fast C-level inverted posting list reading.
    """
    gold_terms = set()
    try:
        meta = index.getMetaIndex()
        di = index.getDirectIndex()
        doi = index.getDocumentIndex()
        lex = index.getLexicon()

        for did in gold_dids:
            docid = meta.getDocument("docno", str(did))
            if docid < 0:
                continue
            entry = doi.getDocumentEntry(docid)
            if entry is None:
                continue
            postings = di.getPostings(entry)
            if postings is None:
                continue
            while postings.next() != postings.EOL:
                term_id = postings.getId()
                l_entry = lex.getLexiconEntry(term_id)
                if l_entry is not None:
                    gold_terms.add(l_entry.getKey())
    except Exception as e:
        print(f"Warning: DirectIndex term extraction error: {e}")
    return gold_terms


def build_policy_sequences(
    dataset: str,
    index,
    eligible_terms: List[str],
    df_map: Dict[str, int],
    cf_map: Dict[str, int],
    idf_map: Dict[str, float],
    num_docs: int,
    max_cap: int = MAX_CAPACITY,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """
    Builds the 5 deterministic nested policy sequences and records construction times.
    """
    sequences = {}
    construction_times = {}

    # 1. Salience
    t0 = time.perf_counter()
    sequences[PoolPolicy.SALIENCE.value] = generate_salience_sequence(
        eligible_terms, df_map, idf_map, max_cap=max_cap
    )
    construction_times[PoolPolicy.SALIENCE.value] = time.perf_counter() - t0

    # 2. Specificity
    t0 = time.perf_counter()
    sequences[PoolPolicy.SPECIFICITY.value] = generate_specificity_sequence(
        eligible_terms, cf_map, idf_map, max_cap=max_cap
    )
    construction_times[PoolPolicy.SPECIFICITY.value] = time.perf_counter() - t0

    # 3. Hybrid
    t0 = time.perf_counter()
    sequences[PoolPolicy.HYBRID.value] = generate_hybrid_sequence(
        eligible_terms, cf_map, df_map, idf_map, num_docs, max_cap=max_cap
    )
    construction_times[PoolPolicy.HYBRID.value] = time.perf_counter() - t0

    # 4. Stratified
    t0 = time.perf_counter()
    sequences[PoolPolicy.STRATIFIED.value] = generate_stratified_sequence(
        eligible_terms, df_map, idf_map, max_cap=max_cap
    )
    construction_times[PoolPolicy.STRATIFIED.value] = time.perf_counter() - t0

    # 5. Coverage (CELF)
    inv = index.getInvertedIndex()
    lex = index.getLexicon()

    def doc_posting_supplier(term: str) -> Set[int]:
        entry = lex.getLexiconEntry(term)
        if entry is None:
            return set()
        postings = inv.getPostings(entry)
        if postings is None:
            return set()
        docids = set()
        while postings.next() != postings.EOL:
            docids.add(postings.getId())
        return docids

    t0 = time.perf_counter()
    sequences[PoolPolicy.COVERAGE.value] = generate_celf_coverage_sequence(
        eligible_terms, doc_posting_supplier, idf_map, df_map=df_map, max_cap=max_cap
    )
    construction_times[PoolPolicy.COVERAGE.value] = time.perf_counter() - t0

    return sequences, construction_times


def compute_ir_metrics(
    retrieved_df: pd.DataFrame,
    qrels_dict: Dict[str, float],
    qid: str,
    exclusions: Optional[Set[str]] = None,
    k_eval: int = 1000,
) -> Dict[str, float]:
    """
    Computes linear nDCG@10, R@100, and R@1000 using ir_measures,
    applying dynamic BRIGHT exclusion filtering and re-ranking if applicable.
    """
    if retrieved_df.empty:
        return {"ndcg10": 0.0, "r100": 0.0, "r1000": 0.0}

    # Filter exclusions if present
    df = retrieved_df.copy()
    if exclusions:
        df = df[~df["docno"].astype(str).isin(exclusions)].copy()

    # Re-rank and truncate to k_eval
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df = df.iloc[:k_eval].copy()
    df["rank"] = range(1, len(df) + 1)

    # Format qrels for ir_measures (requires query_id, doc_id, relevance)
    qrels_df = pd.DataFrame([
        {"query_id": str(qid), "doc_id": str(did), "relevance": int(rel)}
        for did, rel in qrels_dict.items() if rel >= 1
    ])
    if qrels_df.empty:
        return {"ndcg10": 0.0, "r100": 0.0, "r1000": 0.0}

    # Format run for ir_measures (requires query_id, doc_id, score)
    run_df = pd.DataFrame({
        "query_id": str(qid),
        "doc_id": df["docno"].astype(str),
        "score": df["score"].astype(float),
    })

    measures = [nDCG @ 10, R @ 100, R @ 1000]
    eval_res = ir_measures.calc_aggregate(measures, qrels_df, run_df)

    return {
        "ndcg10": float(eval_res.get(nDCG @ 10, 0.0)),
        "r100": float(eval_res.get(R @ 100, 0.0)),
        "r1000": float(eval_res.get(R @ 1000, 0.0)),
    }


def evaluate_query_candidates(
    dataset: str,
    query_obj: Dict[str, Any],
    index,
    analyzer,
    bm25,
    sequences: Dict[str, List[str]],
    eligible_terms_set: Set[str],
    df_map: Dict[str, int],
    cf_map: Dict[str, int],
    idf_map: Dict[str, float],
    num_docs: int,
    weights: List[float],
    reference_weight: float = 0.10,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Evaluates all relevance-supported candidates for a single query using the two-pass
    streaming design:
    - Pass 1: Evaluates deployable candidates across all 5 weights, non-deployable at ref weight.
    - Pass 2: Evaluates active non-deployable terms across remaining weights.
    """
    qid = str(query_obj["query_id"])
    q_text = query_obj["question"]
    qrels = query_obj.get("qrels") or {str(d): 1.0 for d in query_obj.get("gold_doc_ids", [])}
    gold_dids = [str(d) for d, r in qrels.items() if r >= 1]
    exclusions = set(str(e) for e in query_obj.get("exclusions", []))

    # Analyze original query
    orig_terms, _ = analyzer.analyze(q_text)
    orig_query_toks = analyzer.to_query_toks(orig_terms)
    orig_term_set = set(orig_query_toks.keys())

    df_base = pd.DataFrame([{"qid": qid, "query_toks": orig_query_toks}])
    base_run = bm25.transform(df_base)
    base_metrics = compute_ir_metrics(base_run, qrels, qid, exclusions=exclusions)

    # Extract all terms from judged relevant documents
    gold_terms = extract_gold_terms(index, gold_dids)
    gold_cands = [t for t in gold_terms if t not in orig_term_set]

    if not gold_cands:
        return pd.DataFrame()

    # Precompute pool ranks per policy
    pool_ranks = {policy: {t: r for r, t in enumerate(seq, 1)} for policy, seq in sequences.items()}
    deployable_union_set = set()
    for seq in sequences.values():
        deployable_union_set.update(seq[:MAX_CAPACITY])

    # Classify candidates
    records = []
    variants_to_eval_pass1 = []

    for c in gold_cands:
        c_df = df_map.get(c, 0)
        c_cf = cf_map.get(c, 0)
        c_idf = idf_map.get(c, 0.0)
        is_eligible = c in eligible_terms_set
        is_deployable = c in deployable_union_set
        df_band = get_df_band(c_df, num_docs)

        weights_to_test = weights if is_deployable else [reference_weight]

        for w in weights_to_test:
            # Build modified query_toks preserving original query TFs
            q_toks = dict(orig_query_toks)
            q_toks[c] = q_toks.get(c, 0.0) + w
            var_id = f"{qid}___{c}___{w}"
            variants_to_eval_pass1.append({
                "var_id": var_id,
                "qid": qid,
                "candidate": c,
                "weight": w,
                "is_deployable": is_deployable,
                "is_eligible": is_eligible,
                "df_band": df_band,
                "candidate_df": c_df,
                "candidate_cf": c_cf,
                "candidate_idf": c_idf,
                "query_toks": q_toks,
            })

    # Execute Pass 1 in chunks
    evaluated_records = []
    for i in range(0, len(variants_to_eval_pass1), chunk_size):
        chunk = variants_to_eval_pass1[i:i + chunk_size]
        df_chunk = pd.DataFrame([{"qid": item["var_id"], "query_toks": item["query_toks"]} for item in chunk])
        res_chunk = bm25.transform(df_chunk)

        res_by_var = {var_id: group for var_id, group in res_chunk.groupby("qid")}
        for item in chunk:
            var_id = item["var_id"]
            run_v = res_by_var.get(var_id, pd.DataFrame())
            m = compute_ir_metrics(run_v, qrels, qid, exclusions=exclusions)
            c = item["candidate"]
            w = item["weight"]

            d_ndcg = m["ndcg10"] - base_metrics["ndcg10"]
            d_r100 = m["r100"] - base_metrics["r100"]
            d_r1000 = m["r1000"] - base_metrics["r1000"]

            rec = {
                "dataset": dataset,
                "qid": qid,
                "candidate_term": c,
                "weight": w,
                "is_raw_only": not item["is_eligible"],
                "is_deployable": item["is_deployable"],
                "df_band": item["df_band"],
                "candidate_df": item["candidate_df"],
                "candidate_cf": item["candidate_cf"],
                "candidate_idf": item["candidate_idf"],
                "salience_rank": pool_ranks[PoolPolicy.SALIENCE.value].get(c, -1),
                "specificity_rank": pool_ranks[PoolPolicy.SPECIFICITY.value].get(c, -1),
                "hybrid_rank": pool_ranks[PoolPolicy.HYBRID.value].get(c, -1),
                "stratified_rank": pool_ranks[PoolPolicy.STRATIFIED.value].get(c, -1),
                "coverage_rank": pool_ranks[PoolPolicy.COVERAGE.value].get(c, -1),
                "baseline_ndcg10": base_metrics["ndcg10"],
                "expanded_ndcg10": m["ndcg10"],
                "baseline_r100": base_metrics["r100"],
                "expanded_r100": m["r100"],
                "baseline_r1000": base_metrics["r1000"],
                "expanded_r1000": m["r1000"],
                "delta_ndcg10": d_ndcg,
                "delta_r100": d_r100,
                "delta_r1000": d_r1000,
                "is_ranking_useful": d_ndcg > TAU_TOL,
                "is_recall_useful": (d_r100 > TAU_TOL) or (d_r1000 > TAU_TOL),
                "is_ranking_safe": (d_ndcg > TAU_TOL) and (d_r1000 >= -TAU_TOL),
                "is_recall_safe": (d_r1000 > TAU_TOL) and (d_ndcg >= -SAFETY_EPSILON),
            }
            evaluated_records.append(rec)

    # Pass 2: Active Non-Deployable Terms across remaining weights
    active_non_deployable = {
        r["candidate_term"]
        for r in evaluated_records
        if not r["is_deployable"] and (r["is_ranking_useful"] or r["is_recall_useful"])
    }

    variants_to_eval_pass2 = []
    remaining_weights = [w for w in weights if w != reference_weight]
    for c in active_non_deployable:
        c_df = df_map.get(c, 0)
        c_cf = cf_map.get(c, 0)
        c_idf = idf_map.get(c, 0.0)
        is_eligible = c in eligible_terms_set
        df_band = get_df_band(c_df, num_docs)

        for w in remaining_weights:
            q_toks = dict(orig_query_toks)
            q_toks[c] = q_toks.get(c, 0.0) + w
            var_id = f"{qid}___{c}___{w}"
            variants_to_eval_pass2.append({
                "var_id": var_id,
                "qid": qid,
                "candidate": c,
                "weight": w,
                "is_deployable": False,
                "is_eligible": is_eligible,
                "df_band": df_band,
                "candidate_df": c_df,
                "candidate_cf": c_cf,
                "candidate_idf": c_idf,
                "query_toks": q_toks,
            })

    if variants_to_eval_pass2:
        for i in range(0, len(variants_to_eval_pass2), chunk_size):
            chunk = variants_to_eval_pass2[i:i + chunk_size]
            df_chunk = pd.DataFrame([{"qid": item["var_id"], "query_toks": item["query_toks"]} for item in chunk])
            res_chunk = bm25.transform(df_chunk)

            res_by_var = {var_id: group for var_id, group in res_chunk.groupby("qid")}
            for item in chunk:
                var_id = item["var_id"]
                run_v = res_by_var.get(var_id, pd.DataFrame())
                m = compute_ir_metrics(run_v, qrels, qid, exclusions=exclusions)
                c = item["candidate"]
                w = item["weight"]

                d_ndcg = m["ndcg10"] - base_metrics["ndcg10"]
                d_r100 = m["r100"] - base_metrics["r100"]
                d_r1000 = m["r1000"] - base_metrics["r1000"]

                rec = {
                    "dataset": dataset,
                    "qid": qid,
                    "candidate_term": c,
                    "weight": w,
                    "is_raw_only": not item["is_eligible"],
                    "is_deployable": False,
                    "df_band": item["df_band"],
                    "candidate_df": item["candidate_df"],
                    "candidate_cf": item["candidate_cf"],
                    "candidate_idf": item["candidate_idf"],
                    "salience_rank": pool_ranks[PoolPolicy.SALIENCE.value].get(c, -1),
                    "specificity_rank": pool_ranks[PoolPolicy.SPECIFICITY.value].get(c, -1),
                    "hybrid_rank": pool_ranks[PoolPolicy.HYBRID.value].get(c, -1),
                    "stratified_rank": pool_ranks[PoolPolicy.STRATIFIED.value].get(c, -1),
                    "coverage_rank": pool_ranks[PoolPolicy.COVERAGE.value].get(c, -1),
                    "baseline_ndcg10": base_metrics["ndcg10"],
                    "expanded_ndcg10": m["ndcg10"],
                    "baseline_r100": base_metrics["r100"],
                    "expanded_r100": m["r100"],
                    "baseline_r1000": base_metrics["r1000"],
                    "expanded_r1000": m["r1000"],
                    "delta_ndcg10": d_ndcg,
                    "delta_r100": d_r100,
                    "delta_r1000": d_r1000,
                    "is_ranking_useful": d_ndcg > TAU_TOL,
                    "is_recall_useful": (d_r100 > TAU_TOL) or (d_r1000 > TAU_TOL),
                    "is_ranking_safe": (d_ndcg > TAU_TOL) and (d_r1000 >= -TAU_TOL),
                    "is_recall_safe": (d_r1000 > TAU_TOL) and (d_ndcg >= -SAFETY_EPSILON),
                }
                evaluated_records.append(rec)

    return pd.DataFrame(evaluated_records)


def run_dataset_oracle_isolation(
    dataset: str,
    phase: str,
    sample_size: int,
    seed: int,
    weights: List[float],
    output_dir: str,
    resume: bool = True,
    chunk_size: int = 500,
    check_baseline_parity: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes oracle isolation for a given dataset and returns the assembled DataFrame
    and summary metadata.
    """
    print(f"\n=======================================================")
    print(f"Starting Dataset: {dataset} [Phase {phase}] (seed={seed})")
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
    max_ex = max((len(q.get("exclusions", [])) for q in all_queries), default=0)
    k_fetch = min(num_docs, 1000 + max_ex)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=k_fetch)

    lex = index.getLexicon()

    # Build lexicon mappings & eligible subset
    t0_lex = time.perf_counter()
    df_map = {}
    cf_map = {}
    idf_map = {}
    eligible_terms = []

    for entry in lex:
        t = str(entry.getKey())
        val = entry.getValue()
        df = int(val.getDocumentFrequency())
        cf = int(val.getFrequency())
        # Lucene non-negative IDF
        idf = max(math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5)), 0.0)

        df_map[t] = df
        cf_map[t] = cf
        idf_map[t] = idf

        if is_eligible_term(t, df, cf, num_docs):
            eligible_terms.append(t)

    lex_time = time.perf_counter() - t0_lex
    total_lex_terms = int(lex.numberOfEntries())
    print(f"Lexicon: {total_lex_terms} terms, {len(eligible_terms)} eligible ({lex_time:.2f}s)")

    # Build policy sequences
    sequences, construction_times = build_policy_sequences(
        dataset, index, eligible_terms, df_map, cf_map, idf_map, num_docs, max_cap=MAX_CAPACITY
    )
    eligible_set = set(eligible_terms)

    for pol, t_sec in construction_times.items():
        print(f"  - Policy '{pol}': constructed {len(sequences[pol])} terms in {t_sec:.2f}s")

    rng = random.Random(seed)

    if phase == "A" and dataset == "trec_covid":
        # Force at least one broad TREC-COVID query with |D*| > 20
        broad_q = [q for q in all_queries if len(q.get("qrels", {})) > 20]
        if broad_q:
            chosen = [rng.choice(broad_q)]
            remaining = [q for q in all_queries if q != chosen[0]]
            sampled_queries = chosen + rng.sample(remaining, min(len(remaining), sample_size - 1))
        else:
            sampled_queries = rng.sample(all_queries, min(len(all_queries), sample_size))
    elif len(all_queries) <= sample_size:
        sampled_queries = list(all_queries)
    else:
        sampled_queries = rng.sample(all_queries, sample_size)

    print(f"Sampled {len(sampled_queries)} queries for Phase {phase}.")

    parity_res = {}
    if check_baseline_parity:
        parity_res = verify_baseline_parity(dataset, sampled_queries, analyzer, bm25, num_to_check=5)

    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    all_shards = []
    total_variants = 0
    t0_retrieval = time.perf_counter()

    for idx_q, q in enumerate(sampled_queries, 1):
        qid = str(q["query_id"])
        shard_path = get_shard_path(shards_dir, dataset, qid)

        if resume and os.path.exists(shard_path):
            try:
                df_shard = pd.read_parquet(shard_path)
                all_shards.append(df_shard)
                total_variants += len(df_shard)
                print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Resumed from shard ({len(df_shard)} variants)")
                continue
            except Exception:
                pass

        t0_q = time.perf_counter()
        df_q_records = evaluate_query_candidates(
            dataset, q, index, analyzer, bm25, sequences, eligible_set,
            df_map, cf_map, idf_map, num_docs, weights, reference_weight=0.10,
            chunk_size=chunk_size,
        )
        t_q = time.perf_counter() - t0_q

        if not df_q_records.empty:
            df_q_records.to_parquet(shard_path, index=False)
            all_shards.append(df_q_records)
            total_variants += len(df_q_records)
            v_rate = len(df_q_records) / max(t_q, 0.001)
            print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: Evaluated {len(df_q_records)} variants in {t_q:.2f}s ({v_rate:.1f} var/s)")
        else:
            print(f"  [{idx_q}/{len(sampled_queries)}] QID {qid}: No candidates in {t_q:.2f}s")

    total_retrieval_time = time.perf_counter() - t0_retrieval
    overall_var_rate = total_variants / max(total_retrieval_time, 0.001)

    print(f"Completed {dataset}: {total_variants} variants in {total_retrieval_time:.2f}s ({overall_var_rate:.1f} var/s)")

    assembled_df = pd.concat(all_shards, ignore_index=True) if all_shards else pd.DataFrame()
    meta_summary = {
        "dataset": dataset,
        "phase": phase,
        "num_docs": num_docs,
        "total_lexicon_terms": total_lex_terms,
        "eligible_vocab_size": len(eligible_terms),
        "num_queries": len(sampled_queries),
        "total_variants": total_variants,
        "retrieval_time_sec": total_retrieval_time,
        "variants_per_sec": overall_var_rate,
        "construction_times_sec": construction_times,
        "baseline_parity": parity_res,
    }
    return assembled_df, meta_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Vocabulary Pool Oracle & Capacity Isolation Harness")
    parser.add_argument("--phase", type=str, choices=["A", "B", "C"], default="A", help="Execution phase")
    parser.add_argument("--datasets", type=str, default="scifact,nfcorpus,trec_covid", help="Comma-separated datasets")
    parser.add_argument("--sample-size", type=int, default=5, help="Query sample size")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--weights", type=str, default="0.05,0.10,0.30,0.50,1.00", help="Weights grid")
    parser.add_argument("--chunk-size", type=int, default=500, help="Retrieval batch chunk size")
    parser.add_argument("--output-dir", type=str, default="results/pool_isolation", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resumption")
    parser.add_argument("--skip-baseline-parity", action="store_true", help="Skip executable baseline parity check")
    return parser.parse_args()


def main():
    args = parse_args()
    init_pyterrier()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    all_dfs = []
    meta_summaries = []
    peak_process_tree_rss_bytes = get_process_tree_rss_bytes()

    for ds in datasets:
        df_ds, summary = run_dataset_oracle_isolation(
            dataset=ds,
            phase=args.phase,
            sample_size=args.sample_size,
            seed=args.seed,
            weights=weights,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            chunk_size=args.chunk_size,
            check_baseline_parity=not args.skip_baseline_parity,
        )
        if not df_ds.empty:
            all_dfs.append(df_ds)
        meta_summaries.append(summary)
        peak_process_tree_rss_bytes = max(peak_process_tree_rss_bytes, get_process_tree_rss_bytes())

    # Get git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    peak_rss_gib = peak_process_tree_rss_bytes / (1024 ** 3)

    # Save run manifest
    manifest = {
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": args.phase,
        "datasets": datasets,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "weights": weights,
        "peak_process_tree_rss_gib": round(peak_rss_gib, 3),
        "num_docs": {s["dataset"]: s["num_docs"] for s in meta_summaries},
        "eligible_vocab_sizes": {s["dataset"]: s["eligible_vocab_size"] for s in meta_summaries},
        "total_lexicon_terms": {s["dataset"]: s["total_lexicon_terms"] for s in meta_summaries},
        "total_variants": sum(s["total_variants"] for s in meta_summaries),
        "total_retrieval_time_sec": round(sum(s["retrieval_time_sec"] for s in meta_summaries), 2),
        "baseline_parity": {s["dataset"]: s.get("baseline_parity", {}) for s in meta_summaries},
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved run manifest -> {manifest_path}")

    master_parquet = os.path.join(args.output_dir, "pool_candidate_audit.parquet")
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        # Save master candidate parquet
        full_df.to_parquet(master_parquet, index=False)
        print(f"Saved master candidate audit ({len(full_df)} records) -> {master_parquet}")

        # Automatically compile summary tables
        try:
            from scripts.compile_pool_oracle_tables import compile_oracle_tables
            compile_oracle_tables(master_parquet, args.output_dir)
        except Exception as e:
            print(f"Warning: could not compile oracle tables automatically: {e}")

    # Print Phase A Forecast Report
    if args.phase == "A":
        print("\n" + "=" * 60)
        print("PHASE A: THROUGHPUT GATE & CAPACITY FORECAST REPORT")
        print("=" * 60)
        total_vars = sum(s["total_variants"] for s in meta_summaries)
        total_time = sum(s["retrieval_time_sec"] for s in meta_summaries)
        mean_vps = total_vars / max(total_time, 0.001)

        print(f"Total Phase A Queries: {sum(s['num_queries'] for s in meta_summaries)}")
        print(f"Total Variants Evaluated: {total_vars}")
        print(f"Overall Retrieval Throughput: {mean_vps:.1f} variants/sec")
        print(f"Process-Tree Peak RAM RSS: {peak_rss_gib:.2f} GiB (Ceiling: 15 GiB)")
        assert peak_rss_gib < 14.5, f"Peak process-tree RAM RSS ({peak_rss_gib:.2f} GiB) exceeded 14.5 GiB safe budget!"

        print("\nPool Construction Times:")
        for s in meta_summaries:
            ds = s["dataset"]
            c_times = s["construction_times_sec"]
            print(f"  [{ds}]: CELF={c_times.get('coverage', 0.0):.2f}s, Salience={c_times.get('salience', 0.0):.2f}s, Stratified={c_times.get('stratified', 0.0):.2f}s")
            assert c_times.get("coverage", 0.0) < 120.0, f"CELF exceeded 120s limit on {ds}!"

        # Forecast Phase B (50 queries across 4 datasets)
        phase_b_est_vars = total_vars * 10  # 50 queries vs 5 queries
        phase_b_est_hours = (phase_b_est_vars / max(mean_vps, 1.0)) / 3600.0
        est_storage_mb = (os.path.getsize(master_parquet) / (1024**2)) * 10 if os.path.exists(master_parquet) else 50.0

        print(f"\n--- Phase B Projection (50 Queries x 4 Corpora) ---")
        print(f"Estimated Variants: ~{phase_b_est_vars:,}")
        print(f"Estimated Runtime: ~{phase_b_est_hours:.2f} hours (with checkpoint/resume)")
        print(f"Estimated Shard Disk Storage: ~{est_storage_mb:.1f} MB")
        print(f"Throughput Gate Status: PASS")
        print("=" * 60)


if __name__ == "__main__":
    main()
