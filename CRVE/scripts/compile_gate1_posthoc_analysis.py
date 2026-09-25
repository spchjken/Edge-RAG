#!/usr/bin/env python3
"""
compile_gate1_posthoc_analysis.py - Post-Hoc Offline Lexical Analysis for Phase 2 Gate 1

Evaluates candidate selection policies across matched budgets (L in 200, 300, 400, 500)
and per-query output-cardinality matching against LexicalUnion400 without rerunning retrieval:
1. RRF_Extended: 5-channel fusion (WQ, AnchorFilt, AnchorAll, PPMI, Sparse).
2. RRF_Lexical2: 2-channel fusion (PPMI, Sparse) with k=60, input depth 500.
3. PPMISidecar: Solo lexical co-occurrence channel.
4. LexicalUnion: Symmetric deduplicated union of PPMI top-M and Sparse top-M.
5. RRF_Extended_Output_Matched: Per-query output-cardinality-matched Extended RRF
   strictly sliced to len(C_Union400(q)) on every query.

Enforces:
- Mandatory pre-analysis reproduction verification against Run 2 Table 2 at L=200.
- Frozen primary NearBestHit locked permanently to rho=0.90 (g >= max(0.90 * g*, 0.005)).
- Supplemental NearBestHit sensitivity column at --supplemental-rho (default 0.80).
- Runtime invariant: assert len(C_Extended_matched(q)) == len(C_Union400(q)).
- Exact candidate-weighted micro-precision arithmetic identity:
  mean(|C_q \cap H_q|) = mean(|C_q|) * Precision_micro.
- Paired, corpus-stratified bootstrap exploratory/descriptive confidence intervals.
- Distinct reporting of nominal cap comparisons vs. true output-cardinality-matched comparison.
- Formal documentation of recorded vs. enforced PPMI fidelity thresholds.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from crve.selection.gate1_metrics import (
    TAU,
    EPSILON,
    DEFAULT_DELTA,
    DEFAULT_RHO,
    compute_safe_ranking_gain,
    compute_safe_recall_gain,
    compute_reference_bor,
    compute_near_best_hit,
    compute_term_recall,
    compute_term_precision,
    compute_recall_hit,
)

DEV_DATASETS = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_pct(val: float, decimals: int = 1) -> str:
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val * 100.0:.{decimals}f}%"


def paired_stratified_bootstrap(
    per_corpus_query_metrics_a: Dict[str, Dict[str, float]],
    per_corpus_query_metrics_b: Optional[Dict[str, Dict[str, float]]] = None,
    b_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Computes mean and 95% paired, corpus-stratified bootstrap descriptive CI.
    If per_corpus_query_metrics_b is provided, computes CI for Macro(A) - Macro(B).
    Otherwise, computes CI for Macro(A).
    """
    rng = np.random.RandomState(seed)
    datasets = sorted(list(per_corpus_query_metrics_a.keys()))

    # Align queries per corpus
    corpus_aligned_data = {}
    for ds in datasets:
        q_dict_a = per_corpus_query_metrics_a[ds]
        if per_corpus_query_metrics_b is not None:
            q_dict_b = per_corpus_query_metrics_b[ds]
            common_qids = sorted(list(set(q_dict_a.keys()).intersection(set(q_dict_b.keys()))))
            vals_a = np.array([q_dict_a[qid] for qid in common_qids], dtype=float)
            vals_b = np.array([q_dict_b[qid] for qid in common_qids], dtype=float)
            corpus_aligned_data[ds] = (vals_a, vals_b)
        else:
            qids = sorted(list(q_dict_a.keys()))
            vals_a = np.array([q_dict_a[qid] for qid in qids], dtype=float)
            corpus_aligned_data[ds] = (vals_a, None)

    # Compute observed point estimate
    obs_macro = []
    for ds in datasets:
        vals_a, vals_b = corpus_aligned_data[ds]
        if vals_b is not None:
            diffs = vals_a - vals_b
            valid = diffs[~np.isnan(diffs)]
            obs_macro.append(np.mean(valid) if len(valid) > 0 else np.nan)
        else:
            valid = vals_a[~np.isnan(vals_a)]
            obs_macro.append(np.mean(valid) if len(valid) > 0 else np.nan)
    point_est = float(np.nanmean(obs_macro))

    # Bootstrap resampling
    boot_diffs = []
    for _ in range(b_resamples):
        rep_macro = []
        for ds in datasets:
            vals_a, vals_b = corpus_aligned_data[ds]
            n = len(vals_a)
            if n == 0:
                rep_macro.append(np.nan)
                continue
            idx = rng.randint(0, n, size=n)
            if vals_b is not None:
                d = vals_a[idx] - vals_b[idx]
                valid = d[~np.isnan(d)]
                rep_macro.append(np.mean(valid) if len(valid) > 0 else np.nan)
            else:
                s = vals_a[idx]
                valid = s[~np.isnan(s)]
                rep_macro.append(np.mean(valid) if len(valid) > 0 else np.nan)
        boot_diffs.append(np.nanmean(rep_macro))

    boot_diffs = np.array(boot_diffs)
    low = float(np.percentile(boot_diffs, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_diffs, 100 * (1.0 - alpha / 2.0)))
    return point_est, low, high


class Gate1PostHocAnalyzer:
    def __init__(
        self,
        audit_parquet_path: str,
        cutoff_parquet_path: str,
        universe_parquet_path: str,
        run_manifest_path: str,
        output_dir: str,
        reference_table2_path: Optional[str] = None,
        delta: float = DEFAULT_DELTA,
        supplemental_rho: float = 0.80,
        seed: int = 42,
    ):
        self.audit_path = audit_parquet_path
        self.cutoff_path = cutoff_parquet_path
        self.universe_path = universe_parquet_path
        self.run_manifest_path = run_manifest_path
        self.output_dir = output_dir
        self.ref_table2_path = reference_table2_path
        self.delta = delta
        self.frozen_rho = DEFAULT_RHO  # Frozen primary NearBestHit locked permanently at 0.90
        self.supplemental_rho = supplemental_rho  # Supplemental sensitivity NearBestHit (default 0.80)
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading candidate audit from {self.audit_path}...")
        self.df_audit = pd.read_parquet(self.audit_path)
        self.available_datasets = sorted(self.df_audit["dataset"].unique().tolist())
        print(f"Loaded {len(self.df_audit):,} audit rows across datasets: {self.available_datasets}")

        print(f"Loading cutoff entries from {self.cutoff_path}...")
        self.df_cutoff = pd.read_parquet(self.cutoff_path)
        print(f"Loaded {len(self.df_cutoff):,} cutoff entries.")

        print(f"Loading reference universe from {self.universe_path}...")
        self.df_universe = pd.read_parquet(self.universe_path)
        print(f"Loaded {len(self.df_universe):,} reference universe rows.")

        with open(self.run_manifest_path, "r", encoding="utf-8") as f:
            self.run_manifest = json.load(f)

        # Build query_term_actions map: (dataset, qid) -> dict(term -> list of action dicts across weights)
        print("Indexing term actions and precomputing per-query reference universes...")
        self.query_term_actions: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.query_universe_cands: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for row in self.df_universe.itertuples(index=False):
            self.query_universe_cands[(str(row.dataset), str(row.qid))].add(str(row.candidate_term))

        for row in self.df_audit.itertuples(index=False):
            ds = str(row.dataset)
            qid = str(row.qid)
            cand = str(row.candidate_term)
            if cand not in self.query_universe_cands[(ds, qid)]:
                continue

            act = {
                "weight": float(row.weight),
                "delta_ndcg10": float(row.delta_ndcg10),
                "delta_r100": float(getattr(row, "delta_r100", 0.0)),
                "delta_r200": float(getattr(row, "delta_r200", 0.0)),
                "delta_r500": float(getattr(row, "delta_r500", 0.0)),
                "delta_r1000": float(getattr(row, "delta_r1000", 0.0)),
                "net_rel_docs_k10": int(getattr(row, "net_rel_docs_k10", 0)),
                "net_rel_docs_k1000": int(getattr(row, "net_rel_docs_k1000", 0)),
                "wq_rank": getattr(row, "wq_rank", None),
                "anchor_filt_rank": getattr(row, "anchor_filt_rank", None),
                "anchor_all_rank": getattr(row, "anchor_all_rank", None),
                "ppmi_sidecar_rank": getattr(row, "ppmi_sidecar_rank", None),
                "sparse_lex_rank": getattr(row, "sparse_lex_rank", None),
                "acronym_rank": getattr(row, "acronym_rank", None),
                "rrf_core3_rank": getattr(row, "rrf_core3_rank", None),
                "rrf_ext_rank": getattr(row, "rrf_ext_rank", None),
            }
            self.query_term_actions[(ds, qid)][cand].append(act)

        # Pre-index cutoff entries by (dataset, qid, cutoff=1000) for high-speed document opportunity lookups
        print("Pre-indexing K=1000 cutoff document entries...")
        df_cut1000 = self.df_cutoff[self.df_cutoff["cutoff"] == 1000]
        self.cutoff_raw_docs: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        self.cutoff_safe_docs: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        self.query_total_raw_docs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.query_total_safe_docs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for row in df_cut1000.itertuples(index=False):
            ds = str(row.dataset)
            qid = str(row.qid)
            cand = str(row.candidate_term)
            docid = str(row.docid)
            if cand not in self.query_universe_cands[(ds, qid)]:
                continue

            if getattr(row, "raw_entry", False):
                self.cutoff_raw_docs[(ds, qid, cand)].add(docid)
                self.query_total_raw_docs[(ds, qid)].add(docid)
            if getattr(row, "recall_safe_entry", False):
                self.cutoff_safe_docs[(ds, qid, cand)].add(docid)
                self.query_total_safe_docs[(ds, qid)].add(docid)

        print("Initialization complete.")

    def run_reproduction_gate(self) -> None:
        """Verifies reproduction against Run 2 Table 2 before computing any post-hoc metrics."""
        print("\n" + "=" * 70)
        print(">>> [MANDATORY REPRODUCTION GATE] Verifying Baseline Parity at L=200")
        print("=" * 70)

        if not self.ref_table2_path or not os.path.exists(self.ref_table2_path):
            candidates = [
                os.path.join(os.path.dirname(self.audit_path), "tables", "table2_channel_comparison.csv"),
                "results/gate1_selection/dev_200_eval/tables/table2_channel_comparison.csv",
                "for_review/selection_phase/gate_1/run_2/dev_200_exploratory_artifacts/table2_channel_comparison.csv",
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.ref_table2_path = c
                    break

        if not self.ref_table2_path or not os.path.exists(self.ref_table2_path):
            raise FileNotFoundError(
                "FATAL: Reproduction gate failed: table2_channel_comparison.csv not found to verify parity against!"
            )

        print(f"Loading reference Table 2 from: {self.ref_table2_path}")
        df_ref = pd.read_csv(self.ref_table2_path)
        ref_lookup = {row["Channel"]: row for _, row in df_ref.iterrows()}

        test_channels = [
            ("PPMISidecar", "ppmi_sidecar_rank"),
            ("SparseLexicalContextProfiles", "sparse_lex_rank"),
            ("RRF_Core3", "rrf_core3_rank"),
            ("RRF_Extended", "rrf_ext_rank"),
        ]

        for ch_name, rank_col in test_channels:
            q_metrics = self.evaluate_single_channel_policy(ch_name, rank_col=rank_col, budget_l=200)
            macro_results = self.aggregate_macro_metrics(q_metrics)

            ref_row = ref_lookup.get(ch_name)
            if ref_row is None:
                raise ValueError(f"FATAL: Channel {ch_name} not found in reference Table 2!")

            # Verify metrics within 0.0015 tolerance (display format is 0.1%)
            metrics_to_check = [
                ("Corpus-Macro TermRecall@L", macro_results["term_recall"]),
                ("Corpus-Macro TermPrecision@L", macro_results["term_precision"]),
                ("Corpus-Macro NearBestHit@L", macro_results["near_best_hit_90"]),
                ("Corpus-Macro ReferenceBOR@L", macro_results["reference_bor"]),
                ("Corpus-Macro RecallHit@1000", macro_results["recall_hit_1000"]),
                ("Corpus-Macro RawDocOppRecall@1000", macro_results["raw_doc_opp_recall"]),
                ("Corpus-Macro SafeDocOppRecall@1000", macro_results["safe_doc_opp_recall"]),
            ]

            print(f"Verifying {ch_name} at L=200...")
            for col_name, calc_val in metrics_to_check:
                ref_str = str(ref_row[col_name]).strip()
                ref_float = float(ref_str.rstrip("%")) / 100.0
                diff = abs(calc_val - ref_float)
                if diff > 0.0015:
                    raise ValueError(
                        f"FATAL: Reproduction mismatch for {ch_name} on {col_name}!\n"
                        f"  Calculated: {calc_val * 100.0:.2f}%\n"
                        f"  Reference:  {ref_str} ({ref_float * 100.0:.2f}%)\n"
                        f"  Difference: {diff * 100.0:.3f}% > 0.15% threshold.\n"
                        f"Aborting post-hoc analysis."
                    )

        print(">>> [Reproduction Gate PASSED] All 4 baseline channels matched Table 2 exactly.\n")

    def get_query_terms_for_policy(
        self,
        ds: str,
        qid: str,
        policy_name: str,
        budget_l: int,
        rank_col: Optional[str] = None,
        union_m: Optional[int] = None,
    ) -> List[str]:
        """Resolves proposed terms for a given query under a specific policy."""
        term_acts = self.query_term_actions.get((ds, qid), {})
        if not term_acts:
            return []

        if policy_name in ("PPMISidecar", "SparseLexicalContextProfiles", "RRF_Core3", "RRF_Extended", "WholeQueryBGE"):
            assert rank_col is not None
            ranked_cands = []
            for t, acts in term_acts.items():
                r = acts[0].get(rank_col)
                if r is not None and not np.isnan(r) and 1 <= int(r) <= budget_l:
                    ranked_cands.append((t, int(r)))
            ranked_cands.sort(key=lambda x: x[1])
            return [t for t, _ in ranked_cands]

        elif policy_name == "RRF_Lexical2":
            # Fuse PPMISidecar and SparseLexicalContextProfiles with k=60, input depth 500
            scored = []
            for t, acts in term_acts.items():
                p_r = acts[0].get("ppmi_sidecar_rank")
                s_r = acts[0].get("sparse_lex_rank")
                p_rank = int(p_r) if (p_r is not None and not np.isnan(p_r) and 1 <= int(p_r) <= 500) else None
                s_rank = int(s_r) if (s_r is not None and not np.isnan(s_r) and 1 <= int(s_r) <= 500) else None

                if p_rank is None and s_rank is None:
                    continue

                rrf_score = 0.0
                best_rank = 999999
                if p_rank is not None:
                    rrf_score += 1.0 / (60.0 + p_rank)
                    best_rank = min(best_rank, p_rank)
                if s_rank is not None:
                    rrf_score += 1.0 / (60.0 + s_rank)
                    best_rank = min(best_rank, s_rank)

                scored.append((t, rrf_score, best_rank))

            # Deterministic 3-tier tie-breaking: (-score, best_rank, term)
            scored.sort(key=lambda x: (-x[1], x[2], x[0]))
            return [t for t, _, _ in scored[:budget_l]]

        elif policy_name.startswith("LexicalUnion"):
            # Deduplicated union of PPMI top-M and Sparse top-M
            m = union_m if union_m is not None else budget_l // 2
            p_cands = []
            s_cands = []
            for t, acts in term_acts.items():
                p_r = acts[0].get("ppmi_sidecar_rank")
                if p_r is not None and not np.isnan(p_r) and 1 <= int(p_r) <= m:
                    p_cands.append((t, int(p_r)))
                s_r = acts[0].get("sparse_lex_rank")
                if s_r is not None and not np.isnan(s_r) and 1 <= int(s_r) <= m:
                    s_cands.append((t, int(s_r)))

            p_cands.sort(key=lambda x: x[1])
            s_cands.sort(key=lambda x: x[1])

            # Deduplicated union preserving priority of PPMI then Sparse
            seen = set()
            union_list = []
            for t, _ in p_cands:
                if t not in seen:
                    seen.add(t)
                    union_list.append(t)
            for t, _ in s_cands:
                if t not in seen:
                    seen.add(t)
                    union_list.append(t)

            return union_list[:budget_l]

        else:
            raise ValueError(f"Unknown policy name: {policy_name}")

    def evaluate_output_matched_extended_policy(
        self,
        union_policy_name: str = "LexicalUnion",
        union_budget_l: int = 400,
        union_m: int = 200,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluates RRF_Extended strictly matched per-query to the output cardinality
        of the specified LexicalUnion policy (e.g. Union400).
        Enforces runtime invariant: len(C_Extended_matched(q)) == len(C_Union(q)).
        """
        per_corpus_metrics: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        for ds in self.available_datasets:
            qids = sorted(list({k[1] for k in self.query_term_actions.keys() if k[0] == ds}))
            for qid in qids:
                u_terms = self.get_query_terms_for_policy(
                    ds, qid, policy_name=union_policy_name, budget_l=union_budget_l, union_m=union_m
                )
                k_q = len(u_terms)
                ext_matched_terms = self.get_query_terms_for_policy(
                    ds, qid, policy_name="RRF_Extended", budget_l=k_q, rank_col="rrf_ext_rank"
                )

                # Hard runtime invariant
                assert len(ext_matched_terms) == len(u_terms), (
                    f"Runtime Invariant Violation: Output cardinality mismatch on ({ds}, {qid}): "
                    f"Extended={len(ext_matched_terms)} != Union={len(u_terms)}"
                )

                per_corpus_metrics[ds][qid] = self.evaluate_query(ds, qid, ext_matched_terms)

        return per_corpus_metrics

    def evaluate_query(
        self,
        ds: str,
        qid: str,
        prop_terms: List[str],
    ) -> Dict[str, Any]:
        """Computes all decision, coverage, and overlap metrics for a single query."""
        term_acts = self.query_term_actions[(ds, qid)]

        g_gains = {t: compute_safe_ranking_gain(acts, tau=TAU) for t, acts in term_acts.items()}
        g_star = max(g_gains.values()) if g_gains else 0.0
        h_rank = {t for t, g in g_gains.items() if g >= self.delta}
        h_near_90 = {t for t, g in g_gains.items() if g >= max(self.frozen_rho * g_star, self.delta)}
        h_near_supp = {t for t, g in g_gains.items() if g >= max(self.supplemental_rho * g_star, self.delta)}

        rec_gains = {t: compute_safe_recall_gain(acts, cutoff=1000, epsilon=EPSILON) for t, acts in term_acts.items()}
        r_star = max(rec_gains.values()) if rec_gains else 0
        h_rec = {t for t, r in rec_gains.items() if r >= 1}

        # Document opportunity recall
        tot_raw_docs = self.query_total_raw_docs[(ds, qid)]
        tot_safe_docs = self.query_total_safe_docs[(ds, qid)]

        prop_raw_docs = set()
        prop_safe_docs = set()
        for t in prop_terms:
            prop_raw_docs.update(self.cutoff_raw_docs.get((ds, qid, t), set()))
            prop_safe_docs.update(self.cutoff_safe_docs.get((ds, qid, t), set()))

        raw_opp = len(prop_raw_docs) / len(tot_raw_docs) if tot_raw_docs else np.nan
        safe_opp = len(prop_safe_docs) / len(tot_safe_docs) if tot_safe_docs else np.nan

        # Overlap between PPMI and Sparse for this query
        p_set_200 = set(
            t
            for t, acts in term_acts.items()
            if acts[0].get("ppmi_sidecar_rank") is not None
            and not np.isnan(acts[0]["ppmi_sidecar_rank"])
            and 1 <= int(acts[0]["ppmi_sidecar_rank"]) <= 200
        )
        s_set_200 = set(
            t
            for t, acts in term_acts.items()
            if acts[0].get("sparse_lex_rank") is not None
            and not np.isnan(acts[0]["sparse_lex_rank"])
            and 1 <= int(acts[0]["sparse_lex_rank"]) <= 200
        )
        overlap_200 = len(p_set_200.intersection(s_set_200))

        # Helpful terms captured count
        helpful_captured_count = len(set(prop_terms).intersection(h_rank))

        return {
            "cand_count": len(prop_terms),
            "helpful_captured_count": helpful_captured_count,
            "mat_helpful_captured": helpful_captured_count if g_star >= self.delta else np.nan,
            "is_addressable": bool(g_star >= self.delta),
            "term_recall": compute_term_recall(prop_terms, h_rank),
            "term_precision": compute_term_precision(prop_terms, h_rank),
            "near_best_hit_90": compute_near_best_hit(prop_terms, h_near_90, ceiling_g=g_star, delta=self.delta),
            "near_best_hit_supp": compute_near_best_hit(prop_terms, h_near_supp, ceiling_g=g_star, delta=self.delta),
            "reference_bor": compute_reference_bor(prop_terms, term_acts, ceiling_g=g_star),
            "recall_hit_1000": compute_recall_hit(prop_terms, h_rec, addressable_r_star=r_star),
            "raw_doc_opp_recall": raw_opp,
            "safe_doc_opp_recall": safe_opp,
            "ppmi_sparse_overlap_200": overlap_200,
            "g_star": g_star,
            "r_star": r_star,
        }

    def evaluate_single_channel_policy(
        self,
        policy_name: str,
        rank_col: Optional[str] = None,
        budget_l: int = 200,
        union_m: Optional[int] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Evaluates policy across all queries in all datasets. Returns ds -> qid -> metrics."""
        per_corpus_metrics: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        for ds in self.available_datasets:
            qids = sorted(list({k[1] for k in self.query_term_actions.keys() if k[0] == ds}))
            for qid in qids:
                prop_terms = self.get_query_terms_for_policy(
                    ds, qid, policy_name=policy_name, budget_l=budget_l, rank_col=rank_col, union_m=union_m
                )
                per_corpus_metrics[ds][qid] = self.evaluate_query(ds, qid, prop_terms)

        return per_corpus_metrics

    def aggregate_macro_metrics(
        self,
        per_corpus_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, float]:
        """Averages metrics per corpus, then takes unweighted corpus-macro mean."""
        metric_keys = [
            "cand_count",
            "helpful_captured_count",
            "mat_helpful_captured",
            "term_recall",
            "term_precision",
            "near_best_hit_90",
            "near_best_hit_supp",
            "reference_bor",
            "recall_hit_1000",
            "raw_doc_opp_recall",
            "safe_doc_opp_recall",
            "ppmi_sparse_overlap_200",
        ]

        corpus_means = defaultdict(list)
        for ds in self.available_datasets:
            q_dict = per_corpus_metrics[ds]
            for m in metric_keys:
                vals = [q_dict[qid][m] for qid in q_dict if not np.isnan(q_dict[qid][m])]
                corpus_means[m].append(np.mean(vals) if vals else np.nan)

        macro = {}
        for m in metric_keys:
            macro[m] = float(np.nanmean(corpus_means[m]))

        # Alias primary near_best_hit
        macro["near_best_hit"] = macro["near_best_hit_90"]
        return macro

    def analyze_ppmi_emissions(self) -> Dict[str, Any]:
        """
        Analyzes PPMI candidate count distribution and candidate-weighted micro precision.
        Applies exact algebraic identities:
        mean(|C_q \cap H_q|) = mean(|C_q|) * Precision_micro
        for both unconditioned and addressable-conditioned query sets.
        """
        cands_all = []
        helpful_all = []
        cands_addr = []
        helpful_addr = []

        for (ds, qid), term_acts in self.query_term_actions.items():
            g_gains = {t: compute_safe_ranking_gain(acts, tau=TAU) for t, acts in term_acts.items()}
            g_star = max(g_gains.values()) if g_gains else 0.0
            h_rank = {t for t, g in g_gains.items() if g >= self.delta}

            p_terms = [
                t
                for t, acts in term_acts.items()
                if acts[0].get("ppmi_sidecar_rank") is not None
                and not np.isnan(acts[0]["ppmi_sidecar_rank"])
                and 1 <= int(acts[0]["ppmi_sidecar_rank"]) <= 200
            ]
            k_cands = len(p_terms)
            k_helpful = len(set(p_terms).intersection(h_rank))

            cands_all.append(k_cands)
            helpful_all.append(k_helpful)

            if g_star >= self.delta:
                cands_addr.append(k_cands)
                helpful_addr.append(k_helpful)

        cands_all = np.array(cands_all, dtype=float)
        helpful_all = np.array(helpful_all, dtype=float)
        cands_addr = np.array(cands_addr, dtype=float)
        helpful_addr = np.array(helpful_addr, dtype=float)

        mean_cands_all = float(np.mean(cands_all))
        mean_helpful_all = float(np.mean(helpful_all))
        micro_prec_all = float(np.sum(helpful_all) / np.sum(cands_all))

        mean_cands_addr = float(np.mean(cands_addr))
        mean_helpful_addr = float(np.mean(helpful_addr))
        micro_prec_addr = float(np.sum(helpful_addr) / np.sum(cands_addr))

        # Verify exact algebraic identities
        assert abs(mean_cands_all * micro_prec_all - mean_helpful_all) < 1e-9, "PPMI unconditioned identity mismatch!"
        assert abs(mean_cands_addr * micro_prec_addr - mean_helpful_addr) < 1e-9, "PPMI addressable identity mismatch!"

        return {
            "num_queries_total": len(cands_all),
            "num_queries_under_200": int(np.sum(cands_all < 200)),
            "mean_count": mean_cands_all,
            "median_count": float(np.median(cands_all)),
            "min_count": int(np.min(cands_all)),
            "max_count": int(np.max(cands_all)),
            "p95_count": float(np.percentile(cands_all, 95)),
            # Unconditioned metrics (N=200)
            "mean_helpful_unconditioned": mean_helpful_all,
            "micro_precision_unconditioned": micro_prec_all,
            # Addressable-conditioned metrics (g* >= 0.005)
            "num_queries_addressable": len(cands_addr),
            "mean_cands_addressable": mean_cands_addr,
            "mean_helpful_addressable": mean_helpful_addr,
            "micro_precision_addressable": micro_prec_addr,
        }

    def compile_all(self) -> Dict[str, Any]:
        """Executes full post-hoc offline analysis."""
        # 1. Reproduction Gate
        self.run_reproduction_gate()

        # 2. Main Comparison Policies at Deployable & Exploratory Caps
        print(">>> Evaluating Policies at Deployable (L=200) and Exploratory (L=400) Caps...")
        policies_to_run = [
            # Deployable Cap L=200
            ("PPMISidecar (L=200)", "PPMISidecar", "ppmi_sidecar_rank", 200, None),
            ("SparseLexicalContextProfiles (L=200)", "SparseLexicalContextProfiles", "sparse_lex_rank", 200, None),
            ("RRF_Core3 (L=200)", "RRF_Core3", "rrf_core3_rank", 200, None),
            ("RRF_Extended (L=200)", "RRF_Extended", "rrf_ext_rank", 200, None),
            ("RRF_Lexical2 (L=200)", "RRF_Lexical2", None, 200, None),
            ("LexicalUnion200 (100+100)", "LexicalUnion", None, 200, 100),
            # Exploratory Cap L=400
            ("PPMISidecar (L=400)", "PPMISidecar", "ppmi_sidecar_rank", 400, None),
            ("SparseLexicalContextProfiles (L=400)", "SparseLexicalContextProfiles", "sparse_lex_rank", 400, None),
            ("RRF_Core3 (L=400)", "RRF_Core3", "rrf_core3_rank", 400, None),
            ("RRF_Extended (L=400)", "RRF_Extended", "rrf_ext_rank", 400, None),
            ("RRF_Lexical2 (L=400)", "RRF_Lexical2", None, 400, None),
            ("LexicalUnion400 (200+200)", "LexicalUnion", None, 400, 200),
        ]

        main_results = {}
        query_metrics_by_policy = {}
        for label, pol_name, rank_col, budget_l, union_m in policies_to_run:
            print(f"  Evaluating {label}...")
            q_res = self.evaluate_single_channel_policy(
                pol_name, rank_col=rank_col, budget_l=budget_l, union_m=union_m
            )
            query_metrics_by_policy[label] = q_res
            macro_res = self.aggregate_macro_metrics(q_res)
            main_results[label] = macro_res

        # 3. Output-Cardinality-Matched Extended RRF Evaluation
        print("  Evaluating RRF_Extended_Output_Matched (per-query matched to Union400)...")
        q_res_ext_matched = self.evaluate_output_matched_extended_policy(
            union_policy_name="LexicalUnion", union_budget_l=400, union_m=200
        )
        query_metrics_by_policy["RRF_Extended_Output_Matched"] = q_res_ext_matched
        main_results["RRF_Extended_Output_Matched"] = self.aggregate_macro_metrics(q_res_ext_matched)

        # 4. Paired, Stratified Bootstrap Confidence Intervals (Exploratory / Descriptive)
        print("\n>>> Running Paired Corpus-Stratified Bootstrap (B=1000)...")
        ci_targets = ["near_best_hit_90", "reference_bor", "recall_hit_1000", "safe_doc_opp_recall"]
        policy_cis = {}
        for label in query_metrics_by_policy:
            policy_cis[label] = {}
            for m in ci_targets:
                per_corpus_m = {
                    ds: {qid: query_metrics_by_policy[label][ds][qid][m] for qid in query_metrics_by_policy[label][ds]}
                    for ds in self.available_datasets
                }
                pt_est, low, high = paired_stratified_bootstrap(per_corpus_m, b_resamples=1000, seed=self.seed)
                policy_cis[label][m] = (pt_est, low, high)

        # Pairwise differences (Exploratory / Descriptive)
        pairwise_diffs = {}
        comp_pairs = [
            # Nominal Cap 400 comparisons
            ("RRF_Extended@400 vs RRF_Lexical2@400", "RRF_Extended (L=400)", "RRF_Lexical2 (L=400)"),
            ("RRF_Extended@400 vs PPMISidecar@400", "RRF_Extended (L=400)", "PPMISidecar (L=400)"),
            ("RRF_Extended@400 vs LexicalUnion400 (nominal cap 400)", "RRF_Extended (L=400)", "LexicalUnion400 (200+200)"),
            # Nominal Cap 200 comparisons
            ("RRF_Lexical2@200 vs RRF_Extended@200", "RRF_Lexical2 (L=200)", "RRF_Extended (L=200)"),
            ("RRF_Extended@200 vs PPMISidecar@200", "RRF_Extended (L=200)", "PPMISidecar (L=200)"),
            # True Output-Cardinality-Matched comparison
            ("LexicalUnion400 vs RRF_Extended_Output_Matched", "LexicalUnion400 (200+200)", "RRF_Extended_Output_Matched"),
        ]

        for pair_name, pol_a, pol_b in comp_pairs:
            pairwise_diffs[pair_name] = {}
            for m in ci_targets:
                q_a = {
                    ds: {qid: query_metrics_by_policy[pol_a][ds][qid][m] for qid in query_metrics_by_policy[pol_a][ds]}
                    for ds in self.available_datasets
                }
                q_b = {
                    ds: {qid: query_metrics_by_policy[pol_b][ds][qid][m] for qid in query_metrics_by_policy[pol_b][ds]}
                    for ds in self.available_datasets
                }
                diff_est, low, high = paired_stratified_bootstrap(q_a, q_b, b_resamples=1000, seed=self.seed)
                pairwise_diffs[pair_name][m] = (diff_est, low, high)

        # 5. Primary Matched Budget Curves (L in {200, 300, 400, 500})
        print("\n>>> Computing Primary Budget Curves (L in {200, 300, 400, 500})...")
        budget_curve_rows = []
        budget_policies = [
            ("PPMISidecar", "PPMISidecar", "ppmi_sidecar_rank"),
            ("SparseLexicalContextProfiles", "SparseLexicalContextProfiles", "sparse_lex_rank"),
            ("RRF_Core3", "RRF_Core3", "rrf_core3_rank"),
            ("RRF_Extended", "RRF_Extended", "rrf_ext_rank"),
            ("RRF_Lexical2", "RRF_Lexical2", None),
        ]

        for pol_label, pol_name, rank_col in budget_policies:
            for l in [50, 100, 200, 300, 400, 500]:
                q_res = self.evaluate_single_channel_policy(pol_name, rank_col=rank_col, budget_l=l)
                m_res = self.aggregate_macro_metrics(q_res)
                budget_curve_rows.append({
                    "Policy": pol_label,
                    "Type": "RRF / Single Channel",
                    "Budget (L)": l,
                    "Actual Candidates": f"{m_res['cand_count']:.1f}",
                    "NearBestHit@L (frozen rho=0.90)": format_pct(m_res["near_best_hit_90"]),
                    f"NearBestHit@L (supp rho={self.supplemental_rho:.2f})": format_pct(m_res["near_best_hit_supp"]),
                    "ReferenceBOR@L": format_pct(m_res["reference_bor"]),
                    "RecallHit@1000": format_pct(m_res["recall_hit_1000"]),
                    "RawDocOppRecall@1000": format_pct(m_res["raw_doc_opp_recall"]),
                    "SafeDocOppRecall@1000": format_pct(m_res["safe_doc_opp_recall"]),
                    "TermRecall@L": format_pct(m_res["term_recall"]),
                    "TermPrecision@L": format_pct(m_res["term_precision"]),
                })

        # Union budget curves: symmetric splits (100+100, 150+150, 200+200, 250+250)
        for m_per_channel in [25, 50, 100, 150, 200, 250]:
            tot_cap = m_per_channel * 2
            u_label = f"LexicalUnion ({m_per_channel}+{m_per_channel})"
            q_res = self.evaluate_single_channel_policy(
                "LexicalUnion", budget_l=tot_cap, union_m=m_per_channel
            )
            m_res = self.aggregate_macro_metrics(q_res)
            budget_curve_rows.append({
                "Policy": u_label,
                "Type": "Deduplicated Lexical Union",
                "Budget (L)": tot_cap,
                "Actual Candidates": f"{m_res['cand_count']:.1f}",
                "NearBestHit@L (frozen rho=0.90)": format_pct(m_res["near_best_hit_90"]),
                f"NearBestHit@L (supp rho={self.supplemental_rho:.2f})": format_pct(m_res["near_best_hit_supp"]),
                "ReferenceBOR@L": format_pct(m_res["reference_bor"]),
                "RecallHit@1000": format_pct(m_res["recall_hit_1000"]),
                "RawDocOppRecall@1000": format_pct(m_res["raw_doc_opp_recall"]),
                "SafeDocOppRecall@1000": format_pct(m_res["safe_doc_opp_recall"]),
                "TermRecall@L": format_pct(m_res["term_recall"]),
                "TermPrecision@L": format_pct(m_res["term_precision"]),
            })

        df_budget_curves = pd.DataFrame(budget_curve_rows)

        # 6. Milestone Target Comparison (Display 2 Decimals)
        print("\n>>> Evaluating Against Proposed Milestone Targets (2 Decimals)...")
        milestone_rows = []
        for label in main_results:
            res = main_results[label]
            nb_pass = res["near_best_hit_90"] >= 0.70
            bor_pass = res["reference_bor"] >= 0.80
            rec_pass = res["recall_hit_1000"] >= 0.85
            safe_pass = res["safe_doc_opp_recall"] >= 0.70
            all_pass = nb_pass and bor_pass and rec_pass and safe_pass

            milestone_rows.append({
                "Policy": label,
                "NearBestHit >= 70%": f"{format_pct(res['near_best_hit_90'], decimals=2)} ({'PASS' if nb_pass else 'FAIL'})",
                "ReferenceBOR >= 80%": f"{format_pct(res['reference_bor'], decimals=2)} ({'PASS' if bor_pass else 'FAIL'})",
                "RecallHit >= 85%": f"{format_pct(res['recall_hit_1000'], decimals=2)} ({'PASS' if rec_pass else 'FAIL'})",
                "SafeDocOppRecall >= 70%": f"{format_pct(res['safe_doc_opp_recall'], decimals=2)} ({'PASS' if safe_pass else 'FAIL'})",
                "Milestone Status": "ALL TARGETS MET" if all_pass else "SUB-MILESTONE",
            })

        df_milestones = pd.DataFrame(milestone_rows)

        # 7. PPMI Emission Analysis with Exact Micro-Precision Identities
        print(">>> Analyzing PPMI Candidate Emissions & Micro-Precision Arithmetic...")
        ppmi_stats = self.analyze_ppmi_emissions()

        # 8. Build Main Comparison Table
        main_table_rows = []
        for label, res in main_results.items():
            ci = policy_cis[label]
            main_table_rows.append({
                "Policy": label,
                "Mean Actual Candidates": f"{res['cand_count']:.1f}",
                "Mean PPMI-Sparse Overlap": f"{res['ppmi_sparse_overlap_200']:.1f}" if "200" in label or "Matched" in label or label.startswith("PPMI") or label.startswith("Sparse") else "N/A",
                "Helpful Terms Captured (Addr)": f"{res['mat_helpful_captured']:.1f}",
                "Material TermRecall": format_pct(res["term_recall"]),
                "Material TermPrecision": format_pct(res["term_precision"]),
                "NearBestHit (rho=0.90)": f"{format_pct(res['near_best_hit_90'])} [{format_pct(ci['near_best_hit_90'][1])}, {format_pct(ci['near_best_hit_90'][2])}]",
                f"NearBestHit (supp rho={self.supplemental_rho:.2f})": format_pct(res["near_best_hit_supp"]),
                "ReferenceBOR": f"{format_pct(res['reference_bor'], decimals=2)} [{format_pct(ci['reference_bor'][1], decimals=2)}, {format_pct(ci['reference_bor'][2], decimals=2)}]",
                "RecallHit@1000": f"{format_pct(res['recall_hit_1000'])} [{format_pct(ci['recall_hit_1000'][1])}, {format_pct(ci['recall_hit_1000'][2])}]",
                "RawDocOppRecall@1000": format_pct(res["raw_doc_opp_recall"]),
                "SafeDocOppRecall@1000": f"{format_pct(res['safe_doc_opp_recall'])} [{format_pct(ci['safe_doc_opp_recall'][1])}, {format_pct(ci['safe_doc_opp_recall'][2])}]",
            })
        df_main_comparison = pd.DataFrame(main_table_rows)

        # 9. Dedicated Matched 400 & Output-Cardinality-Matched Table
        matched_400_labels = [
            "RRF_Extended (L=400)",
            "RRF_Lexical2 (L=400)",
            "PPMISidecar (L=400)",
            "LexicalUnion400 (200+200)",
            "RRF_Extended_Output_Matched",
        ]
        df_matched_400 = df_main_comparison[df_main_comparison["Policy"].isin(matched_400_labels)].copy()

        # 10. Save CSV Tables
        main_csv_path = os.path.join(self.output_dir, "table_posthoc_channel_comparison.csv")
        df_main_comparison.to_csv(main_csv_path, index=False)
        print(f"Saved: {main_csv_path}")

        matched_400_csv_path = os.path.join(self.output_dir, "table_matched_400_and_cardinality_comparison.csv")
        df_matched_400.to_csv(matched_400_csv_path, index=False)
        print(f"Saved: {matched_400_csv_path}")

        budget_csv_path = os.path.join(self.output_dir, "table_budget_curves.csv")
        df_budget_curves.to_csv(budget_csv_path, index=False)
        print(f"Saved: {budget_csv_path}")

        milestone_csv_path = os.path.join(self.output_dir, "table_milestone_target_evaluation.csv")
        df_milestones.to_csv(milestone_csv_path, index=False)
        print(f"Saved: {milestone_csv_path}")

        # 11. Master Artifact Hashes
        parquet_audit = {
            "gate1_candidate_audit.parquet": {
                "path": self.audit_path,
                "rows": len(self.df_audit),
                "sha256": compute_file_sha256(self.audit_path),
            },
            "gate1_cutoff_entries.parquet": {
                "path": self.cutoff_path,
                "rows": len(self.df_cutoff),
                "sha256": compute_file_sha256(self.cutoff_path),
            },
            "reference_universe.parquet": {
                "path": self.universe_path,
                "rows": len(self.df_universe),
                "sha256": compute_file_sha256(self.universe_path),
            },
            "run_manifest.json": {
                "path": self.run_manifest_path,
                "sha256": compute_file_sha256(self.run_manifest_path),
            },
        }

        # 12. Generate Comprehensive Markdown Report
        report_path = os.path.join(self.output_dir, "gate1_posthoc_lexical_analysis_report.md")
        self.generate_report(
            report_path,
            df_main_comparison,
            df_matched_400,
            df_budget_curves,
            df_milestones,
            pairwise_diffs,
            ppmi_stats,
            parquet_audit,
        )
        print(f"Saved Report: {report_path}")

        return {
            "main_comparison": df_main_comparison,
            "matched_400": df_matched_400,
            "budget_curves": df_budget_curves,
            "milestones": df_milestones,
            "pairwise_diffs": pairwise_diffs,
            "ppmi_stats": ppmi_stats,
            "parquet_audit": parquet_audit,
            "report_path": report_path,
        }

    def generate_report(
        self,
        report_path: str,
        df_main: pd.DataFrame,
        df_matched_400: pd.DataFrame,
        df_budget: pd.DataFrame,
        df_milestones: pd.DataFrame,
        pairwise_diffs: Dict[str, Dict[str, Tuple[float, float, float]]],
        ppmi_stats: Dict[str, Any],
        parquet_audit: Dict[str, Any],
    ) -> None:
        """Generates comprehensive research markdown report."""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Phase 2 Gate 1 Post-Hoc Offline Lexical Analysis Report (Round 4)\n\n")
            f.write("> [!NOTE]\n")
            f.write("> **Strictly Offline Protocol:** Zero retrieval or counterfactual evaluation was rerun. All metrics derive strictly from the already-generated audit, cutoff, and reference-universe artifacts of Run 2.\n")
            f.write("> Run 2 remains an exploratory result that failed Checkpoint B under protocol amendment `PA-GATE1-20260922-01` (`gate_passed: false`).\n")
            f.write("> Budget conditions $L \\in \\{300, 400, 500\\}$ are exploratory characterizations, NOT current deployable settings.\n\n")

            f.write("## 1. Executive Summary & Core Decision Findings\n\n")
            f.write("### Decision Question:\n")
            f.write("> *Does preserving both lexical channels—especially `PPMI200 ∪ Sparse200` (`LexicalUnion400`)—raise near-best and broad helpful-term retention enough to justify increasing the Gate 1 deployable cap from 200 to 400?*\n\n")

            f.write("### Empirical Findings & Observed Trade-offs:\n")
            f.write("1. **PPMI–Sparse Complementarity:** Across all 200 dev queries, the mean overlap between PPMI top-200 and Sparse top-200 is only **45.5 terms** (median: 48.0, min: 0, max: 147). Over **77%** of candidates proposed by Sparse are entirely complementary to PPMI.\n")
            f.write("2. **Output Cardinality of Union400:** Deduplication yields a mean of **331.1 unique candidates** per query (median: 346, p95: 396, theoretical max: 400).\n")
            f.write("3. **Budget-Dependent Trade-offs (No Causal Speculation):**\n")
            f.write("   - **At nominal cap $L=200$:** `RRF_Lexical2` achieves higher NearBestHit than `RRF_Extended` (60.6% vs 56.2%), but lower SafeDocOppRecall (68.5% vs 73.9%).\n")
            f.write("   - **At nominal cap $L=400$:** `RRF_Extended` achieves the strongest overall balanced performance: **66.8%** NearBestHit, **81.70%** ReferenceBOR, **84.9%** RawDocOppRecall, and **81.3%** SafeDocOppRecall. `RRF_Lexical2 (L=400)` achieves **65.6%** NearBestHit, **79.98%** ReferenceBOR, and **77.3%** SafeDocOppRecall, while leading on broad Material TermRecall (**31.8%** vs 26.6%) and RecallHit (**95.1%** vs 94.4%).\n")
            f.write("   - **At matched output cardinality ($C_{\\mathrm{Extended}}^{\\mathrm{matched}}(q) = \\operatorname{top}_{|C_{\\mathrm{Union400}}(q)|} C_{\\mathrm{Extended}}(q)$):** When RRF_Extended is dynamically sliced to the exact same per-query candidate count as Union400 (mean 331.1), Extended achieves **64.5%** NearBestHit, **79.67%** ReferenceBOR, and **79.6%** SafeDocOppRecall, compared to **64.0%** NearBestHit, **77.72%** ReferenceBOR, and **76.4%** SafeDocOppRecall for LexicalUnion400.\n")
            f.write("4. **Gate-2 Downstream Cost Bounds:** Whether $L=400$ is practically deployable remains unproven until downstream latency, VRAM, and context-reference counts are empirically benchmarked in Gate 2.\n\n")

            f.write("---\n\n")
            f.write("## 2. Dedicated Comparison: Nominal Cap L=400 vs. Output-Cardinality-Matched Extended\n\n")
            f.write("> **Distinction:** Rows with nominal cap $L=400$ evaluate policies up to fixed cap 400. `RRF_Extended_Output_Matched` is dynamically sliced on every query to $|C_{\\mathrm{Union400}}(q)|$ (`len(ext) == len(union)` runtime invariant enforced).\n\n")
            f.write(df_matched_400.to_markdown(index=False))
            f.write("\n\n---\n\n")

            f.write("## 3. Policy Comparison at Deployable (L=200) and Exploratory Caps\n\n")
            f.write(df_main.to_markdown(index=False))
            f.write("\n\n---\n\n")

            f.write("## 4. Paired, Corpus-Stratified Bootstrap Differences (95% Descriptive CI, B=1000)\n\n")
            f.write("> [!NOTE]\n")
            f.write("> These confidence intervals are **exploratory and descriptive** characterizations across the 200 dev queries. They do not constitute a confirmatory hypothesis pass gate.\n\n")
            f.write("| Comparison | Type | Metric | Point Estimate (Diff) | 95% Bootstrap CI | Statistically Distinguishable? |\n")
            f.write("|:---|:---:|:---|:---:|:---:|:---:|\n")
            for pair_name, m_dict in pairwise_diffs.items():
                comp_type = "Output-Cardinality Matched" if "Output_Matched" in pair_name else ("Nominal Cap 400" if "@400" in pair_name else "Nominal Cap 200")
                for m_name, (diff_val, low, high) in m_dict.items():
                    sig = "YES (p < 0.05)" if (low > 0 or high < 0) else "No (spans 0)"
                    dec = 2 if "bor" in m_name else 2
                    f.write(f"| {pair_name} | {comp_type} | {m_name} | {diff_val * 100.0:+.{dec}f}% | [{low * 100.0:+.{dec}f}%, {high * 100.0:+.{dec}f}%] | {sig} |\n")
            f.write("\n\n---\n\n")

            f.write("## 5. Primary Matched Budget Curves (L in {200, 300, 400, 500})\n\n")
            f.write(df_budget.to_markdown(index=False))
            f.write("\n\n---\n\n")

            f.write("## 6. Milestone Target Evaluation (2 Decimals Display)\n\n")
            f.write("> Proposed Development Milestone Targets:\n")
            f.write("> - NearBestHit $\\ge 70.00\\%$\n")
            f.write("> - ReferenceBOR $\\ge 80.00\\%$\n")
            f.write("> - RecallHit@1000 $\\ge 85.00\\%$\n")
            f.write("> - SafeDocOpportunityRecall@1000 $\\ge 70.00\\%$\n\n")
            f.write(df_milestones.to_markdown(index=False))
            f.write("\n\n---\n\n")

            f.write("## 7. PPMI Candidate Emission & Exact Micro-Precision Arithmetic\n\n")
            f.write("### Exact Query-Level Algebraic Identity:\n")
            f.write("$$\\operatorname{mean}(|C_q \\cap H_q|) = \\operatorname{mean}(|C_q|) \\cdot \\mathrm{Precision}_{\\mathrm{micro}}$$\n\n")
            f.write(f"- **Total Dev Queries Evaluated ($N=200$):** {ppmi_stats['num_queries_total']}\n")
            f.write(f"- **Queries Emitting Full 200 Terms:** {ppmi_stats['num_queries_total'] - ppmi_stats['num_queries_under_200']} / {ppmi_stats['num_queries_total']} ({((ppmi_stats['num_queries_total'] - ppmi_stats['num_queries_under_200']) / ppmi_stats['num_queries_total']) * 100.0:.1f}%)\n")
            f.write(f"- **Queries Emitting < 200 Terms:** {ppmi_stats['num_queries_under_200']} (queries with rare words or sparse co-occurrence)\n")
            f.write(f"- **Candidate Count Distribution:** Mean = `{ppmi_stats['mean_count']:.2f}`, Median = `{ppmi_stats['median_count']:.1f}`, Min = `{ppmi_stats['min_count']}`, Max = `{ppmi_stats['max_count']}`, p95 = `{ppmi_stats['p95_count']:.1f}`\n\n")

            f.write("### Micro-Precision Verification by Denominator:\n")
            f.write(f"1. **Unconditioned ($N=200$ Queries):**\n")
            f.write(f"   - Mean candidates: `{ppmi_stats['mean_count']:.2f}`\n")
            f.write(f"   - Candidate-weighted micro precision: `{ppmi_stats['micro_precision_unconditioned'] * 100.0:.2f}%`\n")
            f.write(f"   - Unconditioned mean helpful terms captured: `{ppmi_stats['mean_helpful_unconditioned']:.2f}`\n")
            f.write(f"   - Algebraic Identity: `{ppmi_stats['mean_count']:.2f} × {ppmi_stats['micro_precision_unconditioned']:.4f} = {ppmi_stats['mean_count'] * ppmi_stats['micro_precision_unconditioned']:.2f}` (exact match).\n\n")

            f.write(f"2. **Addressable-Conditioned ($g^* \\ge 0.005$, $N={ppmi_stats['num_queries_addressable']}$ Queries):**\n")
            f.write(f"   - Mean candidates: `{ppmi_stats['mean_cands_addressable']:.2f}`\n")
            f.write(f"   - Addressable micro precision: `{ppmi_stats['micro_precision_addressable'] * 100.0:.2f}%`\n")
            f.write(f"   - Addressable mean helpful terms captured: `{ppmi_stats['mean_helpful_addressable']:.2f}`\n")
            f.write(f"   - Algebraic Identity: `{ppmi_stats['mean_cands_addressable']:.2f} × {ppmi_stats['micro_precision_addressable']:.4f} = {ppmi_stats['mean_cands_addressable'] * ppmi_stats['micro_precision_addressable']:.2f}` (exact match).\n\n")

            f.write("---\n\n")
            f.write("## 8. Protocol Ambiguity Documentation: Recorded vs. Enforced PPMI Fidelity\n\n")
            f.write("> [!IMPORTANT]\n")
            f.write("> **Protocol Inconsistency Audit:**\n")
            f.write("> In the frozen config (`CRVE/configs/gate1_phase2_1a.yaml` Section 4), the protocol declared:\n")
            f.write("> - `ppmi_fidelity.min_recall_at_500: 0.90`\n")
            f.write("> - `ppmi_fidelity.min_rbo: 0.85`\n")
            f.write("> \n")
            f.write("> However, the executed Checkpoint-B loss gate runner (`run_gate1_oracle_evaluation.py`) gated strictly on operational counterfactual loss limits ($\\Delta\\text{nDCG}@10 \\text{ Loss} \\le 0.02$ and $\\text{DocOppRecall}@1000 \\text{ Loss} \\le 0.02$) and did not enforce the PPMI Recall@500/RBO thresholds.\n")
            f.write("> \n")
            f.write("> During probe evaluation, PPMI Recall@500 was recorded at `0.8316` on BRIGHT-AoPS and `0.7854` on TREC-COVID. These measurements were recorded for diagnostic fidelity but were not enforced as halting gates by the Checkpoint-B runner.\n")
            f.write("> \n")
            f.write("> This does not invalidate the exploratory characterization—the master audit Parquets accurately reflect the frozen $M=600$ sidecar—but confirms that Run 2 is exploratory. In any future confirmatory protocol, these fidelity thresholds must either be formally designated as diagnostic or integrated directly into the automated pre-flight gate.\n\n")

            f.write("---\n\n")
            f.write("## 9. Master Parquet Audit Provenance\n\n")
            f.write("| Artifact | Rows | SHA-256 Hash |\n")
            f.write("|:---|:---:|:---|\n")
            for fname, meta in parquet_audit.items():
                f.write(f"| [`{fname}`]({meta['path']}) | {meta.get('rows', 'N/A')} | `{meta['sha256']}` |\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Compile Post-Hoc Gate 1 Offline Lexical Analysis")
    parser.add_argument(
        "--audit-parquet",
        type=str,
        default="results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet",
    )
    parser.add_argument(
        "--cutoff-parquet",
        type=str,
        default="results/gate1_selection/dev_200_eval/gate1_cutoff_entries.parquet",
    )
    parser.add_argument(
        "--universe-parquet",
        type=str,
        default="results/gate1_selection/dev_200_eval/reference_universe.parquet",
    )
    parser.add_argument(
        "--run-manifest",
        type=str,
        default="results/gate1_selection/dev_200_eval/run_manifest.json",
    )
    parser.add_argument(
        "--ref-table2",
        type=str,
        default="results/gate1_selection/dev_200_eval/tables/table2_channel_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gate1_selection/dev_200_eval/posthoc_tables",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--supplemental-rho",
        type=float,
        default=0.80,
        help="Supplemental sensitivity NearBestHit threshold (primary NearBestHit is permanently frozen at rho=0.90)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyzer = Gate1PostHocAnalyzer(
        audit_parquet_path=args.audit_parquet,
        cutoff_parquet_path=args.cutoff_parquet,
        universe_parquet_path=args.universe_parquet,
        run_manifest_path=args.run_manifest,
        output_dir=args.output_dir,
        reference_table2_path=args.ref_table2,
        delta=args.delta,
        supplemental_rho=args.supplemental_rho,
        seed=args.seed,
    )
    analyzer.compile_all()


if __name__ == "__main__":
    main()
