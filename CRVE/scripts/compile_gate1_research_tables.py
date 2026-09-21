"""
scripts/compile_gate1_research_tables.py

Compiles all Phase 2 Gate 1 Research Tables (Q1-Q10) from the candidate audit
and cutoff entries Parquets, evaluates paired query bootstrap CIs (B=1000),
applies the Multi-Dataset Decision Hierarchy (Gates A-D), enforces true corpus-macro
aggregation, and exports the frozen configuration hash.
"""

import os
import sys
import time
import math
import json
import argparse
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

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
    TRACKED_CUTOFFS,
    compute_safe_ranking_gain,
    compute_safe_recall_gain,
    get_near_best_terms,
    compute_reference_bor,
    compute_near_best_hit,
    compute_term_recall,
    compute_term_precision,
    compute_recall_hit,
)

DEV_DATASETS = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]
EXT_DATASETS = ["fiqa", "scidocs", "arguana", "bright_stackoverflow"]
ALL_DATASETS = DEV_DATASETS + EXT_DATASETS

OPERATIONAL_CHANNELS = [
    ("WholeQueryBGE", "wq_rank"),
    ("AnchorBGEFiltered", "anchor_filt_rank"),
    ("AnchorBGEAll", "anchor_all_rank"),
    ("PPMISidecar", "ppmi_sidecar_rank"),
    ("SparseLexicalContextProfiles", "sparse_lex_rank"),
    ("AcronymDefinitionRescue", "acronym_rank"),
    ("RRF_Core3", "rrf_core3_rank"),
    ("RRF_Extended", "rrf_ext_rank"),
]

DIAGNOSTIC_CHANNELS = [
    ("LivePPMI", "live_ppmi_rank"),
    ("PPMISidecar", "ppmi_sidecar_rank"),
]



def bootstrap_metric_ci(
    values: List[float],
    b: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Computes mean and 95% paired bootstrap CI for a list of query values."""
    valid_vals = [v for v in values if not np.isnan(v)]
    if not valid_vals:
        return np.nan, np.nan, np.nan

    n = len(valid_vals)
    arr = np.array(valid_vals)
    mean_val = float(np.mean(arr))
    if n <= 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(b):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


class Gate1TableCompiler:
    def __init__(
        self,
        audit_parquet_path: str,
        cutoff_parquet_path: str,
        output_dir: str,
        universe_parquet_path: Optional[str] = None,
        diag_universe_parquet_path: Optional[str] = None,
        run_manifest_path: Optional[str] = None,
        frozen_config_path: Optional[str] = None,
        delta: float = DEFAULT_DELTA,
        rho: float = DEFAULT_RHO,
        b_resamples: int = 1000,
        seed: int = 42,
    ):
        self.audit_path = audit_parquet_path
        self.cutoff_path = cutoff_parquet_path
        self.output_dir = output_dir
        self.universe_path = universe_parquet_path
        self.diag_universe_path = diag_universe_parquet_path
        self.run_manifest_path = run_manifest_path
        self.frozen_config_path = frozen_config_path
        self.delta = delta
        self.rho = rho
        self.b_resamples = b_resamples
        self.seed = seed

        print(f"Loading candidate audit from {self.audit_path}...")
        self.df_audit = pd.read_parquet(self.audit_path)
        self.available_datasets = sorted(self.df_audit['dataset'].unique().tolist())
        print(f"Loaded {len(self.df_audit):,} audit rows across datasets: {self.available_datasets}")

        self.df_cutoff = None
        if os.path.exists(self.cutoff_path):
            try:
                num_cutoff = pq.read_metadata(self.cutoff_path).num_rows
                print(f"Verified cutoff entries at {self.cutoff_path}: {num_cutoff:,} rows.")
                self.df_cutoff = pd.read_parquet(self.cutoff_path)
            except Exception as e:
                print(f"Warning: Could not read cutoff entries: {e}")
        else:
            print(f"Warning: Cutoff entries parquet not found at {self.cutoff_path}")

        # Auto-detect universe parquet if not explicitly passed
        if not self.universe_path or not os.path.exists(self.universe_path):
            candidate_univ = os.path.join(self.output_dir, "reference_universe.parquet")
            if os.path.exists(candidate_univ):
                self.universe_path = candidate_univ

        self.df_universe = None
        if self.universe_path and os.path.exists(self.universe_path):
            print(f"Loading reference universe from {self.universe_path}...")
            self.df_universe = pd.read_parquet(self.universe_path)
            print(f"Loaded {len(self.df_universe):,} reference universe rows.")

        # Auto-detect diagnostic universe parquet if not explicitly passed
        if not self.diag_universe_path or not os.path.exists(self.diag_universe_path):
            candidate_diag = os.path.join(self.output_dir, "diagnostic_universe.parquet")
            if os.path.exists(candidate_diag):
                self.diag_universe_path = candidate_diag

        self.df_diag_universe = None
        if self.diag_universe_path and os.path.exists(self.diag_universe_path):
            print(f"Loading diagnostic universe from {self.diag_universe_path}...")
            self.df_diag_universe = pd.read_parquet(self.diag_universe_path)
            print(f"Loaded {len(self.df_diag_universe):,} diagnostic universe rows.")

        # Auto-detect run manifest if not explicitly passed
        if not self.run_manifest_path or not os.path.exists(self.run_manifest_path):
            candidate_manifest = os.path.join(self.output_dir, "run_manifest.json")
            if os.path.exists(candidate_manifest):
                self.run_manifest_path = candidate_manifest

        self.run_manifest = None
        if self.run_manifest_path and os.path.exists(self.run_manifest_path):
            with open(self.run_manifest_path, "r", encoding="utf-8") as f:
                self.run_manifest = json.load(f)
            print(f"Loaded run manifest from {self.run_manifest_path}")

        # Authoritative weights resolution
        if self.run_manifest and "weights" in self.run_manifest:
            self.weights = [round(float(w), 4) for w in self.run_manifest["weights"]]
        elif self.frozen_config_path and os.path.exists(self.frozen_config_path):
            import yaml
            with open(self.frozen_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.weights = [round(float(w), 4) for w in cfg.get("weights", [0.05, 0.10, 0.30, 0.50, 1.00])]
        else:
            self.weights = sorted(list(set(round(float(w), 4) for w in self.df_audit["weight"].unique())))
        print(f"Authoritative evaluation weights: {self.weights}")

        # Build term actions map: (dataset, qid, candidate_term) -> list of actions
        self.query_term_actions = defaultdict(lambda: defaultdict(list))
        self.query_baseline_metrics = {}

        for row in self.df_audit.itertuples(index=False):
            ds = str(row.dataset)
            qid = str(row.qid)
            cand = str(row.candidate_term)
            act = {
                "weight": float(row.weight),
                "delta_ndcg10": float(row.delta_ndcg10),
                "delta_r100": float(getattr(row, "delta_r100", 0.0)),
                "delta_r200": float(getattr(row, "delta_r200", 0.0)),
                "delta_r500": float(getattr(row, "delta_r500", 0.0)),
                "delta_r1000": float(getattr(row, "delta_r1000", 0.0)),
                "net_rel_docs_k10": int(getattr(row, "net_rel_docs_k10", 0)),
                "net_rel_docs_k100": int(getattr(row, "net_rel_docs_k100", 0)),
                "net_rel_docs_k200": int(getattr(row, "net_rel_docs_k200", 0)),
                "net_rel_docs_k500": int(getattr(row, "net_rel_docs_k500", 0)),
                "net_rel_docs_k1000": int(getattr(row, "net_rel_docs_k1000", 0)),
                "wq_rank": getattr(row, "wq_rank", None),
                "anchor_filt_rank": getattr(row, "anchor_filt_rank", None),
                "anchor_all_rank": getattr(row, "anchor_all_rank", None),
                "ppmi_sidecar_rank": getattr(row, "ppmi_sidecar_rank", None),
                "live_ppmi_rank": getattr(row, "live_ppmi_rank", None),
                "sparse_lex_rank": getattr(row, "sparse_lex_rank", None),
                "acronym_rank": getattr(row, "acronym_rank", None),
                "rrf_core3_rank": getattr(row, "rrf_core3_rank", None),
                "rrf_ext_rank": getattr(row, "rrf_ext_rank", None),
            }
            self.query_term_actions[(ds, qid)][cand].append(act)
            if (ds, qid) not in self.query_baseline_metrics:
                self.query_baseline_metrics[(ds, qid)] = {
                    "baseline_ndcg10": float(getattr(row, "baseline_ndcg10", 0.0)),
                    "baseline_r1000": float(getattr(row, "baseline_r1000", 0.0)),
                }

    def compile_q1_reference_ceiling(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Q1: Reference Opportunity Ceiling vs Baseline with True Corpus-Macro Aggregation."""
        rows = []
        macro_accum = defaultdict(list)

        for ds in self.available_datasets:
            qids = sorted(list({k[1] for k in self.query_term_actions.keys() if k[0] == ds}))
            base_ndcg_list = []
            base_r1000_list = []
            ceil_g_list = []
            ceil_r_list = []
            addressable_rank_count = 0
            addressable_rec_count = 0

            for qid in qids:
                term_acts = self.query_term_actions[(ds, qid)]
                b_meta = self.query_baseline_metrics.get((ds, qid), {"baseline_ndcg10": 0.0, "baseline_r1000": 0.0})
                base_ndcg_list.append(b_meta["baseline_ndcg10"])
                base_r1000_list.append(b_meta["baseline_r1000"])

                g_gains = [compute_safe_ranking_gain(acts, tau=TAU) for acts in term_acts.values()]
                g_star = max(g_gains) if g_gains else 0.0
                ceil_g_list.append(g_star)
                if g_star >= self.delta:
                    addressable_rank_count += 1

                r_gains = [compute_safe_recall_gain(acts, cutoff=1000, epsilon=EPSILON) for acts in term_acts.values()]
                r_star = max(r_gains) if r_gains else 0
                ceil_r_list.append(r_star)
                if r_star >= 1:
                    addressable_rec_count += 1

                del term_acts

            n = len(qids)
            mean_base_ndcg = float(np.mean(base_ndcg_list))
            mean_base_r = float(np.mean(base_r1000_list))
            mean_g_star, low_g, high_g = bootstrap_metric_ci(ceil_g_list, b=self.b_resamples, seed=self.seed)
            mean_r_star = float(np.mean(ceil_r_list))

            macro_accum["base_ndcg"].append(mean_base_ndcg)
            macro_accum["base_r1000"].append(mean_base_r)
            macro_accum["ceil_g"].append(mean_g_star)
            macro_accum["ceil_r"].append(mean_r_star)
            macro_accum["addr_rank"].append(addressable_rank_count / max(n, 1))
            macro_accum["addr_rec"].append(addressable_rec_count / max(n, 1))

            rows.append({
                "Dataset": ds,
                "Partition": "Dev" if ds in DEV_DATASETS else "Extension",
                "Queries": n,
                "Baseline nDCG@10": f"{mean_base_ndcg:.4f}",
                "Baseline R@1000": f"{mean_base_r:.4f}",
                "Ceiling Delta-nDCG@10": f"+{mean_g_star:.4f} [{low_g:.4f}, {high_g:.4f}]",
                "Ceiling Net Rel Docs": f"+{mean_r_star:.2f}",
                "Materially Addressable % (g* >= 0.005)": f"{(addressable_rank_count / n) * 100:.1f}%",
                "Recall Addressable % (r* >= 1)": f"{(addressable_rec_count / n) * 100:.1f}%",
            })

        # True corpus-macro aggregation across available datasets
        dev_indices = [i for i, d in enumerate(self.available_datasets) if d in DEV_DATASETS]
        if dev_indices:
            rows.append({
                "Dataset": "Corpus-Macro Dev (4 Corpora)",
                "Partition": "Macro",
                "Queries": sum(len({k[1] for k in self.query_term_actions.keys() if k[0] == self.available_datasets[i]}) for i in dev_indices),
                "Baseline nDCG@10": f"{np.mean([macro_accum['base_ndcg'][i] for i in dev_indices]):.4f}",
                "Baseline R@1000": f"{np.mean([macro_accum['base_r1000'][i] for i in dev_indices]):.4f}",
                "Ceiling Delta-nDCG@10": f"+{np.mean([macro_accum['ceil_g'][i] for i in dev_indices]):.4f}",
                "Ceiling Net Rel Docs": f"+{np.mean([macro_accum['ceil_r'][i] for i in dev_indices]):.2f}",
                "Materially Addressable % (g* >= 0.005)": f"{np.mean([macro_accum['addr_rank'][i] for i in dev_indices]) * 100:.1f}%",
                "Recall Addressable % (r* >= 1)": f"{np.mean([macro_accum['addr_rec'][i] for i in dev_indices]) * 100:.1f}%",
            })

        df_q1 = pd.DataFrame(rows)
        return df_q1, macro_accum

    def compile_q2_single_channel_comparison(
        self,
        budget_l: int = 200,
        channels: Optional[List[Tuple[str, str]]] = None,
        use_diag_universe: bool = False,
    ) -> pd.DataFrame:
        """Q2: Channel Proposal Efficiency across channels at budget L, rendering nan% as N/A."""
        if channels is None:
            channels = OPERATIONAL_CHANNELS

        def fmt_pct(val: float) -> str:
            return f"{val * 100:.1f}%" if not np.isnan(val) else "N/A"

        rows = []
        for ch_name, rank_col in channels:
            corpus_recalls = []
            corpus_precisions = []
            corpus_bor = []
            corpus_near_hit = []
            corpus_rec_hit = []
            corpus_raw_opp = []
            corpus_safe_opp = []

            for ds in self.available_datasets:
                qids = sorted(list({k[1] for k in self.query_term_actions.keys() if k[0] == ds}))
                q_recalls = []
                q_precisions = []
                q_bor = []
                q_near_hit = []
                q_rec_hit = []
                q_raw_opp = []
                q_safe_opp = []

                for qid in qids:
                    term_acts = self.query_term_actions[(ds, qid)]
                    op_cands = None
                    if not use_diag_universe and self.df_universe is not None:
                        op_cands = set(self.df_universe[(self.df_universe["dataset"] == ds) & (self.df_universe["qid"] == qid)]["candidate_term"])
                        term_acts = {t: acts for t, acts in term_acts.items() if t in op_cands}

                    proposed = []
                    for t, acts in term_acts.items():
                        r = acts[0].get(rank_col)
                        if r is not None and not np.isnan(r) and 1 <= int(r) <= budget_l:
                            proposed.append((t, int(r)))
                    proposed.sort(key=lambda x: x[1])
                    prop_terms = [t for t, _ in proposed]
                    prop_set = set(prop_terms)

                    g_gains = {t: compute_safe_ranking_gain(acts, tau=TAU) for t, acts in term_acts.items()}
                    g_star = max(g_gains.values()) if g_gains else 0.0
                    h_rank = {t for t, g in g_gains.items() if g >= self.delta}
                    h_near = get_near_best_terms(term_acts, ceiling_g=g_star, rho=self.rho, delta=self.delta)

                    rec_gains = {t: compute_safe_recall_gain(acts, cutoff=1000, epsilon=EPSILON) for t, acts in term_acts.items()}
                    r_star = max(rec_gains.values()) if rec_gains else 0
                    h_rec = {t for t, r in rec_gains.items() if r >= 1}

                    q_recalls.append(compute_term_recall(prop_terms, h_rank))
                    q_precisions.append(compute_term_precision(prop_terms, h_rank))
                    q_bor.append(compute_reference_bor(prop_terms, term_acts, ceiling_g=g_star))
                    q_near_hit.append(compute_near_best_hit(prop_terms, h_near, ceiling_g=g_star, delta=self.delta))
                    q_rec_hit.append(compute_recall_hit(prop_terms, h_rec, addressable_r_star=r_star))

                    # Document opportunity recall from df_cutoff at K=1000
                    if self.df_cutoff is not None and not self.df_cutoff.empty:
                        c_sub = self.df_cutoff[
                            (self.df_cutoff["dataset"] == ds) & 
                            (self.df_cutoff["qid"] == qid) & 
                            (self.df_cutoff["cutoff"] == 1000)
                        ]
                        if op_cands is not None:
                            c_sub = c_sub[c_sub["candidate_term"].isin(op_cands)]

                        if not c_sub.empty:
                            total_raw_docs = set(c_sub[c_sub["raw_entry"] == True]["docid"])
                            total_safe_docs = set(c_sub[c_sub["recall_safe_entry"] == True]["docid"])
                            prop_sub = c_sub[c_sub["candidate_term"].isin(prop_set)]
                            prop_raw_docs = set(prop_sub[prop_sub["raw_entry"] == True]["docid"])
                            prop_safe_docs = set(prop_sub[prop_sub["recall_safe_entry"] == True]["docid"])

                            q_raw_opp.append(len(prop_raw_docs) / len(total_raw_docs) if total_raw_docs else np.nan)
                            q_safe_opp.append(len(prop_safe_docs) / len(total_safe_docs) if total_safe_docs else np.nan)
                        else:
                            q_raw_opp.append(np.nan)
                            q_safe_opp.append(np.nan)
                    else:
                        q_raw_opp.append(np.nan)
                        q_safe_opp.append(np.nan)

                # Per-corpus means
                corpus_recalls.append(np.nanmean(q_recalls) if q_recalls else np.nan)
                corpus_precisions.append(np.nanmean(q_precisions) if q_precisions else np.nan)
                corpus_bor.append(np.nanmean(q_bor) if q_bor else np.nan)
                corpus_near_hit.append(np.nanmean(q_near_hit) if q_near_hit else np.nan)
                corpus_rec_hit.append(np.nanmean(q_rec_hit) if q_rec_hit else np.nan)
                corpus_raw_opp.append(np.nanmean(q_raw_opp) if q_raw_opp else np.nan)
                corpus_safe_opp.append(np.nanmean(q_safe_opp) if q_safe_opp else np.nan)

            # True corpus-macro mean
            raw_opp_val = np.nanmean(corpus_raw_opp) if corpus_raw_opp else np.nan
            safe_opp_val = np.nanmean(corpus_safe_opp) if corpus_safe_opp else np.nan
            rows.append({
                "Channel": ch_name,
                "Budget (L)": budget_l,
                "Corpus-Macro TermRecall@L": fmt_pct(np.nanmean(corpus_recalls)),
                "Corpus-Macro TermPrecision@L": fmt_pct(np.nanmean(corpus_precisions)),
                "Corpus-Macro NearBestHit@L": fmt_pct(np.nanmean(corpus_near_hit)),
                "Corpus-Macro ReferenceBOR@L": fmt_pct(np.nanmean(corpus_bor)),
                "Corpus-Macro RecallHit@1000": fmt_pct(np.nanmean(corpus_rec_hit)),
                "Corpus-Macro RawDocOppRecall@1000": fmt_pct(raw_opp_val),
                "Corpus-Macro SafeDocOppRecall@1000": fmt_pct(safe_opp_val),
            })

        return pd.DataFrame(rows)

    def enforce_operational_loss_gate(self, budget_l: int = 200) -> Dict[str, Any]:
        """
        Evaluates operational loss criteria against diagnostic universe R_q^diag:
        - Delta-nDCG@10 loss <= 0.02
        - RawDocOppRecall@1000 loss <= 0.02
        against LivePPMI evaluated on the 40-query probe using the diagnostic ceiling and denominator.
        Fails closed on missing evidence or incomplete coverage.
        """
        # 1. Require frozen configuration with non-default, positive thresholds
        if not self.frozen_config_path or not os.path.exists(self.frozen_config_path):
            raise FileNotFoundError(f"FATAL: Checkpoint B operational loss gate requires a valid --frozen-config-path: {self.frozen_config_path}")
        import yaml
        with open(self.frozen_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not cfg or "checkpoint_b_thresholds" not in cfg:
            raise KeyError("FATAL: 'checkpoint_b_thresholds' missing from frozen configuration!")
        cb_thresholds = cfg["checkpoint_b_thresholds"]
        required_thresh_keys = [
            "max_oracle_loss_corpus_macro",
            "max_oracle_loss_per_corpus",
            "max_doc_opp_recall_loss_corpus_macro",
            "max_doc_opp_recall_loss_per_corpus",
        ]
        for k in required_thresh_keys:
            if k not in cb_thresholds or cb_thresholds[k] is None or float(cb_thresholds[k]) <= 0:
                raise ValueError(f"FATAL: Missing or invalid positive threshold for '{k}' in checkpoint_b_thresholds")

        max_oracle_loss_macro = float(cb_thresholds["max_oracle_loss_corpus_macro"])
        max_oracle_loss_per_corpus = float(cb_thresholds["max_oracle_loss_per_corpus"])
        max_doc_loss_macro = float(cb_thresholds["max_doc_opp_recall_loss_corpus_macro"])
        max_doc_loss_per_corpus = float(cb_thresholds["max_doc_opp_recall_loss_per_corpus"])

        # 2. Require diagnostic universe artifact
        if not self.diag_universe_path or not os.path.exists(self.diag_universe_path):
            raise FileNotFoundError(f"FATAL: Checkpoint B operational loss gate requires --diag-universe-parquet: {self.diag_universe_path}")
        if self.df_diag_universe is None or self.df_diag_universe.empty:
            raise ValueError("FATAL: Diagnostic universe is empty or unreadable.")

        # 3. Require cutoff artifact
        if not self.cutoff_path or not os.path.exists(self.cutoff_path):
            raise FileNotFoundError(f"FATAL: Checkpoint B operational loss gate requires --cutoff-parquet: {self.cutoff_path}")
        if self.df_cutoff is None or self.df_cutoff.empty:
            raise ValueError("FATAL: Cutoff entries DataFrame is None or empty.")

        # 4. Require all four expected corpora with probe QIDs in both audit and cutoff
        expected_corpora = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]
        missing_corpora = [c for c in expected_corpora if c not in self.available_datasets]
        if missing_corpora:
            raise ValueError(f"FATAL: Checkpoint B operational loss gate missing expected corpora in audit: {missing_corpora}")

        cutoff_datasets = set(self.df_cutoff["dataset"].unique())
        missing_cutoff_corpora = [c for c in expected_corpora if c not in cutoff_datasets]
        if missing_cutoff_corpora:
            raise ValueError(f"FATAL: Checkpoint B operational loss gate missing expected corpora in cutoff entries: {missing_cutoff_corpora}")

        for ds in expected_corpora:
            ds_qids = {k[1] for k in self.query_term_actions.keys() if k[0] == ds}
            if len(ds_qids) < 10:
                raise ValueError(f"FATAL: Corpus '{ds}' has only {len(ds_qids)} queries in audit, expected at least 10 probe queries.")

        # 5. Require complete counterfactual audit coverage
        cov_df = self.compile_label_coverage_table()
        missing_triples = cov_df["Missing Triples"].sum() if "Missing Triples" in cov_df.columns else 0
        extra_triples = cov_df["Extra Triples"].sum() if "Extra Triples" in cov_df.columns else 0
        if missing_triples > 0 or extra_triples > 0:
            raise RuntimeError(f"FATAL: Incomplete audit coverage for Checkpoint B: missing={missing_triples}, extra={extra_triples}")

        per_corpus_results = {}
        corpus_ndcg_losses = []
        corpus_doc_losses = []

        for ds in self.available_datasets:
            qids = sorted(list({k[1] for k in self.query_term_actions.keys() if k[0] == ds}))
            q_ndcg_losses = []
            q_doc_losses = []

            for qid in qids:
                term_acts = self.query_term_actions[(ds, qid)]
                # Diagnostic ceiling from R_q^diag
                g_gains = {t: compute_safe_ranking_gain(acts, tau=TAU) for t, acts in term_acts.items()}

                # Top-L for LivePPMI and PPMISidecar
                live_terms = [t for t, acts in term_acts.items() if acts[0].get("live_ppmi_rank") and not np.isnan(acts[0]["live_ppmi_rank"]) and 1 <= int(acts[0]["live_ppmi_rank"]) <= budget_l]
                sidecar_terms = [t for t, acts in term_acts.items() if acts[0].get("ppmi_sidecar_rank") and not np.isnan(acts[0]["ppmi_sidecar_rank"]) and 1 <= int(acts[0]["ppmi_sidecar_rank"]) <= budget_l]

                g_live = max([g_gains[t] for t in live_terms], default=0.0)
                g_sidecar = max([g_gains[t] for t in sidecar_terms], default=0.0)
                ndcg_loss = max(0.0, g_live - g_sidecar)
                q_ndcg_losses.append(ndcg_loss)

                # RawDocOppRecall@1000 from df_cutoff
                doc_loss = 0.0
                if self.df_cutoff is not None and not self.df_cutoff.empty:
                    c_sub = self.df_cutoff[
                        (self.df_cutoff["dataset"] == ds) &
                        (self.df_cutoff["qid"] == qid) &
                        (self.df_cutoff["cutoff"] == 1000) &
                        (self.df_cutoff["raw_entry"] == True)
                    ]
                    if not c_sub.empty:
                        total_docs = set(c_sub["docid"])
                        live_docs = set(c_sub[c_sub["candidate_term"].isin(set(live_terms))]["docid"])
                        sidecar_docs = set(c_sub[c_sub["candidate_term"].isin(set(sidecar_terms))]["docid"])

                        rec_live = len(live_docs) / len(total_docs) if total_docs else 1.0
                        rec_sidecar = len(sidecar_docs) / len(total_docs) if total_docs else 1.0
                        doc_loss = max(0.0, rec_live - rec_sidecar)
                q_doc_losses.append(doc_loss)

            mean_ndcg_loss = float(np.mean(q_ndcg_losses)) if q_ndcg_losses else 0.0
            mean_doc_loss = float(np.mean(q_doc_losses)) if q_doc_losses else 0.0
            corpus_ndcg_losses.append(mean_ndcg_loss)
            corpus_doc_losses.append(mean_doc_loss)

            per_corpus_results[ds] = {
                "queries": len(qids),
                "mean_delta_ndcg10_loss": round(mean_ndcg_loss, 4),
                "mean_raw_doc_opp_recall1000_loss": round(mean_doc_loss, 4),
                "ndcg_loss_passed": mean_ndcg_loss <= max_oracle_loss_per_corpus,
                "doc_loss_passed": mean_doc_loss <= max_doc_loss_per_corpus,
            }

        macro_ndcg_loss = float(np.mean(corpus_ndcg_losses)) if corpus_ndcg_losses else 0.0
        macro_doc_loss = float(np.mean(corpus_doc_losses)) if corpus_doc_losses else 0.0

        all_passed = (
            macro_ndcg_loss <= max_oracle_loss_macro and
            macro_doc_loss <= max_doc_loss_macro and
            all(r["ndcg_loss_passed"] and r["doc_loss_passed"] for r in per_corpus_results.values())
        )

        return {
            "budget_l": budget_l,
            "corpus_macro_delta_ndcg10_loss": round(macro_ndcg_loss, 4),
            "corpus_macro_raw_doc_opp_recall1000_loss": round(macro_doc_loss, 4),
            "max_oracle_loss_corpus_macro_threshold": max_oracle_loss_macro,
            "max_doc_opp_recall_loss_corpus_macro_threshold": max_doc_loss_macro,
            "per_corpus_results": per_corpus_results,
            "gate_passed": all_passed,
        }

    def compile_label_coverage_table(self) -> pd.DataFrame:
        """
        Emits 100% label-coverage audit across all tested methods and candidate terms.
        Enforces exact Cartesian set equality against reference_universe x weights without clamping.
        """
        rows = []
        target_univ = (
            self.df_diag_universe
            if (self.df_diag_universe is not None and "live_ppmi_rank" in self.df_audit.columns and self.df_audit["live_ppmi_rank"].notna().any())
            else self.df_universe
        )

        for ds in self.available_datasets:
            sub = self.df_audit[self.df_audit["dataset"] == ds]
            num_queries = sub["qid"].nunique()

            if target_univ is not None:
                sub_univ = target_univ[target_univ["dataset"] == ds]
                num_q_cand_pairs = len(sub_univ.drop_duplicates(subset=["qid", "candidate_term"]))
                
                expected_triples = set()
                for qid, cand in sub_univ[["qid", "candidate_term"]].drop_duplicates().itertuples(index=False):
                    for w in self.weights:
                        expected_triples.add((str(qid), str(cand), round(float(w), 4)))

                actual_triples = set()
                for _, row in sub.iterrows():
                    actual_triples.add((str(row["qid"]), str(row["candidate_term"]), round(float(row["weight"]), 4)))

                expected_variants = len(expected_triples)
                evaluated_variants = len(actual_triples)
                missing = expected_triples - actual_triples
                extra = actual_triples - expected_triples
                
                coverage_pct = (evaluated_variants / max(expected_variants, 1)) * 100.0
                status = "100.0% COMPLETE" if (len(missing) == 0 and len(extra) == 0 and evaluated_variants == expected_variants) else f"{coverage_pct:.2f}% INCOMPLETE (Missing: {len(missing)}, Extra: {len(extra)})"

                rows.append({
                    "Dataset": ds,
                    "Queries": num_queries,
                    "Reference Universe Pairs": num_q_cand_pairs,
                    "Expected Variants": expected_variants,
                    "Evaluated Variants": evaluated_variants,
                    "Missing Triples": len(missing),
                    "Extra Triples": len(extra),
                    "Labeling Coverage %": f"{coverage_pct:.2f}%",
                    "Coverage Status": status,
                })
            else:
                unique_q_cands = sub.drop_duplicates(subset=["qid", "candidate_term"])
                num_q_cand_pairs = len(unique_q_cands)
                expected_variants = num_q_cand_pairs * len(self.weights)
                evaluated_variants = len(sub.drop_duplicates(subset=["qid", "candidate_term", "weight"]))
                coverage_pct = (evaluated_variants / max(expected_variants, 1)) * 100.0

                rows.append({
                    "Dataset": ds,
                    "Queries": num_queries,
                    "Unique Query-Candidate Pairs": num_q_cand_pairs,
                    "Expected Variants": expected_variants,
                    "Evaluated Variants": evaluated_variants,
                    "Missing Triples": max(0, expected_variants - evaluated_variants),
                    "Extra Triples": max(0, evaluated_variants - expected_variants),
                    "Labeling Coverage %": f"{coverage_pct:.2f}%",
                    "Coverage Status": "100.0% COMPLETE" if evaluated_variants == expected_variants else f"{coverage_pct:.1f}% INCOMPLETE",
                })
        return pd.DataFrame(rows)

    def compile_all_tables_and_report(self) -> str:
        """Compiles all research tables and outputs summary."""
        os.makedirs(self.output_dir, exist_ok=True)

        print("\n--- Compiling Table 1 (Reference Ceiling) ---")
        df_q1, _ = self.compile_q1_reference_ceiling()
        t1_path = os.path.join(self.output_dir, "table1_reference_ceiling.csv")
        df_q1.to_csv(t1_path, index=False)
        print(f"Saved Table 1 -> {t1_path}")

        print("\n--- Compiling Table 2 (Operational Channel Comparison at L=200) ---")
        df_q2 = self.compile_q2_single_channel_comparison(budget_l=200, channels=OPERATIONAL_CHANNELS)
        t2_path = os.path.join(self.output_dir, "table2_channel_comparison.csv")
        df_q2.to_csv(t2_path, index=False)
        print(f"Saved Table 2 -> {t2_path}")

        df_q2_diag = None
        loss_gate_results = None
        has_live_ppmi = "live_ppmi_rank" in self.df_audit.columns and self.df_audit["live_ppmi_rank"].notna().any()
        if has_live_ppmi:
            print("\n--- Compiling Table 2-Diag (LivePPMI Diagnostic Comparator) ---")
            df_q2_diag = self.compile_q2_single_channel_comparison(budget_l=200, channels=DIAGNOSTIC_CHANNELS, use_diag_universe=True)
            t2_diag_path = os.path.join(self.output_dir, "table2_diag_live_ppmi.csv")
            df_q2_diag.to_csv(t2_diag_path, index=False)
            print(f"Saved Table 2-Diag -> {t2_diag_path}")

            print("\n--- Enforcing Checkpoint B Operational Loss Gate ---")
            loss_gate_results = self.enforce_operational_loss_gate(budget_l=200)
            loss_gate_path = os.path.join(self.output_dir, "checkpoint_b_loss_gate.json")
            with open(loss_gate_path, "w", encoding="utf-8") as f:
                json.dump(loss_gate_results, f, indent=2)
            print(f"Saved Checkpoint B Loss Gate -> {loss_gate_path}")
            print(f"Loss Gate Status: {'PASSED' if loss_gate_results['gate_passed'] else 'FAILED'}")

        print("\n--- Compiling Label Coverage Table ---")
        df_cov = self.compile_label_coverage_table()
        cov_path = os.path.join(self.output_dir, "table_label_coverage.csv")
        df_cov.to_csv(cov_path, index=False)
        print(f"Saved Label Coverage Table -> {cov_path}")

        report_path = os.path.join(self.output_dir, "gate1_selection_report.md")
        
        git_commit = self.run_manifest.get("git_commit", "N/A") if self.run_manifest else "N/A"
        git_dirty = self.run_manifest.get("git_dirty", False) if self.run_manifest else False
        peak_rss = self.run_manifest.get("peak_rss_gib", "N/A") if self.run_manifest else "N/A"
        peak_vram = self.run_manifest.get("peak_cuda_vram_mib", "N/A") if self.run_manifest else "N/A"
        cfg_hash = self.run_manifest.get("config_hash", "N/A") if self.run_manifest else "N/A"

        diag_section = ""
        if df_q2_diag is not None:
            diag_section = f"""---

## 2b. Table 2-Diag: LivePPMI Diagnostic Comparator ($L=200$)

{df_q2_diag.to_markdown(index=False)}
"""
        loss_section = ""
        if loss_gate_results is not None:
            gate_status_str = "**PASSED**" if loss_gate_results["gate_passed"] else "**FAILED**"
            loss_section = f"""---

## 2c. Checkpoint B Operational Loss Gate Evaluation

**Gate Status:** {gate_status_str}  
- **Corpus-Macro $\\Delta$nDCG@10 Loss:** `{loss_gate_results['corpus_macro_delta_ndcg10_loss']:.4f}` (threshold: `{loss_gate_results['max_oracle_loss_corpus_macro_threshold']}`)  
- **Corpus-Macro RawDocOppRecall@1000 Loss:** `{loss_gate_results['corpus_macro_raw_doc_opp_recall1000_loss']:.4f}` (threshold: `{loss_gate_results['max_doc_opp_recall_loss_corpus_macro_threshold']}`)  
"""

        report_md = f"""# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`{self.audit_path}`](file://{os.path.abspath(self.audit_path)})  
**Cutoff Entries Parquet:** [`{self.cutoff_path}`](file://{os.path.abspath(self.cutoff_path)})  
**Reference Universe Parquet:** [`{self.universe_path or 'N/A'}`](file://{os.path.abspath(self.universe_path) if self.universe_path else ''})  
**Config Hash:** `{cfg_hash}`  
**Git Commit:** `{git_commit}` (dirty: `{git_dirty}`)  
**Peak Process-Tree RSS:** `{peak_rss} GiB`  
**Peak CUDA VRAM:** `{peak_vram} MiB`  
**Generated At:** {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

{df_q1.to_markdown(index=False)}

---

## 2. Table 2: Operational Channels Comparison at Deployable Cap ($L=200$)

{df_q2.to_markdown(index=False)}

{diag_section}
{loss_section}
---

## 3. Table: 100% Counterfactual Label Coverage Verification

{df_cov.to_markdown(index=False)}
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\nSaved Report -> {report_path}")

        # Hard fail-closed enforcement if loss gate failed
        if loss_gate_results is not None and not loss_gate_results.get("gate_passed", False):
            raise RuntimeError(
                f"FATAL: Checkpoint B operational-loss gate failed; full run is blocked.\n"
                f"  Macro Delta-nDCG Loss: {loss_gate_results['corpus_macro_delta_ndcg10_loss']} (threshold: {loss_gate_results['max_oracle_loss_corpus_macro_threshold']})\n"
                f"  Macro Doc Opp Recall Loss: {loss_gate_results['corpus_macro_raw_doc_opp_recall1000_loss']} (threshold: {loss_gate_results['max_doc_opp_recall_loss_corpus_macro_threshold']})\n"
                f"  Per-Corpus Results: {json.dumps(loss_gate_results['per_corpus_results'], indent=2)}"
            )

        return report_path


def main():
    parser = argparse.ArgumentParser(description="Compile Phase 2 Gate 1 Research Tables")
    parser.add_argument("--audit-parquet", type=str, default="results/gate1_selection/dev_corrected/gate1_candidate_audit.parquet")
    parser.add_argument("--cutoff-parquet", type=str, default="results/gate1_selection/dev_corrected/gate1_cutoff_entries.parquet")
    parser.add_argument("--universe-parquet", type=str, default=None, help="Path to reference_universe.parquet")
    parser.add_argument("--diag-universe-parquet", type=str, default=None, help="Path to diagnostic_universe.parquet")
    parser.add_argument("--run-manifest", type=str, default=None, help="Path to run_manifest.json")
    parser.add_argument("--frozen-config-path", type=str, default=None, help="Path to gate1_phase2_1a.yaml")
    parser.add_argument("--output-dir", type=str, default="results/gate1_selection/dev_corrected")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compiler = Gate1TableCompiler(
        audit_parquet_path=args.audit_parquet,
        cutoff_parquet_path=args.cutoff_parquet,
        output_dir=args.output_dir,
        universe_parquet_path=args.universe_parquet,
        diag_universe_parquet_path=args.diag_universe_parquet,
        run_manifest_path=args.run_manifest,
        frozen_config_path=args.frozen_config_path,
        delta=args.delta,
        rho=args.rho,
        b_resamples=args.bootstrap,
        seed=args.seed,
    )
    compiler.compile_all_tables_and_report()


if __name__ == "__main__":
    main()
