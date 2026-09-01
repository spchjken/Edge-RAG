"""
src/pipeline_v2/expansion/v7_aspect_extractor.py

V7: 5-Phase High-Speed Anchored Lexical-Semantic Retriever.
Encapsulates Phase 2 (Aspect Anchors & POS Priors), Phase 3 (Dense Probing & Adaptive Gating),
and Phase 4 (IT-MPE Mass Allocation & Sparse Vector Compilation).
"""

import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
import torch
import numpy as np

from ..indexer.corpus_idf_registry import CorpusIDFRegistry
from ..indexer.dense_vocab_matrix import DenseVocabMatrix
from ..indexer.analyzer import EdgeRAGAnalyzer, LUCENE_STOPWORDS


class POSTaggerHelper:
    """POS Tagger using NLTK Penn Treebank with heuristic entity protection."""

    def __init__(self):
        self.nltk_available = False
        try:
            import nltk
            nltk.pos_tag(["test"])
            self.nltk_available = True
        except Exception:
            self.nltk_available = False

    def tag_query(self, query: str, heuristic_entities: Set[str]) -> Dict[str, str]:
        """
        Maps lowercased token to POS category: 'noun', 'verb', 'modifier'.
        """
        raw_tokens = re.findall(r'\b[A-Za-z0-9\-_.]+\b', query)
        token_to_cat: Dict[str, str] = {}
        if self.nltk_available and raw_tokens:
            try:
                import nltk
                tags = nltk.pos_tag(raw_tokens)
                for word, tag in tags:
                    w_lower = word.lower()
                    if w_lower in heuristic_entities:
                        token_to_cat[w_lower] = "noun"
                    elif tag.startswith("NN") or tag.startswith("PRP"):
                        token_to_cat[w_lower] = "noun"
                    elif tag.startswith("VB"):
                        token_to_cat[w_lower] = "verb"
                    elif tag.startswith("JJ") or tag.startswith("RB"):
                        token_to_cat[w_lower] = "modifier"
                    else:
                        token_to_cat[w_lower] = "modifier"
            except Exception:
                pass

        # Rule-based fallback for any unassigned tokens
        for w in raw_tokens:
            w_lower = w.lower()
            if w_lower not in token_to_cat:
                if w_lower in heuristic_entities or len(w_lower) > 3:
                    token_to_cat[w_lower] = "noun"
                else:
                    token_to_cat[w_lower] = "modifier"
        return token_to_cat


class V7AspectExtractor:
    """
    V7 5-Phase Anchored Lexical-Semantic Retriever.

    Parameters:
        idf_registry: Global CorpusIDFRegistry for non-negative Lucene IDF and stem mapping
        vocab_matrix: DenseVocabMatrix containing cached coverage pool and stem embeddings
        analyzer: EdgeRAGAnalyzer enforcing WordNet suppletion overrides and KStem
        tau_base: Base cosine similarity cutoff (default: 0.55)
        delta_tau: Signed adaptive gate slope across anchor IDF (default: 0.0)
        beta: Similarity balance (1.0 = pure anchor-to-candidate, <1.0 = Dual BGE)
        mu_ceil: Query-level IT-MPE expansion budget ceiling (default: 0.50)
        eta: Budget direction scaling factor (default: 0.0)
        pos_ratios: Prior weights by POS category (noun: 1.0, verb: 0.75, modifier: 0.60)
        bailout_tau_idf: Rare anchor IDF threshold for triggering boundary candidate rescue
        min_len_rescue: Minimum anchor string length for bailout candidate generation
    """

    def __init__(
        self,
        idf_registry: CorpusIDFRegistry,
        vocab_matrix: DenseVocabMatrix,
        analyzer: Optional[EdgeRAGAnalyzer] = None,
        tau_base: float = 0.55,
        delta_tau: float = 0.0,
        beta: float = 1.0,
        mu_ceil: float = 0.50,
        eta: float = 0.0,
        pos_ratios: Optional[Dict[str, float]] = None,
        bailout_tau_idf: float = 3.0,
        min_len_rescue: int = 3,
        mass_floor: float = 0.0,
        epsilon: Optional[float] = None,
        allocation: str = "normalized_cosine",
        gate_variant: str = "single",
        trace: bool = False
    ):
        self.idf_registry = idf_registry
        self.vocab_matrix = vocab_matrix
        self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
        self.tau_base = tau_base
        self.delta_tau = delta_tau
        self.beta = beta
        self.mu_ceil = mu_ceil
        self.eta = eta
        self.pos_ratios = pos_ratios if pos_ratios is not None else {"noun": 1.0, "verb": 0.75, "modifier": 0.60}
        self.bailout_tau_idf = bailout_tau_idf
        self.min_len_rescue = min_len_rescue
        self.mass_floor = float(epsilon) if epsilon is not None else float(mass_floor)
        self.allocation = allocation
        self.gate_variant = gate_variant
        self.trace = trace
        self.pos_tagger = POSTaggerHelper()
        self.vocab_term_to_idx = {t: i for i, t in enumerate(self.vocab_matrix.vocab_terms)}
        self.vocab_idfs_np = np.array(
            [self.idf_registry.get_idf(t) for t in self.vocab_matrix.vocab_terms],
            dtype=np.float32
        ) if self.vocab_matrix.vocab_terms else np.empty(0, dtype=np.float32)

    def extract_heuristics(self, query: str) -> List[str]:
        """Extracts acronyms, hyphenated terms, and quoted phrases."""
        entities = []
        # Acronyms (e.g., NASA, RAG, API)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        entities.extend(acronyms)
        # Hyphenated / compound terms (e.g., qwen2.5-7b, high-speed)
        hyphenated = re.findall(r'\b[a-zA-Z0-9]+-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\b', query)
        entities.extend(hyphenated)
        # Exact quotes
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        return list(dict.fromkeys(entities))

    def extract(self, query: str, top_candidate_chunks: Optional[List[str]] = None, trace: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executes V7 Phase 2 -> Phase 3 -> Phase 4.
        """
        do_trace = self.trace if trace is None else trace

        # --- PHASE 2: Anchored Aspect Groups & POS Prior Weighting ---
        t_p2_0 = time.perf_counter()
        heuristic_entities = self.extract_heuristics(query)
        heur_set = set(h.lower() for h in heuristic_entities)

        # 1. Analyzed stems as anchors (p = 1.0)
        analyzed_anchors = self.analyzer.analyze(query)
        analyzed_heuristics = []
        for h in heuristic_entities:
            analyzed_heuristics.extend(self.analyzer.analyze(h))
        all_candidate_anchors = list(dict.fromkeys(analyzed_anchors + analyzed_heuristics))

        # 2. Dedup & OOV clamping via registry
        distinct_anchors = []
        for a in all_candidate_anchors:
            if a in self.idf_registry.doc_freqs or a in heur_set or len(a) >= 2:
                distinct_anchors.append(a)
        distinct_anchors = list(dict.fromkeys(distinct_anchors))
        t_p2_analyze = time.perf_counter()

        # 3. Penn Treebank POS Prior Mapping
        pos_map = self.pos_tagger.tag_query(query, heur_set)

        # 4. Assign base anchor weights
        anchor_base_weights: Dict[str, float] = {}
        for a in distinct_anchors:
            cat = pos_map.get(a, "noun" if (a in heur_set or len(a) > 3) else "modifier")
            anchor_base_weights[a] = float(self.pos_ratios.get(cat, 1.0))
        t_p2_pos = time.perf_counter()

        max_corpus_idf = max(self.idf_registry.max_idf, 1.0)
        vocab_terms_set = set(self.vocab_matrix.vocab_terms)

        # Batch encode all query anchors in a single FlagEmbedding call (1-Pass GEMM)
        anchor_vecs = self.vocab_matrix.encode_terms(distinct_anchors)  # [N_anchors, 384]
        t_p2_anchor = time.perf_counter()

        # 5. Anchor Bailout (O(1) Pre-Indexed Candidate Lookup & Semantic Assessment)
        bailed_candidates_per_anchor: Dict[str, List[Dict[str, Any]]] = {a: [] for a in distinct_anchors}
        total_bailed = 0

        for i, a in enumerate(distinct_anchors):
            anchor_idf = self.idf_registry.get_idf(a)
            # Bailout Gate: rare anchor (IDF >= bailout_tau_idf, len >= min_len_rescue) OR technical entity
            if (anchor_idf >= self.bailout_tau_idf and len(a) >= self.min_len_rescue) or (a in heur_set):
                anchor_emb = anchor_vecs[i:i+1] if anchor_vecs.numel() > 0 else None
                if anchor_emb is None:
                    continue

                tau_sim_a = self.tau_base + self.delta_tau * min(1.0, anchor_idf / max_corpus_idf)

                # O(1) Pre-indexed Boundary Candidates Lookup from Registry
                matching_terms = [
                    t for t in self.idf_registry.get_boundary_candidates(a)
                    if t not in vocab_terms_set and t != a and len(t) >= 2
                ]

                if matching_terms:
                    valid_stems, cand_mat = self.vocab_matrix.get_stem_embeddings_batch(matching_terms)
                    if cand_mat.numel() > 0:
                        cossims = torch.mv(cand_mat, anchor_emb.squeeze(0)).cpu().numpy()
                        above_indices = np.where(cossims >= tau_sim_a)[0]

                        for idx in above_indices:
                            cand_term = valid_stems[idx]
                            cossim = float(cossims[idx])
                            cand_idf = self.idf_registry.get_idf(cand_term)
                            damped_w = anchor_base_weights[a] * min(1.0, anchor_idf / max(cand_idf, 1e-6))
                            bailed_candidates_per_anchor[a].append({
                                "term": cand_term,
                                "similarity": round(cossim, 4),
                                "idf": round(cand_idf, 4),
                                "weight": round(damped_w, 4)
                            })
                            total_bailed += 1
        t_p2_bail = time.perf_counter()

        # --- PHASE 3: Dense Semantic Probing & Adaptive Gating ---
        query_sim_np = None
        if self.vocab_matrix.vocab_embeddings is not None and self.vocab_matrix.vocab_embeddings.numel() > 0 and anchor_vecs.numel() > 0:
            sim_matrix = torch.mm(anchor_vecs, self.vocab_matrix.vocab_embeddings.T)  # [N_anchors, N_vocab]
            if self.beta < 1.0 or self.gate_variant in ("two_gate", "soft_reweight"):
                query_vec = self.vocab_matrix.encode_query(query)
                query_sim = torch.mm(query_vec, self.vocab_matrix.vocab_embeddings.T)
                query_sim_np = query_sim.cpu().numpy()[0]
                if self.beta < 1.0:
                    sim_matrix = self.beta * sim_matrix + (1.0 - self.beta) * query_sim
            sim_matrix_np = sim_matrix.cpu().numpy()
        else:
            sim_matrix_np = np.empty((len(distinct_anchors), 0))
        t_p3_prob = time.perf_counter()

        # --- PHASE 4: Vectorized SIMD IT-MPE Mass Allocation & Sparse Vector Compilation ---
        anchor_idfs_np = np.array([self.idf_registry.get_idf(a) for a in distinct_anchors], dtype=np.float32)
        anchor_base_weights_np = np.array([anchor_base_weights[a] for a in distinct_anchors], dtype=np.float32)
        max_query_idf = float(np.max(anchor_idfs_np)) if len(anchor_idfs_np) > 0 else max_corpus_idf

        # Query Expansion Budget mu(Q)
        mu_q = self.mu_ceil * (1.0 - self.eta * min(1.0, max_query_idf / max_corpus_idf))
        mu_q = max(0.0, min(1.0, mu_q))

        tau_sim_vec = self.tau_base + self.delta_tau * np.minimum(1.0, anchor_idfs_np / max_corpus_idf)
        vocab_terms = self.vocab_matrix.vocab_terms

        # Step 4: Fast Vectorized Anchor Base-Weight Init
        final_term_weights: Dict[str, float] = dict(zip(distinct_anchors, anchor_base_weights_np.tolist()))

        # Apply Bailout Candidates Directly (Outside mu budget)
        for a in distinct_anchors:
            for b in bailed_candidates_per_anchor.get(a, []):
                final_term_weights[b["term"]] = final_term_weights.get(b["term"], 0.0) + b["weight"]

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0
        aspects = []
        aspect_traces = []

        if len(distinct_anchors) > 0 and sim_matrix_np.shape[1] > 0 and mu_q > 0:
            effective_sims = sim_matrix_np.copy()

            # Handle Gate Variants
            if self.gate_variant == "two_gate" and query_sim_np is not None:
                effective_sims = np.where(query_sim_np[None, :] >= 0.40, effective_sims, 0.0)
            elif self.gate_variant == "soft_reweight" and query_sim_np is not None:
                effective_sims = effective_sims * np.maximum(0.0, query_sim_np[None, :])

            # 1. Vectorized Boolean Mask
            mask = (effective_sims >= tau_sim_vec[:, None])

            # Step 3: Fast Vectorized Self-Anchor Mask
            anchor_cols = [self.vocab_term_to_idx[a] for a in distinct_anchors if a in self.vocab_term_to_idx]
            if anchor_cols:
                rows = [i for i, a in enumerate(distinct_anchors) if a in self.vocab_term_to_idx]
                mask[rows, anchor_cols] = False

            total_cands_above_tau = int(np.sum(mask))
            masked_sims = np.where(mask, effective_sims, 0.0)

            # 2. Probability Allocation Matrix p(s | a)
            if self.allocation == "uniform":
                counts = np.sum(mask, axis=1, keepdims=True)
                p_cond_mat = np.where(mask, 1.0 / np.where(counts == 0, 1.0, counts), 0.0)
            elif "softmax" in self.allocation:
                tau = 0.1 if "0.1" in self.allocation else 1.0
                exp_sims = np.exp(np.where(mask, (effective_sims - np.max(effective_sims, axis=1, keepdims=True)) / tau, -100.0))
                exp_sims = np.where(mask, exp_sims, 0.0)
                sum_exp = np.sum(exp_sims, axis=1, keepdims=True)
                p_cond_mat = exp_sims / np.where(sum_exp == 0, 1.0, sum_exp)
            else:
                # Default: normalized_cosine
                sum_sims = np.sum(masked_sims, axis=1, keepdims=True)
                p_cond_mat = masked_sims / np.where(sum_sims == 0, 1.0, sum_sims)

            # 3. Vectorized Score-Space Damping: min(1.0, IDF_a / IDF_s)
            vocab_idfs = self.vocab_idfs_np[None, :] if len(self.vocab_idfs_np) > 0 else np.ones((1, len(vocab_terms)), dtype=np.float32)
            damping_mat = np.minimum(1.0, anchor_idfs_np[:, None] / np.maximum(vocab_idfs, 1e-6))

            # 4. Injected Weights Matrix: W[i, j] = w_a * damping * mu_q * p(s | a)
            w_syn_mat = anchor_base_weights_np[:, None] * damping_mat * (mu_q * p_cond_mat)

            # 5. Mass Floor Pruning: drop w(s | a) < epsilon * w(a)
            if self.mass_floor > 0.0:
                floor_thresh = self.mass_floor * anchor_base_weights_np[:, None]
                w_syn_mat = np.where(w_syn_mat >= floor_thresh, w_syn_mat, 0.0)

            # Step 5: Vectorized Column-wise Collision Sum into Sparse Dictionary
            syn_weights_per_term = np.sum(w_syn_mat, axis=0)
            non_zero_vocab_indices = np.where(syn_weights_per_term > 0)[0]
            rounded_weights = np.round(syn_weights_per_term[non_zero_vocab_indices], 4)
            for v_idx, w_val in zip(non_zero_vocab_indices, rounded_weights):
                cand = vocab_terms[v_idx]
                final_term_weights[cand] = final_term_weights.get(cand, 0.0) + float(w_val)

            # Step 1: Optional/Lazy Trace Compilation
            if do_trace:
                for i, a in enumerate(distinct_anchors):
                    active_cand_indices = np.where(w_syn_mat[i] > 0)[0]
                    if len(active_cand_indices) == 0:
                        starved_aspects += 1
                    syn_entries = []
                    for v_idx in active_cand_indices:
                        syn_entries.append({
                            "term": vocab_terms[v_idx],
                            "similarity": round(float(effective_sims[i, v_idx]), 4),
                            "weight": round(float(w_syn_mat[i, v_idx]), 4)
                        })
                        total_synonyms_injected += 1

                    aspect_data = {
                        "anchor": a,
                        "anchor_reps": 1,
                        "anchor_base_weight": float(anchor_base_weights_np[i]),
                        "anchor_idf": round(float(anchor_idfs_np[i]), 3),
                        "tau_sim_a": round(float(tau_sim_vec[i]), 4),
                        "synonyms": syn_entries,
                        "bailed_candidates": bailed_candidates_per_anchor.get(a, [])
                    }
                    aspects.append(aspect_data)
                    aspect_traces.append(aspect_data)
                unique_syns_count = len({s["term"] for a in aspects for s in a["synonyms"]})
                aug_tokens = [a["anchor"] for a in aspects] + [s["term"] for a in aspects for s in a["synonyms"]]
            else:
                total_synonyms_injected = int(np.count_nonzero(w_syn_mat))
                starved_aspects = int(np.sum(np.count_nonzero(w_syn_mat, axis=1) == 0))
                unique_syns_count = len(non_zero_vocab_indices)
                aug_tokens = list(final_term_weights.keys())
        else:
            if do_trace:
                for i, a in enumerate(distinct_anchors):
                    aspect_data = {
                        "anchor": a,
                        "anchor_reps": 1,
                        "anchor_base_weight": float(anchor_base_weights_np[i]),
                        "anchor_idf": round(float(anchor_idfs_np[i]), 3),
                        "tau_sim_a": round(float(tau_sim_vec[i]), 4),
                        "synonyms": [],
                        "bailed_candidates": bailed_candidates_per_anchor.get(a, [])
                    }
                    aspects.append(aspect_data)
                    aspect_traces.append(aspect_data)
            unique_syns_count = 0
            aug_tokens = list(distinct_anchors)

        t_p4_end = time.perf_counter()

        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": final_term_weights,
            "telemetry": {
                "num_anchors": len(distinct_anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "total_synonym_links": total_synonyms_injected,
                "unique_synonyms": unique_syns_count,
                "final_qvec_len": len(final_term_weights),
                "total_bailed_candidates": total_bailed,
                "starved_aspects_count": starved_aspects,
                "expansion_budget_mu": float(mu_q),
                "qaug_length": len(aug_tokens),
                "timings_ms": {
                    "anchor_analyze": round((t_p2_analyze - t_p2_0) * 1000, 3),
                    "anchor_pos": round((t_p2_pos - t_p2_analyze) * 1000, 3),
                    "anchor_bge": round((t_p2_anchor - t_p2_pos) * 1000, 3),
                    "anchor_encoding": round((t_p2_anchor - t_p2_0) * 1000, 3),
                    "boundary_bailout": round((t_p2_bail - t_p2_anchor) * 1000, 3),
                    "batch_gemm_probing": round((t_p3_prob - t_p2_bail) * 1000, 3),
                    "itmpe_allocation": round((t_p4_end - t_p3_prob) * 1000, 3),
                    "total_expansion": round((t_p4_end - t_p2_0) * 1000, 3)
                }
            },
            "aspect_traces": aspect_traces
        }
