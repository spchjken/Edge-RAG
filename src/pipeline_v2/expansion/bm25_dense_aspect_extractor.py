import os
import re
import yaml
from typing import List, Dict, Any, Tuple, Optional, Set
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..indexer.corpus_idf_registry import CorpusIDFRegistry
from ..indexer.dense_vocab_matrix import DenseVocabMatrix
from ..indexer.corpus_vocab_builder import CorpusVocabBuilder
from ..indexer.analyzer import EdgeRAGAnalyzer, LUCENE_STOPWORDS
from ..indexer.tokenizer import EdgeRAGTokenizer
from .v7_aspect_extractor import V7AspectExtractor, POSTaggerHelper


class BM25DenseAspectExtractor:
    """
    Hybrid BM25 IDF + Aspect-Grouped Dense Vocabulary Query Expansion Module.
    Supports Schemas 1-6 (legacy token repetition) and delegates V7 to V7AspectExtractor.
    """

    def __init__(
        self,
        idf_registry: CorpusIDFRegistry,
        vocab_matrix: DenseVocabMatrix,
        schema: str = "BM25Dense_V7",
        p: float = 0.50,
        C_exp: int = 2,
        tau_sim: float = 0.55,
        beta: float = 1.0,
        c: int = -1,
        r_min: int = 2,
        r_max: int = 4,
        n_reps: int = 3,
        tau_base: float = 0.55,
        delta_tau: float = 0.0,
        mu_ceil: float = 0.50,
        eta: float = 0.0,
        pos_ratios: Optional[Dict[str, float]] = None,
        bailout_tau_idf: float = 3.0,
        min_len_rescue: int = 3,
        analyzer: Optional[EdgeRAGAnalyzer] = None
    ):
        self.idf_registry = idf_registry
        self.vocab_matrix = vocab_matrix
        self.schema = schema
        self.p = p
        self.C_exp = C_exp
        self.tau_sim = tau_sim
        self.beta = beta
        self.c = c
        self.r_min = r_min
        self.r_max = r_max
        self.n_reps = n_reps
        self.tau_base = tau_base
        self.delta_tau = delta_tau
        self.mu_ceil = mu_ceil
        self.eta = eta
        self.pos_ratios = pos_ratios if pos_ratios is not None else {"noun": 1.0, "verb": 0.75, "modifier": 0.60}
        self.bailout_tau_idf = bailout_tau_idf
        self.min_len_rescue = min_len_rescue
        self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
        self.pos_tagger = POSTaggerHelper()

        # Dedicated V7 Retriever Delegate
        self._v7_extractor = V7AspectExtractor(
            idf_registry=self.idf_registry,
            vocab_matrix=self.vocab_matrix,
            analyzer=self.analyzer,
            tau_base=self.tau_base,
            delta_tau=self.delta_tau,
            beta=self.beta,
            mu_ceil=self.mu_ceil,
            eta=self.eta,
            pos_ratios=self.pos_ratios,
            bailout_tau_idf=self.bailout_tau_idf,
            min_len_rescue=self.min_len_rescue
        )

    @classmethod
    def from_config(
        cls,
        idf_registry: CorpusIDFRegistry,
        vocab_matrix: DenseVocabMatrix,
        config_path: str = "configs/pipeline_v2.yaml",
        **overrides
    ) -> "BM25DenseAspectExtractor":
        """
        Instantiates BM25DenseAspectExtractor directly from YAML configuration,
        allowing configs/pipeline_v2.yaml to serve as the single source of truth.
        """
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
                pipe_v2 = raw_cfg.get("pipeline_v2", {})
                config_data.update(pipe_v2.get("expansion", {}))
                if "schema" in pipe_v2:
                    config_data["schema"] = pipe_v2["schema"]

        # Merge YAML defaults with runtime overrides
        merged_kwargs = {**config_data, **overrides}
        valid_keys = {
            "schema", "p", "C_exp", "tau_sim", "beta", "c", "r_min", "r_max", "n_reps",
            "tau_base", "delta_tau", "mu_ceil", "eta", "pos_ratios", "bailout_tau_idf",
            "min_len_rescue"
        }
        filtered_kwargs = {k: v for k, v in merged_kwargs.items() if k in valid_keys}

        return cls(idf_registry=idf_registry, vocab_matrix=vocab_matrix, **filtered_kwargs)

    def extract_heuristics(self, query: str) -> List[str]:
        """Extracts technical acronyms, hyphenated terms, proper nouns, and quotes."""
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        hyphenated = re.findall(r'\b[A-Za-z0-9\.]+(?:-[A-Za-z0-9\.]+)+\b', query)
        quotes = re.findall(r'"([^"]+)"', query)
        
        entities = set(acronyms + hyphenated + quotes)
        return list(entities)

    def extract(self, query: str, top_candidate_chunks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main extraction entry point.
        Returns:
            {
                "aspects": [...],
                "augmented_token_list": [...],
                "term_weights": {...},
                "telemetry": {...}
            }
        """
        # Schema 7: 5-Phase Anchored Lexical-Semantic Retriever (V7)
        if self.schema in ("BM25Dense_V7", "pipeline_v2_v7", "v7", "BM25Dense_5Phase"):
            return self._v7_extractor.extract(query, top_candidate_chunks)

    def _compute_dual_sim(self, anchors: List[str], query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Dual BGE Similarity matrix between all anchors & candidate vocab terms.
        Dual_Sim = beta * CosSim(A_k, v) + (1 - beta) * CosSim(Q_full, v)
        """
        if not anchors or self.vocab_matrix.vocab_embeddings is None or len(self.vocab_matrix.vocab_terms) == 0:
            return torch.empty((len(anchors), 0)), torch.empty((0,))

        # Embed anchors & full query
        anchor_vecs = self.vocab_matrix.encode_terms(anchors) # [N_anchors, dim]
        query_vec = self.vocab_matrix.encode_query(query)     # [1, dim]

        # Cosine similarities (vectors are already L2 normalized)
        # CosSim(A_k, v) -> [N_anchors, N_vocab]
        cos_sim_anchors = torch.mm(anchor_vecs, self.vocab_matrix.vocab_embeddings.T)
        # CosSim(Q_full, v) -> [1, N_vocab]
        cos_sim_query = torch.mm(query_vec, self.vocab_matrix.vocab_embeddings.T)

        # Combine Dual Similarity
        dual_sim_matrix = self.beta * cos_sim_anchors + (1.0 - self.beta) * cos_sim_query
        return dual_sim_matrix, cos_sim_query.squeeze(0)

    def _build_augmented_tokens(self, aspects: List[Dict[str, Any]]) -> List[str]:
        """
        Builds ordered list of unique lowercase tokens present across all aspect terms.
        Deduplicates repeated tokens while preserving first-appearance order.
        """
        tokens = []
        for asp in aspects:
            for kw in asp.get("keywords", []):
                term = kw["term"]
                for token in term.lower().split():
                    if token not in tokens:
                        tokens.append(token)
        return tokens

    def _build_term_weights(self, aspects: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Builds a weighted term dict for vectorized BM25 retrieval.
        - Core anchors: receive their full assigned anchor_weight (e.g. 3.0, 4.0, etc.).
        - Expansion terms / synonyms: capped at max 1.0 total weight across all aspects.
        - For multi-word terms (bigrams), weight is distributed to constituent tokens.
        """
        anchor_weights: Dict[str, float] = {}
        expansion_weights: Dict[str, float] = {}

        for asp in aspects:
            for idx, kw in enumerate(asp.get("keywords", [])):
                term = kw["term"]
                is_anchor = kw.get("is_anchor", (idx == 0 and kw.get("anchor_weight", 1.0) > 1.0))
                weight = float(kw.get("weight", 1.0))
                anchor_w = kw.get("anchor_weight", kw.get("repeat_count", None))

                for token in term.lower().split():
                    if is_anchor:
                        multiplier = float(anchor_w) if anchor_w is not None else (3.0 if weight >= 0.99 else 2.0)
                        anchor_weights[token] = anchor_weights.get(token, 0.0) + multiplier
                    else:
                        multiplier = min(1.0, float(weight))
                        expansion_weights[token] = expansion_weights.get(token, 0.0) + multiplier

        final_term_weights: Dict[str, float] = {}
        # 1. Anchors get full weight
        for token, w in anchor_weights.items():
            final_term_weights[token] = round(w, 4)

        # 2. Expansion terms are capped at 1.0 total
        for token, w in expansion_weights.items():
            if token not in final_term_weights:
                final_term_weights[token] = round(min(1.0, w), 4)

        return final_term_weights


    def _extract_aspect_inject(self, query: str, anchors: List[str]) -> Dict[str, Any]:
        """Schema 1: Force-injected anchors (w=1.0) + Top C_exp per-aspect expansion."""
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        aspect_traces = []

        vocab_terms = self.vocab_matrix.vocab_terms
        max_corpus_idf = self.idf_registry.max_idf if self.idf_registry.max_idf > 0 else 1.0

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0

        for i, anchor in enumerate(anchors):
            aspect_kw = [{"term": anchor, "weight": 1.0, "anchor_weight": float(self.n_reps), "is_anchor": True}]
            anchor_idf = self.idf_registry.get_idf(anchor)
            cands_above_tau = []
            selected_synonyms = []
            
            if dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                # Filter by tau_sim threshold
                valid_mask = sims >= self.tau_sim
                valid_indices = torch.where(valid_mask)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    term_idf = self.idf_registry.get_idf(term)
                    final_weight = sim_val * (0.5 + 0.5 * (term_idf / max_corpus_idf))
                    candidates.append((term, min(0.95, final_weight), sim_val, term_idf))

                total_cands_above_tau += len(candidates)
                if len(candidates) < self.C_exp:
                    starved_aspects += 1

                candidates.sort(key=lambda x: x[1], reverse=True)
                for item in candidates[:10]:
                    cands_above_tau.append({
                        "term": item[0],
                        "final_weight": float(item[1]),
                        "similarity": float(item[2]),
                        "idf": float(item[3])
                    })

                for term, weight, _, _ in candidates[:self.C_exp]:
                    aspect_kw.append({"term": term, "weight": float(weight), "anchor_weight": 1.0, "is_anchor": False})
                    selected_synonyms.append(term)
                    total_synonyms_injected += 1
            else:
                starved_aspects += 1

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })
            aspect_traces.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "anchor_term": anchor,
                "is_heuristic_entity": False,
                "anchor_idf": float(anchor_idf),
                "anchor_weight": float(self.n_reps),
                "capacity_cap": self.C_exp,
                "total_candidates_above_tau": len(cands_above_tau),
                "candidates_above_tau": cands_above_tau,
                "injected_synonyms": selected_synonyms
            })

        aug_tokens = self._build_augmented_tokens(aspects)
        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": self._build_term_weights(aspects),
            "telemetry": {
                "num_anchors": len(anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "starved_aspects_count": starved_aspects,
                "avg_r_anchor": 3.0,
                "qaug_length": len(aug_tokens),
                "aspect_traces": aspect_traces
            }
        }

    def _extract_aspect_weighted(self, query: str, anchors: List[str]) -> Dict[str, Any]:
        """Schema 2: Aspect-Weighted IDF Anchors + Soft-Dampened (gamma=0.5) Expansion."""
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        max_query_idf = max([self.idf_registry.get_idf(a) for a in anchors]) if anchors else 1.0
        vocab_terms = self.vocab_matrix.vocab_terms

        for i, anchor in enumerate(anchors):
            anchor_idf = self.idf_registry.get_idf(anchor)
            anchor_weight = min(1.0, max(0.5, anchor_idf / max_query_idf)) if max_query_idf > 0 else 1.0
            aspect_kw = [{"term": anchor, "weight": float(anchor_weight)}]

            if dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                valid_indices = torch.where(sims >= self.tau_sim)[0]
                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    # Soft dampening factor gamma = 0.5
                    dampened_weight = 0.5 * sim_val * anchor_weight
                    candidates.append((term, min(0.90, dampened_weight)))

                candidates.sort(key=lambda x: x[1], reverse=True)
                for term, weight in candidates[:self.C_exp]:
                    aspect_kw.append({"term": term, "weight": float(weight)})

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })

        return {
            "aspects": aspects,
            "augmented_token_list": self._build_augmented_tokens(aspects),
            "term_weights": self._build_term_weights(aspects)
        }

    def _extract_local_cascade(self, query: str, anchors: List[str], top_chunks: List[str]) -> Dict[str, Any]:
        """Schema 3: Ephemeral Local-Chunk Aspect Expansion."""
        local_builder = CorpusVocabBuilder(self.idf_registry, max_vocab_size=300)
        local_vocab = local_builder.build_clean_vocabulary(top_chunks)
        local_matrix = DenseVocabMatrix(use_gpu=self.vocab_matrix.use_gpu)
        local_matrix.build_matrix(local_vocab)

        old_matrix = self.vocab_matrix
        self.vocab_matrix = local_matrix
        result = self._extract_aspect_inject(query, anchors)
        self.vocab_matrix = old_matrix
        return result

    def _extract_aspect_fusion(self, query: str, anchors: List[str]) -> Dict[str, Any]:
        """Schema 4: HAC Aspect Clustering + Joint BM25/BGE Score Fusion."""
        if not anchors:
            return {"aspects": [], "augmented_token_list": [], "term_weights": {}}

        # Step A: HAC Clustering on Anchors
        anchor_vecs = self.vocab_matrix.encode_terms(anchors).numpy()
        if len(anchors) > 1:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.35, # Similarity threshold = 0.65
                metric="cosine",
                linkage="average"
            ).fit(anchor_vecs)
            labels = clustering.labels_
        else:
            labels = [0]

        # Group anchors by cluster label
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(anchors[idx])

        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        vocab_terms = self.vocab_matrix.vocab_terms

        for label, cluster_anchors in clusters.items():
            cluster_name = "_".join(cluster_anchors[:2])
            aspect_kw = [{"term": a, "weight": 1.0} for a in cluster_anchors]

            if dual_sim_matrix.numel() > 0:
                # Average dual sim across cluster anchors
                anchor_indices = [anchors.index(a) for a in cluster_anchors]
                avg_sims = dual_sim_matrix[anchor_indices].mean(dim=0)
                valid_indices = torch.where(avg_sims >= self.tau_sim)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() in [a.lower() for a in cluster_anchors]:
                        continue
                    sim_val = avg_sims[idx].item()
                    norm_idf = self.idf_registry.get_normalized_idf(term)
                    fused_score = 0.65 * norm_idf + 0.35 * sim_val
                    candidates.append((term, min(0.95, fused_score)))

                candidates.sort(key=lambda x: x[1], reverse=True)
                for term, weight in candidates[:self.C_exp]:
                    aspect_kw.append({"term": term, "weight": float(weight)})

            aspects.append({
                "aspect_id": f"asp_cluster_{label}_{cluster_name}",
                "keywords": aspect_kw
            })

        return {
            "aspects": aspects,
            "augmented_token_list": self._build_augmented_tokens(aspects),
            "term_weights": self._build_term_weights(aspects)
        }

    def _extract_fixed_rep_dynamic_capacity(
        self,
        query: str,
        anchors: List[str],
        heuristic_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Schema 5a: Fixed Anchor Repetition (n_reps) + Dynamic Per-Aspect Capacity Capping.
        Anchor repetition is fixed at n_reps for all anchors.
        Synonym capacity C_exp(A_k) = clamp(R_dynamic_IDF + c, 1, 5) scaled by Max Query IDF.
        """
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        aspect_traces = []
        vocab_terms = self.vocab_matrix.vocab_terms
        max_corpus_idf = self.idf_registry.max_idf if self.idf_registry.max_idf > 0 else 1.0
        heur_set = set(h.lower() for h in heuristic_entities)

        # Compute Max Query IDF
        anchor_idfs = [self.idf_registry.get_idf(a) for a in anchors]
        max_query_idf = max(anchor_idfs) if anchor_idfs and max(anchor_idfs) > 0 else max_corpus_idf

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0

        for i, anchor in enumerate(anchors):
            # Fixed repetition count for anchors
            r_anchor = self.n_reps
            is_heur = anchor.lower() in heur_set

            # Dynamic IDF tier calculation for synonym capacity
            anchor_idf = self.idf_registry.get_idf(anchor)
            if is_heur:
                r_dynamic_idf = self.r_max
            else:
                scaled = self.r_min + (self.r_max - self.r_min) * (anchor_idf / max_query_idf)
                r_dynamic_idf = int(np.clip(np.round(scaled), self.r_min, self.r_max))

            aspect_kw = [{"term": anchor, "weight": 1.0, "anchor_weight": float(r_anchor), "is_anchor": True}]
            c_exp_aspect = int(np.clip(r_dynamic_idf + self.c, 1, 5))
            cands_above_tau = []
            selected_synonyms = []

            if dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                valid_mask = sims >= self.tau_sim
                valid_indices = torch.where(valid_mask)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    term_idf = self.idf_registry.get_idf(term)
                    final_weight = sim_val * (0.5 + 0.5 * (term_idf / max_corpus_idf))
                    candidates.append((term, min(0.95, final_weight), sim_val, term_idf))

                total_cands_above_tau += len(candidates)
                if len(candidates) < c_exp_aspect:
                    starved_aspects += 1

                candidates.sort(key=lambda x: x[1], reverse=True)
                for item in candidates[:10]:
                    cands_above_tau.append({
                        "term": item[0],
                        "final_weight": float(item[1]),
                        "similarity": float(item[2]),
                        "idf": float(item[3])
                    })

                for term, weight, _, _ in candidates[:c_exp_aspect]:
                    aspect_kw.append({"term": term, "weight": float(weight), "anchor_weight": 1.0, "is_anchor": False})
                    selected_synonyms.append(term)
                    total_synonyms_injected += 1
            else:
                starved_aspects += 1

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })
            aspect_traces.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "anchor_term": anchor,
                "is_heuristic_entity": is_heur,
                "anchor_idf": float(anchor_idf),
                "anchor_weight": float(r_anchor),
                "capacity_cap": c_exp_aspect,
                "total_candidates_above_tau": len(cands_above_tau),
                "candidates_above_tau": cands_above_tau,
                "injected_synonyms": selected_synonyms
            })

        aug_tokens = self._build_augmented_tokens(aspects)
        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": self._build_term_weights(aspects),
            "telemetry": {
                "num_anchors": len(anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "starved_aspects_count": starved_aspects,
                "avg_r_anchor": float(self.n_reps),
                "qaug_length": len(aug_tokens),
                "aspect_traces": aspect_traces
            }
        }

    def _extract_dynamic_aspect_inject(
        self,
        query: str,
        anchors: List[str],
        heuristic_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Schema 5b: Dynamic Anchor Repetition (R in [r_min, r_max]) + 
                   Coupled Synonym Capacity Capping (C_exp = R + c).
        Scaled using Max Query IDF & np.round.
        """
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        aspect_traces = []
        vocab_terms = self.vocab_matrix.vocab_terms
        max_corpus_idf = self.idf_registry.max_idf if self.idf_registry.max_idf > 0 else 1.0
        heur_set = set(h.lower() for h in heuristic_entities)

        # Compute Max Query IDF
        anchor_idfs = [self.idf_registry.get_idf(a) for a in anchors]
        max_query_idf = max(anchor_idfs) if anchor_idfs and max(anchor_idfs) > 0 else max_corpus_idf

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0
        r_anchors = []

        for i, anchor in enumerate(anchors):
            is_heur = anchor.lower() in heur_set
            anchor_idf = self.idf_registry.get_idf(anchor)

            # Dynamic R_anchor in [r_min, r_max] relative to Max Query IDF
            if is_heur:
                r_anchor = self.r_max
            else:
                scaled = self.r_min + (self.r_max - self.r_min) * (anchor_idf / max_query_idf)
                r_anchor = int(np.clip(np.round(scaled), self.r_min, self.r_max))

            r_anchors.append(r_anchor)
            aspect_kw = [{"term": anchor, "weight": 1.0, "anchor_weight": float(r_anchor), "is_anchor": True}]

            # Dynamic capacity cap for this aspect (C_exp = R_anchor + c, clamped in [1, 5])
            c_exp_aspect = int(np.clip(r_anchor + self.c, 1, 5))
            cands_above_tau = []
            selected_synonyms = []

            if dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                valid_mask = sims >= self.tau_sim
                valid_indices = torch.where(valid_mask)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    term_idf = self.idf_registry.get_idf(term)
                    final_weight = sim_val * (0.5 + 0.5 * (term_idf / max_corpus_idf))
                    candidates.append((term, min(0.95, final_weight), sim_val, term_idf))

                total_cands_above_tau += len(candidates)
                if len(candidates) < c_exp_aspect:
                    starved_aspects += 1

                candidates.sort(key=lambda x: x[1], reverse=True)
                for item in candidates[:10]:
                    cands_above_tau.append({
                        "term": item[0],
                        "final_weight": float(item[1]),
                        "similarity": float(item[2]),
                        "idf": float(item[3])
                    })

                for term, weight, _, _ in candidates[:c_exp_aspect]:
                    aspect_kw.append({"term": term, "weight": float(weight), "anchor_weight": 1.0, "is_anchor": False})
                    selected_synonyms.append(term)
                    total_synonyms_injected += 1
            else:
                starved_aspects += 1

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })
            aspect_traces.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "anchor_term": anchor,
                "is_heuristic_entity": is_heur,
                "anchor_idf": float(anchor_idf),
                "anchor_weight": float(r_anchor),
                "capacity_cap": c_exp_aspect,
                "total_candidates_above_tau": len(cands_above_tau),
                "candidates_above_tau": cands_above_tau,
                "injected_synonyms": selected_synonyms
            })

        aug_tokens = self._build_augmented_tokens(aspects)
        avg_r = float(np.mean(r_anchors)) if r_anchors else 3.0
        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": self._build_term_weights(aspects),
            "telemetry": {
                "num_anchors": len(anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "starved_aspects_count": starved_aspects,
                "avg_r_anchor": avg_r,
                "qaug_length": len(aug_tokens),
                "aspect_traces": aspect_traces
            }
        }

    def _score_and_deduplicate_anchors_v6(
        self,
        query: str,
        heuristic_entities: List[str],
        candidate_words: List[str],
        min_entity_idf: float = 1.0
    ) -> Tuple[List[str], List[str]]:
        """
        Schema 6 Anchor Selection:
        1. Fix B: Validates Heuristic Entities (IDF >= min_entity_idf).
        2. Query Centrality Scoring: Centrality_Score(w) = IDF(w) * CosSim(e_w, e_Q).
        3. Stem & Semantic Deduplication (suppressing redundant morphological variants).
        """
        # Validate heuristic entities against IDF threshold (Fix B)
        validated_entities = []
        unvalidated_entity_words = []
        for h in heuristic_entities:
            h_idf = self.idf_registry.get_idf(h)
            if h_idf >= min_entity_idf:
                validated_entities.append(h)
            else:
                unvalidated_entity_words.append(h)

        # Pool of words to rank by centrality (non-entities + unvalidated low-IDF entities)
        pool_words = [w for w in (candidate_words + unvalidated_entity_words) if w not in validated_entities]
        pool_words = list(dict.fromkeys(pool_words)) # Deduplicate pool while preserving order

        if not pool_words:
            return validated_entities, validated_entities

        # Compute query vector and term vectors
        query_vec = self.vocab_matrix.encode_query(query) # [1, dim]
        term_vecs = self.vocab_matrix.encode_terms(pool_words) # [N, dim]

        # CosSim(e_w, e_Q)
        cos_sim_q = torch.mm(term_vecs, query_vec.T).squeeze(1).cpu().numpy()

        scored_candidates = []
        for idx, w in enumerate(pool_words):
            idf_val = self.idf_registry.get_idf(w)
            sim_to_q = max(0.1, float(cos_sim_q[idx]))
            centrality_score = idf_val * sim_to_q
            scored_candidates.append((w, centrality_score, idf_val, term_vecs[idx]))

        # Sort by Centrality Score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Target number of regular anchors based on p ratio
        N_target = max(2, int(np.ceil(self.p * len(pool_words))))

        # Apply Stem & Semantic Deduplication
        selected_regular_anchors: List[str] = []
        selected_vectors: List[torch.Tensor] = []

        for w, score, idf_val, w_vec in scored_candidates:
            is_dup = False
            w_lower = w.lower()

            # 1. Morphological stem prefix/suffix check (e.g. upload vs uploads, connect vs connection)
            for sel in selected_regular_anchors:
                sel_lower = sel.lower()
                if (w_lower.startswith(sel_lower) or sel_lower.startswith(w_lower)) and min(len(w_lower), len(sel_lower)) >= 4:
                    is_dup = True
                    break

            # 2. High semantic cosine similarity check (>= 0.90)
            if not is_dup and len(selected_vectors) > 0:
                stacked_vecs = torch.stack(selected_vectors)
                sims_to_selected = torch.mm(w_vec.unsqueeze(0), stacked_vecs.T).squeeze(0)
                if torch.any(sims_to_selected >= 0.90):
                    is_dup = True

            if not is_dup:
                selected_regular_anchors.append(w)
                selected_vectors.append(w_vec)
                if len(selected_regular_anchors) >= N_target:
                    break

        all_anchors = validated_entities + [a for a in selected_regular_anchors if a not in validated_entities]
        return all_anchors, validated_entities

    def _extract_centrality_fixed_rep(
        self,
        query: str,
        anchors: List[str],
        validated_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Schema 6a: Query Centrality Anchors + Fixed Anchor Repetition (n_reps) + 
                   Zero-Floor Dynamic Synonym Capacity C_exp = max(0, R_dyn + c).
        """
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        aspect_traces = []
        vocab_terms = self.vocab_matrix.vocab_terms
        max_corpus_idf = self.idf_registry.max_idf if self.idf_registry.max_idf > 0 else 1.0
        val_entity_set = set(h.lower() for h in validated_entities)

        anchor_idfs = [self.idf_registry.get_idf(a) for a in anchors]
        max_query_idf = max(anchor_idfs) if anchor_idfs and max(anchor_idfs) > 0 else max_corpus_idf

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0

        for i, anchor in enumerate(anchors):
            is_heur = anchor.lower() in val_entity_set
            anchor_idf = self.idf_registry.get_idf(anchor)

            # Fixed anchor weight
            r_anchor = self.n_reps
            aspect_kw = [{"term": anchor, "weight": 1.0, "anchor_weight": float(r_anchor), "is_anchor": True}]

            # Dynamic capacity with Zero-Floor (Fix B)
            if is_heur:
                r_dyn = self.r_max
            else:
                scaled = self.r_min + (self.r_max - self.r_min) * (anchor_idf / max_query_idf)
                r_dyn = int(np.clip(np.round(scaled), self.r_min, self.r_max))

            c_exp_aspect = int(np.clip(r_dyn + self.c, 0, 5))
            cands_above_tau = []
            selected_synonyms = []

            if c_exp_aspect > 0 and dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                valid_mask = sims >= self.tau_sim
                valid_indices = torch.where(valid_mask)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    term_idf = self.idf_registry.get_idf(term)
                    final_weight = sim_val * (0.5 + 0.5 * (term_idf / max_corpus_idf))
                    candidates.append((term, min(0.95, final_weight), sim_val, term_idf))

                total_cands_above_tau += len(candidates)
                if len(candidates) < c_exp_aspect:
                    starved_aspects += 1

                candidates.sort(key=lambda x: x[1], reverse=True)
                for item in candidates[:10]:
                    cands_above_tau.append({
                        "term": item[0],
                        "final_weight": float(item[1]),
                        "similarity": float(item[2]),
                        "idf": float(item[3])
                    })

                for term, weight, _, _ in candidates[:c_exp_aspect]:
                    aspect_kw.append({"term": term, "weight": float(weight), "anchor_weight": 1.0, "is_anchor": False})
                    selected_synonyms.append(term)
                    total_synonyms_injected += 1
            else:
                if c_exp_aspect > 0:
                    starved_aspects += 1

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })
            aspect_traces.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "anchor_term": anchor,
                "is_heuristic_entity": is_heur,
                "anchor_idf": float(anchor_idf),
                "anchor_weight": float(r_anchor),
                "capacity_cap": c_exp_aspect,
                "total_candidates_above_tau": len(cands_above_tau),
                "candidates_above_tau": cands_above_tau,
                "injected_synonyms": selected_synonyms
            })

        aug_tokens = self._build_augmented_tokens(aspects)
        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": self._build_term_weights(aspects),
            "telemetry": {
                "num_anchors": len(anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "starved_aspects_count": starved_aspects,
                "avg_r_anchor": float(self.n_reps),
                "qaug_length": len(aug_tokens),
                "aspect_traces": aspect_traces
            }
        }

    def _extract_centrality_dynamic_inject(
        self,
        query: str,
        anchors: List[str],
        validated_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Schema 6b: Query Centrality Anchors + Fix B Validated Entity Boost +
                   Max Query IDF Dynamic Repetition + Zero-Floor Capacity C_exp = max(0, R + c).
        """
        dual_sim_matrix, _ = self._compute_dual_sim(anchors, query)
        aspects = []
        aspect_traces = []
        vocab_terms = self.vocab_matrix.vocab_terms
        max_corpus_idf = self.idf_registry.max_idf if self.idf_registry.max_idf > 0 else 1.0
        val_entity_set = set(h.lower() for h in validated_entities)

        anchor_idfs = [self.idf_registry.get_idf(a) for a in anchors]
        max_query_idf = max(anchor_idfs) if anchor_idfs and max(anchor_idfs) > 0 else max_corpus_idf

        total_cands_above_tau = 0
        total_synonyms_injected = 0
        starved_aspects = 0
        r_anchors = []

        for i, anchor in enumerate(anchors):
            is_heur = anchor.lower() in val_entity_set
            anchor_idf = self.idf_registry.get_idf(anchor)

            # Fix B: Only validated entities get r_max
            if is_heur:
                r_anchor = self.r_max
            else:
                scaled = self.r_min + (self.r_max - self.r_min) * (anchor_idf / max_query_idf)
                r_anchor = int(np.clip(np.round(scaled), self.r_min, self.r_max))

            r_anchors.append(r_anchor)
            aspect_kw = [{"term": anchor, "weight": 1.0, "anchor_weight": float(r_anchor), "is_anchor": True}]

            # Dynamic capacity with Zero-Floor (Fix B)
            c_exp_aspect = int(np.clip(r_anchor + self.c, 0, 5))
            cands_above_tau = []
            selected_synonyms = []

            if c_exp_aspect > 0 and dual_sim_matrix.numel() > 0:
                sims = dual_sim_matrix[i]
                valid_mask = sims >= self.tau_sim
                valid_indices = torch.where(valid_mask)[0]

                candidates = []
                for idx in valid_indices:
                    term = vocab_terms[idx.item()]
                    if term.lower() == anchor.lower():
                        continue
                    sim_val = sims[idx].item()
                    term_idf = self.idf_registry.get_idf(term)
                    final_weight = sim_val * (0.5 + 0.5 * (term_idf / max_corpus_idf))
                    candidates.append((term, min(0.95, final_weight), sim_val, term_idf))

                total_cands_above_tau += len(candidates)
                if len(candidates) < c_exp_aspect:
                    starved_aspects += 1

                candidates.sort(key=lambda x: x[1], reverse=True)
                for item in candidates[:10]:
                    cands_above_tau.append({
                        "term": item[0],
                        "final_weight": float(item[1]),
                        "similarity": float(item[2]),
                        "idf": float(item[3])
                    })

                for term, weight, _, _ in candidates[:c_exp_aspect]:
                    aspect_kw.append({"term": term, "weight": float(weight), "anchor_weight": 1.0, "is_anchor": False})
                    selected_synonyms.append(term)
                    total_synonyms_injected += 1
            else:
                if c_exp_aspect > 0:
                    starved_aspects += 1

            aspects.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "keywords": aspect_kw
            })
            aspect_traces.append({
                "aspect_id": f"asp_{i}_{anchor}",
                "anchor_term": anchor,
                "is_heuristic_entity": is_heur,
                "anchor_idf": float(anchor_idf),
                "anchor_weight": float(r_anchor),
                "capacity_cap": c_exp_aspect,
                "total_candidates_above_tau": len(cands_above_tau),
                "candidates_above_tau": cands_above_tau,
                "injected_synonyms": selected_synonyms
            })

        aug_tokens = self._build_augmented_tokens(aspects)
        avg_r = float(np.mean(r_anchors)) if r_anchors else 3.0
        return {
            "aspects": aspects,
            "augmented_token_list": aug_tokens,
            "term_weights": self._build_term_weights(aspects),
            "telemetry": {
                "num_anchors": len(anchors),
                "total_candidates_above_tau": total_cands_above_tau,
                "total_synonyms_injected": total_synonyms_injected,
                "starved_aspects_count": starved_aspects,
                "avg_r_anchor": avg_r,
                "qaug_length": len(aug_tokens),
                "aspect_traces": aspect_traces
            }
        }
