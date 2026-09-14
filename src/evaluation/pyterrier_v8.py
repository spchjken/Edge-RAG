"""
src/evaluation/pyterrier_v8.py

V8 Retrieval Engine: Entropy-Constrained Dense-Lexical Projection
Implements the formal architecture defined in docs/v8_plan.md:
- Empirical Bounded Frequency Filtering for vocabulary pool (no quadratic salience bell curve)
- Zero POS tagging dependency (pure information-theoretic and orthographic classification)
- Parametric & technical ID freezing (exact match on numbers, versions, compound punctuation)
- Anchor Information Specificity Gating (S(t) = IDF(t) / IDF_max >= 0.65)
- Directional Asymmetric Specificity Gating (Delta_IDF(e, t) = IDF(e) / IDF(t) >= 0.85)
- Anchor-level and synonym-level damping (top-1 anchor, top-1 synonym, max 30% weight)
"""

import os
import math
import time
import re
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
import torch
import pyterrier as pt

from src.evaluation.pyterrier_qe import (
    BGEVocabSidecarManager,
    TerrierQueryAnalyzer,
    get_terrier_analyzer,
    DEFAULT_QE_CONFIG,
)


def is_frozen_technical_id(surface: str) -> bool:
    """
    Returns True if the surface token is a parametric ID, measurement,
    version string, or compound unit that MUST NOT be expanded.
    """
    if not surface:
        return True
    # 1. Any token containing digits (e.g., '5mmol', 'nav2', 'qwen2.5', 'p53', '10mg')
    if any(c.isdigit() for c in surface):
        return True
    # 2. Short fragments or single chars (e.g., 'c', 'x', 'v1')
    if len(surface) <= 2:
        return True
    # 3. Punctuation compounds that survived tokenization ('nav2_bringup', '5mmol/l')
    if any(p in surface for p in ('-', '_', '/', '.')):
        return True
    return False


def is_acronym(surface: str) -> bool:
    """Returns True if the token is an all-caps acronym of length 2-5."""
    return surface.isupper() and 2 <= len(surface) <= 5 and surface.isalpha()


class V8VocabPool:
    """
    Constructs and manages the Empirical Bounded Frequency Vocabulary Pool.
    Eliminates the flawed Salience bell curve and retains all informative domain terms.
    """
    def __init__(
        self,
        dataset_name: str,
        index,
        sidecar_mgr: Optional[BGEVocabSidecarManager] = None,
        max_cap: int = 25000,
        device: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.index = index
        self.max_cap = max_cap
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sidecar_mgr = sidecar_mgr or BGEVocabSidecarManager(DEFAULT_QE_CONFIG["bge_vocab_qe"])

        # Load sidecar (15,000+ terms)
        self.sidecar = self.sidecar_mgr.build_or_load_sidecar(dataset_name, index)
        raw_terms = self.sidecar["retrieval_terms"]
        raw_surfaces = self.sidecar["display_surfaces"]
        raw_df = np.array(self.sidecar["df"], dtype=np.int32)
        raw_cf = np.array(self.sidecar["cf"], dtype=np.int32)
        raw_embeddings = self.sidecar["embeddings"]  # [V, 384]

        # Collection statistics & Lucene IDF
        self.num_docs = int(index.getCollectionStatistics().getNumberOfDocuments())
        self.lex = index.getLexicon()

        # Step 1: Filter candidate terms by Empirical Bounded Frequency:
        # Noise floor: CF >= 3, DF >= 2, len >= 2, not pure digits
        # Stopword ceiling: DF / num_docs <= 0.12
        n_terms = len(raw_terms)
        valid_mask = np.zeros(n_terms, dtype=bool)

        for i in range(n_terms):
            df_i = int(raw_df[i])
            cf_i = int(raw_cf[i])
            term_i = raw_terms[i]

            if (
                df_i >= 2
                and (df_i / max(self.num_docs, 1)) <= 0.12
                and cf_i >= 3
                and len(term_i) >= 2
                and not term_i.isdigit()
            ):
                valid_mask[i] = True

        valid_indices = np.where(valid_mask)[0]

        # Step 2: Capacity envelope: up to max_cap
        if len(valid_indices) > self.max_cap:
            # Sort descending by CF within bounds
            valid_cfs = raw_cf[valid_indices]
            sub_sort = np.argsort(-valid_cfs)[:self.max_cap]
            chosen_indices = valid_indices[sub_sort]
        else:
            chosen_indices = valid_indices

        self.pool_terms: List[str] = [raw_terms[idx] for idx in chosen_indices]
        self.pool_surfaces: List[str] = [raw_surfaces[idx] for idx in chosen_indices]
        self.pool_df: np.ndarray = raw_df[chosen_indices]

        # Calculate Lucene IDFs for all pool terms
        self.pool_idfs_np = np.array([
            float(math.log(1.0 + (self.num_docs - int(df_val) + 0.5) / (int(df_val) + 0.5)))
            for df_val in self.pool_df
        ], dtype=np.float32)

        # Slice GPU/CPU FP16 tensor
        pool_vecs_np = raw_embeddings[chosen_indices]
        self.gpu_pool = torch.tensor(pool_vecs_np, device=self.device, dtype=torch.float16)
        self.gpu_pool = torch.nn.functional.normalize(self.gpu_pool, p=2, dim=-1)

        # Hash map lookups
        self.term_to_pool_idx: Dict[str, int] = {t: i for i, t in enumerate(self.pool_terms)}
        self.surf_to_pool_idx: Dict[str, int] = {s.lower(): i for i, s in enumerate(self.pool_surfaces)}

    def get_term_idf(self, term: str) -> float:
        """Returns non-negative Lucene IDF for any term from Terrier Lexicon in O(1)."""
        lex_entry = self.lex.getLexiconEntry(term)
        df_val = int(lex_entry.getDocumentFrequency()) if lex_entry else 1
        return float(math.log(1.0 + (self.num_docs - df_val + 0.5) / (df_val + 0.5)))

    def get_term_df(self, term: str) -> int:
        """Returns document frequency for any term from Terrier Lexicon in O(1)."""
        lex_entry = self.lex.getLexiconEntry(term)
        return int(lex_entry.getDocumentFrequency()) if lex_entry else 1


class V8PyTerrierRewriter(pt.Transformer):
    """
    PyTerrier Transformer implementing V8 Entropy-Constrained Dense-Lexical Projection:
    1. Analyzes query using native Terrier tokenizer and stemmer.
    2. Zero POS tagger dependency: classifies via Orthography and Information Specificity.
    3. Strictly freezes parametric IDs, measurements, and numerical tokens.
    4. Evaluates Anchor Information Specificity S(t) = IDF(t) / IDF_max >= tau_spec.
    5. Applies Asymmetric Specificity Gating Delta_IDF(e, t) >= gamma_asym.
    6. Strict damping: top-1 anchor (top-2 if |Q| >= 6), top-1 synonym, max 30% mass.
    7. Emits query_toks dictionary for downstream Terrier MatchOp scoring.
    """
    def __init__(
        self,
        vocab_pool: V8VocabPool,
        tau_sim: float = 0.65,
        tau_spec: float = 0.65,
        gamma_asym: float = 0.85,
        mu_ceil: float = 0.30,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        encoder: Optional[Any] = None,
        surface_cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.pool = vocab_pool
        self.tau_sim = float(tau_sim)
        self.tau_spec = float(tau_spec)
        self.gamma_asym = float(gamma_asym)
        self.mu_ceil = float(mu_ceil)
        self.device = vocab_pool.device
        self.encoder_name = encoder_name
        self.analyzer = get_terrier_analyzer()
        self._encoder = encoder
        self.surface_cache: Dict[str, torch.Tensor] = surface_cache if surface_cache is not None else {}

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name, device=self.device)
        return self._encoder

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = df.copy()
        new_query_toks = []
        chunk_queries = res_df["query"].tolist()

        # Pass 1: Analyze queries and identify unseen surface words across chunk
        analyzed_queries = []
        needed_surfaces = {}
        for q_text in chunk_queries:
            terms, surfs = self.analyzer.analyze(q_text)
            analyzed_queries.append((terms, surfs))
            for s in surfs:
                s_lower = s.lower()
                if (
                    s_lower not in self.pool.surf_to_pool_idx
                    and s_lower not in self.surface_cache
                    and s_lower not in needed_surfaces
                ):
                    needed_surfaces[s_lower] = s

        # Batch encode unseen surfaces on GPU/CPU
        unseen_keys = list(needed_surfaces.keys())
        if unseen_keys:
            encoder = self._get_encoder()
            unseen_vecs = encoder.encode(
                unseen_keys,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for i, k in enumerate(unseen_keys):
                self.surface_cache[k] = torch.tensor(
                    unseen_vecs[i], device=self.device, dtype=torch.float16
                )

        # Pass 2: Per-query V8 Anchor Classification & Entropy-Constrained Expansion
        for idx, (terms, surfs) in enumerate(analyzed_queries):
            q_text = chunk_queries[idx]
            if not terms or not surfs:
                new_query_toks.append({})
                continue

            orig_toks = self.analyzer.to_query_toks(terms)
            num_query_terms = len(orig_toks)

            # Map stem term -> primary surface, IDF, and DF
            term_info: Dict[str, Dict[str, Any]] = {}
            for t, s in zip(terms, surfs):
                if t not in term_info:
                    idf_t = self.pool.get_term_idf(t)
                    df_t = self.pool.get_term_df(t)
                    term_info[t] = {
                        "surface": s,
                        "surface_lower": s.lower(),
                        "idf": idf_t,
                        "df": df_t,
                    }

            # Find maximum IDF in the query
            idfs = [info["idf"] for info in term_info.values()]
            max_idf = max(idfs) if idfs else 1.0

            # Classify tokens: determine which anchors are eligible for expansion
            eligible_anchors: List[Tuple[str, float]] = []  # (term, specificity)

            for t, info in term_info.items():
                s = info["surface"]
                s_lower = info["surface_lower"]
                idf_t = info["idf"]
                df_t = info["df"]

                # Rule 1: Technical ID / Measurement / Numerical Compound -> STRICTLY FREEZE
                if is_frozen_technical_id(s):
                    continue

                # Rule 2: Acronym (all-caps, len 2-5) -> ALWAYS ELIGIBLE (S = 1.0)
                if is_acronym(s):
                    eligible_anchors.append((t, 1.0))
                    continue

                # Rule 3: Stopword / Domain background ceiling (DF > 12% of docs) -> FREEZE
                if (df_t / max(self.pool.num_docs, 1)) > 0.12:
                    continue

                # Rule 4: Relative Information Specificity S(t) = IDF(t) / IDF_max
                spec = idf_t / max(max_idf, 1e-6)
                if spec >= self.tau_spec:
                    eligible_anchors.append((t, spec))

            # If no eligible anchors, emit base query unchanged
            if not eligible_anchors:
                new_query_toks.append(orig_toks)
                continue

            # Anchor-level damping: sort descending by specificity
            eligible_anchors.sort(key=lambda x: -x[1])
            max_allowed_anchors = 2 if num_query_terms >= 6 else 1
            expanding_anchors = [a[0] for a in eligible_anchors[:max_allowed_anchors]]

            # Gather anchor embeddings
            anchor_vec_list = []
            for a in expanding_anchors:
                s_lower = term_info[a]["surface_lower"]
                if s_lower in self.pool.surf_to_pool_idx:
                    anchor_vec_list.append(self.pool.gpu_pool[self.pool.surf_to_pool_idx[s_lower]])
                elif s_lower in self.surface_cache:
                    anchor_vec_list.append(self.surface_cache[s_lower])
                else:
                    enc_v = self._get_encoder().encode([s_lower], normalize_embeddings=True, show_progress_bar=False)[0]
                    t_v = torch.tensor(enc_v, device=self.device, dtype=torch.float16)
                    self.surface_cache[s_lower] = t_v
                    anchor_vec_list.append(t_v)

            # High-speed GPU/CPU GEMM: [N_anchors, Pool_size] (<0.06 ms)
            a_tensor = torch.stack(anchor_vec_list)
            a_tensor = torch.nn.functional.normalize(a_tensor, p=2, dim=-1)
            sim_matrix = torch.matmul(a_tensor, self.pool.gpu_pool.T).cpu().numpy()

            final_toks = dict(orig_toks)

            # For each expanding anchor, evaluate candidates under Asymmetric Specificity
            for i, a in enumerate(expanding_anchors):
                a_idf = term_info[a]["idf"]
                a_sims = sim_matrix[i]

                # Mask out self
                if a in self.pool.term_to_pool_idx:
                    a_sims[self.pool.term_to_pool_idx[a]] = 0.0

                # Filter 1: Cosine similarity >= tau_sim (0.65)
                sim_mask = a_sims >= self.tau_sim

                if not np.any(sim_mask):
                    continue

                # Candidates passing similarity
                cand_indices = np.where(sim_mask)[0]
                cand_idfs = self.pool.pool_idfs_np[cand_indices]
                cand_dfs = self.pool.pool_df[cand_indices]

                # Filter 2: Asymmetric Specificity Ratio: IDF(e) / IDF(a) >= gamma_asym (0.85)
                # Filter 3: Anti-junk: DF(e) >= 3
                asym_mask = (cand_idfs >= (self.gamma_asym * a_idf)) & (cand_dfs >= 3)

                if not np.any(asym_mask):
                    continue

                valid_cand_indices = cand_indices[asym_mask]
                valid_sims = a_sims[valid_cand_indices]
                valid_idfs = cand_idfs[asym_mask]

                # Composite score: Cosine Similarity * Specificity Ratio
                scores = valid_sims * (valid_idfs / max(a_idf, 1e-6))
                best_idx_in_valid = int(np.argmax(scores))

                best_pool_idx = valid_cand_indices[best_idx_in_valid]
                best_term = self.pool.pool_terms[best_pool_idx]
                best_sim = float(valid_sims[best_idx_in_valid])
                best_asym = float(valid_idfs[best_idx_in_valid] / max(a_idf, 1e-6))

                # Weight Bounding: w_syn = min(w_anchor * cos * asym, mu_ceil * w_anchor)
                w_anchor = float(orig_toks.get(a, 1.0))
                syn_weight = min(w_anchor * best_sim * best_asym, self.mu_ceil * w_anchor)

                # Inject single top synonym
                final_toks[best_term] = float(final_toks.get(best_term, 0.0) + syn_weight)

            new_query_toks.append(final_toks)

        res_df["query_toks"] = new_query_toks
        return res_df
