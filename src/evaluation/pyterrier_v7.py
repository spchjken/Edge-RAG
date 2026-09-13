"""
src/evaluation/pyterrier_v7.py

V7 Anchored Lexical-Semantic Retriever for PyTerrier:
1. Top-1,000 Salience Vocabulary Pool (IDF * ln(1 + DF))
2. Linguistic Content Anchoring with Penn Treebank POS Priors (Noun 1.0, Verb 0.75, Mod 0.60)
3. 1-Pass CUDA GEMM Semantic Probing (<0.02 ms) against Salience Pool
4. Strict Similarity Gate (tau_base in [0.55, 0.75]) with graceful starvation
5. Information-Theoretic Mass-Preserving Expansion (IT-MPE) with:
   - mu_ceil = 1/3 (guaranteeing >= 75% anchor mass retention)
   - Score-space damping: min(1.0, IDF_a / IDF_s)
   - Collision summing
6. Native query_toks emission for downstream Terrier BM25/DPH scoring.
"""

import os
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import torch
import pyterrier as pt

from src.evaluation.pyterrier_qe import (
    BGEVocabSidecarManager,
    get_terrier_analyzer,
    DEFAULT_QE_CONFIG,
)
from src.pipeline_v2.expansion.v7_aspect_extractor import POSTaggerHelper


class V7SaliencePool:
    """
    Constructs and manages the top-1,000 salience vocabulary pool for a dataset:
    Salience(t) = IDF(t) * ln(1 + DF(t))
    """
    def __init__(
        self,
        dataset_name: str,
        index,
        sidecar_mgr: Optional[BGEVocabSidecarManager] = None,
        pool_size: int = 1000,
        device: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.index = index
        self.pool_size = pool_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sidecar_mgr = sidecar_mgr or BGEVocabSidecarManager(DEFAULT_QE_CONFIG["bge_vocab_qe"])
        
        # Load sidecar
        self.sidecar = self.sidecar_mgr.build_or_load_sidecar(dataset_name, index)
        self.retrieval_terms = self.sidecar["retrieval_terms"]
        self.display_surfaces = self.sidecar["display_surfaces"]
        self.df_vals = self.sidecar["df"]
        self.raw_embeddings = self.sidecar["embeddings"]  # [V, 384]
        
        # Calculate collection stats & Lucene IDF
        self.num_docs = int(index.getCollectionStatistics().getNumberOfDocuments())
        self.lex = index.getLexicon()
        
        # Compute salience for all terms
        n_terms = len(self.retrieval_terms)
        salience_scores = np.zeros(n_terms, dtype=np.float32)
        idfs = np.zeros(n_terms, dtype=np.float32)
        
        for i in range(n_terms):
            df_i = int(self.df_vals[i])
            idf_i = float(math.log(1.0 + (self.num_docs - df_i + 0.5) / (df_i + 0.5)))
            idfs[i] = idf_i
            salience_scores[i] = idf_i * float(math.log(1.0 + df_i))
            
        # Select top pool_size by salience
        sorted_indices = np.argsort(-salience_scores)
        top_k = min(self.pool_size, n_terms)
        pool_indices = sorted_indices[:top_k]
        
        self.pool_terms = [self.retrieval_terms[idx] for idx in pool_indices]
        self.pool_surfaces = [self.display_surfaces[idx] for idx in pool_indices]
        self.pool_idfs_np = idfs[pool_indices]
        
        # Slice GPU tensor: [pool_size, 384] FP16
        pool_vecs_np = self.raw_embeddings[pool_indices]
        self.gpu_pool = torch.tensor(pool_vecs_np, device=self.device, dtype=torch.float16)
        self.gpu_pool = torch.nn.functional.normalize(self.gpu_pool, p=2, dim=-1)
        
        # Lookup tables
        self.term_to_pool_idx = {t: i for i, t in enumerate(self.pool_terms)}
        self.surf_to_pool_idx = {s.lower(): i for i, s in enumerate(self.pool_surfaces)}

    def get_term_idf(self, term: str) -> float:
        """Returns non-negative Lucene IDF for any term from Terrier Lexicon."""
        lex_entry = self.lex.getLexiconEntry(term)
        df_val = int(lex_entry.getDocumentFrequency()) if lex_entry else 1
        return float(math.log(1.0 + (self.num_docs - df_val + 0.5) / (df_val + 0.5)))


class V7PyTerrierRewriter(pt.Transformer):
    """
    PyTerrier Transformer implementing the V7 Anchored Retriever:
    - Analyzes query with Terrier's native tokenizer & stemmer
    - Content anchoring & POS priors
    - Fast CUDA GEMM against 1,000 Salience Pool
    - Hard similarity threshold tau_base in [0.55, 0.75]
    - IT-MPE mass allocation (mu_ceil = 1/3) with score-space damping
    - Emits query_toks dictionary for downstream Terrier MatchOp scoring.
    """
    def __init__(
        self,
        salience_pool: V7SaliencePool,
        tau_base: float = 0.55,
        mu_ceil: float = 1.0 / 3.0,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        encoder: Optional[Any] = None,
        surface_cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.pool = salience_pool
        self.tau_base = float(tau_base)
        self.mu_ceil = float(mu_ceil)
        self.device = salience_pool.device
        self.encoder_name = encoder_name
        self.pos_tagger = POSTaggerHelper()
        self.pos_ratios = {"noun": 1.0, "verb": 0.75, "modifier": 0.60}
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
        
        # 1. First pass: analyze queries and identify unseen surface words across chunk
        analyzed_queries = []
        needed_surfaces = {}
        for q_text in chunk_queries:
            terms, surfs = self.analyzer.analyze(q_text)
            analyzed_queries.append((terms, surfs))
            for s in surfs:
                s_lower = s.lower()
                if (s_lower not in self.pool.surf_to_pool_idx and 
                    s_lower not in self.surface_cache and 
                    s_lower not in needed_surfaces):
                    needed_surfaces[s_lower] = s

        # Batch encode unseen surfaces on GPU
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

        # 2. Second pass: per-query V7 anchoring, GEMM probing, gating, and IT-MPE compilation
        for idx, (terms, surfs) in enumerate(analyzed_queries):
            q_text = chunk_queries[idx]
            if not terms or not surfs:
                new_query_toks.append({})
                continue

            orig_toks = self.analyzer.to_query_toks(terms)
            pos_map = self.pos_tagger.tag_query(q_text, set())

            # Distinct anchors & base weights
            anchor_surfs: Dict[str, str] = {}
            anchor_weights: Dict[str, float] = {}
            anchor_idfs: Dict[str, float] = {}

            for t, s in zip(terms, surfs):
                s_lower = s.lower()
                if t not in anchor_surfs:
                    anchor_surfs[t] = s_lower
                    pos_cat = pos_map.get(s_lower, "noun")
                    anchor_weights[t] = self.pos_ratios.get(pos_cat, 0.75)
                    anchor_idfs[t] = self.pool.get_term_idf(t)

            anchors = list(orig_toks.keys())
            if not anchors:
                new_query_toks.append(orig_toks)
                continue

            # Gather anchor vectors
            anchor_vec_list = []
            for a in anchors:
                s_lower = anchor_surfs[a]
                if s_lower in self.pool.surf_to_pool_idx:
                    anchor_vec_list.append(self.pool.gpu_pool[self.pool.surf_to_pool_idx[s_lower]])
                elif s_lower in self.surface_cache:
                    anchor_vec_list.append(self.surface_cache[s_lower])
                else:
                    # Fallback single embed (rare)
                    enc_v = self._get_encoder().encode([s_lower], normalize_embeddings=True, show_progress_bar=False)[0]
                    t_v = torch.tensor(enc_v, device=self.device, dtype=torch.float16)
                    self.surface_cache[s_lower] = t_v
                    anchor_vec_list.append(t_v)

            # Stack anchor vectors: [N_anchors, 384]
            a_tensor = torch.stack(anchor_vec_list)
            a_tensor = torch.nn.functional.normalize(a_tensor, p=2, dim=-1)

            # High-speed GPU Matrix Multiplication: [N_anchors, 1000] (<0.02 ms)
            sim_matrix = torch.matmul(a_tensor, self.pool.gpu_pool.T).cpu().numpy()

            # IT-MPE Precomputations
            anchor_w_np = np.array([anchor_weights[a] for a in anchors], dtype=np.float32)
            anchor_idf_np = np.array([anchor_idfs[a] for a in anchors], dtype=np.float32)
            mu_q = self.mu_ceil

            # Hard Similarity Gating: Sim(a, s) >= tau_base
            mask = (sim_matrix >= self.tau_base)

            # Self-anchor column exclusion (anchor cannot expand to itself)
            for i, a in enumerate(anchors):
                if a in self.pool.term_to_pool_idx:
                    mask[i, self.pool.term_to_pool_idx[a]] = False

            masked_sims = np.where(mask, sim_matrix, 0.0)
            sum_sims = np.sum(masked_sims, axis=1, keepdims=True)
            p_cond_mat = masked_sims / np.where(sum_sims == 0, 1.0, sum_sims)

            # Score-space damping: min(1.0, IDF_a / IDF_s)
            damping_mat = np.minimum(
                1.0,
                anchor_idf_np[:, None] / np.maximum(self.pool.pool_idfs_np[None, :], 1e-6)
            )

            # Final injected weight matrix: W[i, j] = w_a * damping * mu_q * p_cond
            w_syn_mat = anchor_w_np[:, None] * damping_mat * (mu_q * p_cond_mat)

            # Column-wise collision sum across anchors
            syn_weights = np.sum(w_syn_mat, axis=0)
            active_v_indices = np.where(syn_weights > 0)[0]

            # Compile into final_toks
            final_toks = dict(orig_toks)
            for v_idx in active_v_indices:
                cand_term = self.pool.pool_terms[v_idx]
                final_toks[cand_term] = float(final_toks.get(cand_term, 0.0) + syn_weights[v_idx])

            new_query_toks.append(final_toks)

        res_df["query_toks"] = new_query_toks
        return res_df
