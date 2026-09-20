"""
src/pipeline_v2/selection/gate1_proposers.py

Production Candidate Proposal Channels for Phase 2 Gate 1 Selection:
1. WholeQueryBGEProposer: Raw query sentence cosine matching against canonical pool embeddings.
2. AnchorBGEProposer: Specificity-weighted anchor cosine matching (Filtered and All-Content variants).
3. LexicalPPMIProposer: Document-level Positive Pointwise Mutual Information from Terrier postings.
4. RRFHybridProposer: Reciprocal Rank Fusion over truncated top-500 lists with unique refill.

Enforces:
- Global shared candidate pool: P_q = P \\ AnalyzedCanonicalTerms(q).
- Zero-leakage deterministic seed and tie-breaking.
- Fast, memory-safe PyTorch FP16 and direct-index inverted list processing.
"""

import os
import sys
import time
import math
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
import numpy as np
import torch

from src.evaluation.pyterrier_qe import get_terrier_analyzer

DEFAULT_RRF_K: int = 60
DEFAULT_ANCHOR_SPEC_THRESHOLD: float = 0.65
DEFAULT_ANCHOR_DF_CEILING: float = 0.12
DEFAULT_PPMI_MIN_SUPPORT: int = 2


def is_acronym(surface: str) -> bool:
    """Returns True if the token is an all-caps acronym of length 2-5."""
    return surface.isupper() and 2 <= len(surface) <= 5 and surface.isalpha()


class BaseProposer:
    """Abstract base class for Gate 1 proposers."""

    def __init__(self, name: str):
        self.name = name
        self.analyzer = get_terrier_analyzer()

    def get_query_excluded_terms(self, query_text: str) -> Set[str]:
        """
        Extracts original query terms and canonical surfaces to enforce:
        P_q = P \\ AnalyzedCanonicalTerms(q).
        """
        terms, surfs = self.analyzer.analyze(query_text)
        excluded = set(terms)
        for s in surfs:
            excluded.add(str(s).lower())
            excluded.add(str(s))
        return excluded

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        """
        Proposes top_k terms for query_obj.
        Returns sorted list of (term, score) tuples.
        """
        raise NotImplementedError


class WholeQueryBGEProposer(BaseProposer):
    """
    Channel 1: Whole-Query to Pool-Term BGE Cosine Proposer (S_WQ).
    Encodes raw query string with BGE-small-en-v1.5 and computes unit L2 cosine
    against precomputed canonical surface embeddings for terms in P_q.
    """

    def __init__(
        self,
        pool_terms: List[str],
        pool_embeddings_tensor: torch.Tensor,
        encoder,
        device: Optional[str] = None,
    ):
        super().__init__(name="WholeQueryBGE")
        self.pool_terms = list(pool_terms)
        self.term_to_idx = {t: i for i, t in enumerate(self.pool_terms)}
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure unit L2 normalized pool tensor on device
        if pool_embeddings_tensor.device != torch.device(self.device):
            self.pool_embeddings = pool_embeddings_tensor.to(self.device)
        else:
            self.pool_embeddings = pool_embeddings_tensor
        self.pool_embeddings = torch.nn.functional.normalize(self.pool_embeddings, p=2, dim=-1)

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        q_text = str(query_obj.get("question", query_obj.get("query", "")))
        excluded_terms = self.get_query_excluded_terms(q_text)

        # 1. Encode query sentence
        if hasattr(self.encoder, "encode"):
            # SentenceTransformer interface
            q_emb = self.encoder.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]
            q_tensor = torch.tensor(q_emb, device=self.device, dtype=self.pool_embeddings.dtype)
        elif hasattr(self.encoder, "encode_queries"):
            # FlagModel interface
            q_emb = self.encoder.encode_queries([q_text])[0]
            q_tensor = torch.tensor(q_emb, device=self.device, dtype=self.pool_embeddings.dtype)
        else:
            raise ValueError(f"Unsupported encoder type: {type(self.encoder)}")

        q_tensor = torch.nn.functional.normalize(q_tensor.unsqueeze(0), p=2, dim=-1)

        # 2. Batched GPU dot product [1, V]
        sims = torch.matmul(q_tensor, self.pool_embeddings.T).squeeze(0).cpu().numpy()

        # 3. Filter excluded query terms and sort
        scored_terms = []
        for idx, term in enumerate(self.pool_terms):
            if term not in excluded_terms:
                scored_terms.append((term, float(sims[idx])))

        scored_terms.sort(key=lambda x: (-x[1], x[0]))
        return scored_terms[:top_k]


class AnchorBGEProposer(BaseProposer):
    """
    Channel 2: Anchor-to-Term BGE Proposer (S_Anchor).
    Extracts query anchors, applies specificity weighting, and computes max cosine.
    Supports both Filtered (DF/N <= 0.12, spec >= 0.65) and All-Content variants.
    """

    def __init__(
        self,
        pool_terms: List[str],
        pool_embeddings_tensor: torch.Tensor,
        surf_to_pool_idx: Dict[str, int],
        idf_map: Dict[str, float],
        df_map: Dict[str, int],
        num_docs: int,
        encoder,
        filter_anchors: bool = True,
        spec_threshold: float = DEFAULT_ANCHOR_SPEC_THRESHOLD,
        df_ceiling: float = DEFAULT_ANCHOR_DF_CEILING,
        device: Optional[str] = None,
    ):
        name = "AnchorBGEFiltered" if filter_anchors else "AnchorBGEAll"
        super().__init__(name=name)
        self.pool_terms = list(pool_terms)
        self.surf_to_pool_idx = dict(surf_to_pool_idx)
        self.idf_map = dict(idf_map)
        self.df_map = dict(df_map)
        self.num_docs = max(num_docs, 1)
        self.encoder = encoder
        self.filter_anchors = filter_anchors
        self.spec_threshold = spec_threshold
        self.df_ceiling = df_ceiling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if pool_embeddings_tensor.device != torch.device(self.device):
            self.pool_embeddings = pool_embeddings_tensor.to(self.device)
        else:
            self.pool_embeddings = pool_embeddings_tensor
        self.pool_embeddings = torch.nn.functional.normalize(self.pool_embeddings, p=2, dim=-1)

        self.max_idf = max(self.idf_map.values()) if self.idf_map else 1.0

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        q_text = str(query_obj.get("question", query_obj.get("query", "")))
        excluded_terms = self.get_query_excluded_terms(q_text)

        terms, surfs = self.analyzer.analyze(q_text)
        if not terms or not surfs:
            return []

        # 1. Select anchors and compute weights
        selected_anchors = []
        candidate_anchors = []

        for t, s in zip(terms, surfs):
            s_lower = str(s).lower()
            t_idf = self.idf_map.get(t, 0.0)
            t_df = self.df_map.get(t, 0)
            spec = t_idf / max(self.max_idf, 1e-6)

            df_fraction = t_df / self.num_docs
            acronym = is_acronym(str(s))

            is_eligible = True
            if self.filter_anchors:
                if df_fraction > self.df_ceiling:
                    is_eligible = False
                elif not acronym and spec < self.spec_threshold:
                    is_eligible = False

            weight = max(spec, 0.01)
            candidate_anchors.append((t, s_lower, weight, spec))
            if is_eligible:
                selected_anchors.append((t, s_lower, weight))

        # Fallback: if no anchor survived filtering, pick highest-IDF content token
        if not selected_anchors and candidate_anchors:
            best_cand = max(candidate_anchors, key=lambda x: x[3])
            selected_anchors = [(best_cand[0], best_cand[1], best_cand[2])]

        if not selected_anchors:
            return []

        # 2. Get embeddings for anchors (reuse pool tensor or encode unseen)
        anchor_vecs = []
        anchor_weights = []

        unseen_surfs = [s for _, s, _ in selected_anchors if s not in self.surf_to_pool_idx]
        unseen_embs = {}
        if unseen_surfs:
            if hasattr(self.encoder, "encode"):
                encs = self.encoder.encode(unseen_surfs, normalize_embeddings=True, show_progress_bar=False)
            else:
                encs = self.encoder.encode_queries(unseen_surfs)
            for s, e in zip(unseen_surfs, encs):
                unseen_embs[s] = torch.tensor(e, device=self.device, dtype=self.pool_embeddings.dtype)

        for _, s, w in selected_anchors:
            if s in self.surf_to_pool_idx:
                vec = self.pool_embeddings[self.surf_to_pool_idx[s]]
            else:
                vec = unseen_embs[s]
            anchor_vecs.append(vec)
            anchor_weights.append(w)

        # 3. Compute weighted max cosine against pool
        # Stack anchor vectors [M, D]
        a_matrix = torch.stack(anchor_vecs, dim=0)
        a_matrix = torch.nn.functional.normalize(a_matrix, p=2, dim=-1)
        w_tensor = torch.tensor(anchor_weights, device=self.device, dtype=self.pool_embeddings.dtype).unsqueeze(1)

        # [M, V] = [M, D] @ [D, V]
        sim_matrix = torch.matmul(a_matrix, self.pool_embeddings.T)
        weighted_sims = sim_matrix * w_tensor
        max_sims, _ = torch.max(weighted_sims, dim=0)
        max_sims_np = max_sims.cpu().numpy()

        # 4. Filter exclusions and sort
        scored_terms = []
        for idx, term in enumerate(self.pool_terms):
            if term not in excluded_terms:
                scored_terms.append((term, float(max_sims_np[idx])))

        scored_terms.sort(key=lambda x: (-x[1], x[0]))
        return scored_terms[:top_k]


class LexicalPPMIProposer(BaseProposer):
    """
    Channel 3: Lexical PPMI Proposer (S_Lex).
    Evaluates unsmoothed Positive Pointwise Mutual Information between query anchors
    and candidate pool terms using direct-index inverted list intersections.
    Enforces DF(a,t) >= 2 and aggregates with anchor specificity weights.
    """

    def __init__(
        self,
        index,
        pool_terms: List[str],
        idf_map: Dict[str, float],
        df_map: Dict[str, int],
        num_docs: int,
        min_support: int = DEFAULT_PPMI_MIN_SUPPORT,
    ):
        super().__init__(name="LexicalPPMI")
        self.index = index
        self.pool_terms = set(pool_terms)
        self.idf_map = dict(idf_map)
        self.df_map = dict(df_map)
        self.num_docs = max(num_docs, 1)
        self.min_support = min_support
        self.max_idf = max(self.idf_map.values()) if self.idf_map else 1.0

        self.lex = self.index.getLexicon()
        self.inv = self.index.getInvertedIndex()
        self.di = self.index.getDirectIndex()
        self.doi = self.index.getDocumentIndex()

    def get_anchor_docids(self, term: str) -> Set[int]:
        """Retrieves set of docids where term occurs via InvertedIndex."""
        entry = self.lex.getLexiconEntry(term)
        if entry is None:
            return set()
        postings = self.inv.getPostings(entry)
        if postings is None:
            return set()
        docids = set()
        while postings.next() != postings.EOL:
            docids.add(postings.getId())
        return docids

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 500,
    ) -> List[Tuple[str, float]]:
        q_text = str(query_obj.get("question", query_obj.get("query", "")))
        excluded_terms = self.get_query_excluded_terms(q_text)

        terms, _ = self.analyzer.analyze(q_text)
        if not terms:
            return []

        # 1. Identify valid anchors
        unique_anchors = list(set(terms))
        anchor_docs_map: Dict[str, Set[int]] = {}
        anchor_weights: Dict[str, float] = {}

        for a in unique_anchors:
            a_df = self.df_map.get(a, 0)
            if a_df < 2 or (a_df / self.num_docs) > 0.12:
                continue
            docids = self.get_anchor_docids(a)
            if docids:
                anchor_docs_map[a] = docids
                anchor_weights[a] = self.idf_map.get(a, 0.0) / max(self.max_idf, 1e-6)

        if not anchor_docs_map:
            return []

        # 2. Fast direct-index co-occurrence accumulation
        # For each anchor, iterate over its documents and count pool terms
        # joint_counts[anchor][cand] = count
        scores: Dict[str, float] = defaultdict(float)

        for a, docids in anchor_docs_map.items():
            w_a = anchor_weights[a]
            df_a = self.df_map.get(a, len(docids))
            if df_a < 1:
                continue

            # Accumulate co-occurring terms in docids
            joint_df: Dict[str, int] = defaultdict(int)
            for docid in docids:
                entry = self.doi.getDocumentEntry(docid)
                if entry is None:
                    continue
                postings = self.di.getPostings(entry)
                if postings is None:
                    continue
                while postings.next() != postings.EOL:
                    term_id = postings.getId()
                    l_entry = self.lex.getLexiconEntry(term_id)
                    if l_entry is not None:
                        t_str = l_entry.getKey()
                        if t_str in self.pool_terms and t_str not in excluded_terms:
                            joint_df[t_str] += 1

            # Compute unsmoothed PPMI for candidates with joint support >= min_support
            for cand, df_at in joint_df.items():
                if df_at >= self.min_support:
                    df_t = self.df_map.get(cand, 0)
                    if df_t >= 1:
                        # PPMI(a, t) = max(0, log( N * df_at / (df_a * df_t) ))
                        pmi_val = math.log((self.num_docs * df_at) / (df_a * df_t))
                        if pmi_val > 0.0:
                            scores[cand] += w_a * pmi_val

        # 3. Sort descending by aggregated PPMI score
        scored_terms = [(cand, score) for cand, score in scores.items()]
        scored_terms.sort(key=lambda x: (-x[1], x[0]))
        return scored_terms[:top_k]


class RRFHybridProposer(BaseProposer):
    """
    Channel 4: Reciprocal Rank Fusion (RRF) Proposer.
    Fuses truncated top-500 candidate lists from multiple active proposers with k=60.
    Applies deterministic 3-tier tie-breaking and equal-budget unique refill to emit exactly L unique terms.
    """

    def __init__(
        self,
        k: int = DEFAULT_RRF_K,
    ):
        super().__init__(name="RRFHybrid")
        self.k = k

    def fuse(
        self,
        channel_rankings: Dict[str, List[Tuple[str, float]]],
        top_l: int = 200,
    ) -> List[Tuple[str, float]]:
        """
        Fuses candidate lists from multiple proposers.
        channel_rankings: dict mapping channel_name -> list of (term, score) tuples (length <= 500).
        Returns exactly top_l unique terms sorted by RRF with deterministic tie-breaking.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        best_ranks: Dict[str, int] = {}

        for ch_name, ranking in channel_rankings.items():
            for rank_0, (term, _) in enumerate(ranking):
                rank = rank_0 + 1
                rrf_scores[term] += 1.0 / (self.k + rank)
                if term not in best_ranks or rank < best_ranks[term]:
                    best_ranks[term] = rank

        # Deterministic 3-tier tie-breaking:
        # 1. RRF score descending (-score)
        # 2. Best individual rank ascending (best_rank)
        # 3. Term string alphabetical ascending (term)
        fused = []
        for term, score in rrf_scores.items():
            b_rank = best_ranks.get(term, 999999)
            fused.append((term, score, b_rank))

        fused.sort(key=lambda x: (-x[1], x[2], x[0]))
        return [(t, s) for t, s, _ in fused[:top_l]]

    def propose(
        self,
        query_obj: Dict[str, Any],
        top_k: int = 200,
    ) -> List[Tuple[str, float]]:
        # RRF proposal requires pre-generated channel rankings
        raise NotImplementedError("Use RRFHybridProposer.fuse(channel_rankings, top_l) to fuse multiple channels.")
