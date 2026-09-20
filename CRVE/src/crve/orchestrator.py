"""
src/crve/orchestrator.py

CRVEOrchestrator: Canonical 1st-Stage Retrieval Orchestrator for Context-Reranked Vocabulary Expansion (CRVE).
Coordinates:
1. Indexer: EdgeRAGAnalyzer, CorpusIDFRegistry, CorpusVocabBuilder, DenseVocabMatrix.
2. Selection: Gate 1 Proposers (WholeQueryBGE, AnchorBGE, LexicalPPMI, RRFHybrid).
3. Retrieval: Formulates expanded query representations for PyTerrier 1st-stage execution.
"""

import os
import time
import yaml
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import numpy as np

from .indexer.analyzer import EdgeRAGAnalyzer
from .indexer.corpus_idf_registry import CorpusIDFRegistry
from .indexer.corpus_vocab_builder import CorpusVocabBuilder
from .indexer.dense_vocab_matrix import DenseVocabMatrix
from .selection.gate1_proposers import (
    BaseProposer,
    WholeQueryBGEProposer,
    AnchorBGEProposer,
    LexicalPPMIProposer,
    RRFHybridProposer,
    DEFAULT_RRF_K,
    DEFAULT_ANCHOR_SPEC_THRESHOLD,
    DEFAULT_ANCHOR_DF_CEILING,
    DEFAULT_PPMI_MIN_SUPPORT,
)


class CRVEOrchestrator:
    """
    CRVE 1st-Stage Retrieval Orchestrator.
    Connects Corpus Vocabulary Extraction -> Dense Vocab Matrix -> Gate 1 Selection -> PyTerrier Retrieval.
    """

    def __init__(
        self,
        config_path: str = "configs/crve.yaml",
        corpus: Optional[List[str]] = None,
        doc_freqs: Optional[Dict[str, int]] = None,
        num_docs: Optional[int] = None,
        proposer_override: Optional[str] = None,
    ):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

        vocab_cfg = self.cfg.get("vocabulary", {})
        dense_cfg = self.cfg.get("dense_matrix", {})
        gate1_cfg = self.cfg.get("gate1_selection", {})

        # 1. Linguistic Analyzer & IDF Registry
        self.analyzer = EdgeRAGAnalyzer()
        self.idf_registry = CorpusIDFRegistry(
            corpus=corpus,
            doc_freqs=doc_freqs,
            num_docs=num_docs,
            analyzer=self.analyzer
        )

        # 2. Corpus Vocabulary Builder
        self.pool_size = vocab_cfg.get("pool_size", 10000)
        self.vocab_builder = CorpusVocabBuilder(
            idf_registry=self.idf_registry,
            pool_size=self.pool_size,
            analyzer=self.analyzer
        )

        # 3. Dense Vocabulary Matrix
        self.dense_model_name = dense_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        self.use_gpu = dense_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
        self.vocab_matrix = DenseVocabMatrix(
            model_name=self.dense_model_name,
            use_gpu=self.use_gpu
        )

        # 4. Gate 1 Selection Configuration
        self.default_proposer_name = proposer_override or gate1_cfg.get("default_proposer", "rrf_core")
        self.deployable_budget_l = gate1_cfg.get("deployable_budget_l", 200)
        self.rrf_k = gate1_cfg.get("rrf_k", DEFAULT_RRF_K)
        self.channels_cfg = gate1_cfg.get("channels", {})

        self.active_proposer: Optional[BaseProposer] = None
        self.proposers: Dict[str, BaseProposer] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads YAML configuration file."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def build_vocabulary_and_matrix(
        self,
        corpus: Optional[List[str]] = None,
        strategy: str = "salience"
    ) -> None:
        """
        Builds the vocabulary candidate pool and computes the dense embedding matrix.
        """
        if corpus is not None and not self.idf_registry.doc_freqs:
            self.idf_registry = CorpusIDFRegistry(
                corpus=corpus,
                analyzer=self.analyzer
            )
            self.vocab_builder.idf_registry = self.idf_registry

        if corpus:
            candidate_stems, surface_forms = self.vocab_builder.extract_candidates_with_surface_forms(corpus)
            self.vocab_matrix.build_with_fps(
                candidate_stems,
                surface_forms=surface_forms,
                target_pool_size=min(self.pool_size, len(candidate_stems))
            )
        elif self.idf_registry.doc_freqs:
            pool_stems, full_stems, full_surfaces = self.vocab_builder.build_pool_with_full(
                strategy=strategy
            )
            self.vocab_matrix.build(
                vocab_stems=pool_stems,
                surface_forms=[self.vocab_builder.stem_to_surface.get(s, s) for s in pool_stems],
                full_stems=full_stems,
                full_surfaces=full_surfaces
            )

    def setup_gate1_proposers(
        self,
        pool_terms: List[str],
        pool_embeddings_tensor: torch.Tensor,
        encoder: Any,
        terrier_index: Optional[Any] = None,
        term_to_postings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes Gate 1 proposers using the precomputed pool embeddings and postings.
        """
        device = "cuda" if self.use_gpu else "cpu"

        # Whole-Query BGE Proposer
        wq_proposer = WholeQueryBGEProposer(
            pool_terms=pool_terms,
            pool_embeddings_tensor=pool_embeddings_tensor,
            encoder=encoder,
            device=device,
        )
        self.proposers["whole_query"] = wq_proposer

        # Anchor BGE Proposers (Filtered and All)
        abge_flt_cfg = self.channels_cfg.get("anchor_bge_filtered", {})
        anchor_proposer_flt = AnchorBGEProposer(
            pool_terms=pool_terms,
            pool_embeddings_tensor=pool_embeddings_tensor,
            encoder=encoder,
            max_df_ratio=abge_flt_cfg.get("max_df_ratio", DEFAULT_ANCHOR_DF_CEILING),
            min_spec=abge_flt_cfg.get("min_specificity", DEFAULT_ANCHOR_SPEC_THRESHOLD),
            filter_anchors=True,
            device=device,
        )
        self.proposers["anchor_bge_filtered"] = anchor_proposer_flt

        anchor_proposer_all = AnchorBGEProposer(
            pool_terms=pool_terms,
            pool_embeddings_tensor=pool_embeddings_tensor,
            encoder=encoder,
            filter_anchors=False,
            device=device,
        )
        self.proposers["anchor_bge_all"] = anchor_proposer_all

        # Lexical PPMI Proposer (if index or postings available)
        ppmi_cfg = self.channels_cfg.get("lexical_ppmi", {})
        if terrier_index is not None or term_to_postings is not None:
            ppmi_proposer = LexicalPPMIProposer(
                pool_terms=pool_terms,
                terrier_index=terrier_index,
                term_to_postings=term_to_postings,
                min_support=ppmi_cfg.get("min_joint_support", DEFAULT_PPMI_MIN_SUPPORT),
            )
            self.proposers["lexical_ppmi"] = ppmi_proposer

        # RRF Hybrid Proposers
        available_channels = [p for p in [wq_proposer, anchor_proposer_flt, self.proposers.get("lexical_ppmi")] if p is not None]
        self.proposers["rrf_core"] = RRFHybridProposer(
            proposers=available_channels,
            k=self.rrf_k,
        )

        all_channels = [p for p in [wq_proposer, anchor_proposer_flt, anchor_proposer_all, self.proposers.get("lexical_ppmi")] if p is not None]
        self.proposers["rrf_all"] = RRFHybridProposer(
            proposers=all_channels,
            k=self.rrf_k,
        )

        self.active_proposer = self.proposers.get(self.default_proposer_name, self.proposers.get("rrf_core"))

    def propose_expansion(
        self,
        query: str,
        query_id: str = "q0",
        budget: Optional[int] = None,
        proposer_name: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Executes Gate 1 candidate selection for a query.
        Returns sorted list of (term, score) tuples up to the specified budget.
        """
        k = budget if budget is not None else self.deployable_budget_l
        proposer = self.proposers.get(proposer_name) if proposer_name else self.active_proposer

        if proposer is None:
            raise RuntimeError("No active Gate 1 proposer initialized. Call setup_gate1_proposers first.")

        query_obj = {"qid": query_id, "query": query}
        return proposer.propose(query_obj, top_k=k)

    def build_expanded_query_string(
        self,
        query: str,
        expansion_terms: List[Tuple[str, float]],
        alpha: float = 0.60
    ) -> str:
        """
        Constructs a Terrier-compatible weighted query string:
        Original terms retain base weight (1.0), expansion terms receive weight alpha * normalized_score.
        """
        if not expansion_terms:
            return query

        max_score = max(score for _, score in expansion_terms) if expansion_terms else 1.0
        scale = alpha / max_score if max_score > 0 else alpha

        parts = [query]
        for term, score in expansion_terms:
            w = round(score * scale, 4)
            if w > 0.0001:
                parts.append(f"{term}^{w}")

        return " ".join(parts)
