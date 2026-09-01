import time
import yaml
from typing import List, Dict, Any, Optional

from .indexer.corpus_idf_registry import CorpusIDFRegistry
from .indexer.bm25_lucene_indexer import BM25LuceneIndexer
from .indexer.corpus_vocab_builder import CorpusVocabBuilder
from .indexer.dense_vocab_matrix import DenseVocabMatrix
from .expansion.bm25_dense_aspect_extractor import BM25DenseAspectExtractor
from .routing.bm25_cascade_router import BM25CascadeRouter
from .reranker.listwise_reranker import ListwiseLLMRerankerV2
from .expansion_late.late_expansion import LateExpansionV2
from src.utils.llm_client import LLMClient


class PipelineV2Orchestrator:
    """
    End-to-End Execution Orchestrator for Pipeline V2.
    Integrates Indexer -> Aspect Expansion -> Cascade Routing -> Listwise Reranker -> Late Expansion.
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        config_path: str = "configs/pipeline_v2.yaml",
        llm_client: Optional[LLMClient] = None
    ):
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [f"c_{i}" for i in range(len(corpus))]
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f).get("pipeline_v2", {})
        except Exception:
            cfg = {}

        model_name = cfg.get("model_name", "qwen3.5-2b")
        self.llm_client = llm_client if llm_client is not None else LLMClient(model_name=model_name)

        schema = cfg.get("schema", "BM25Dense_V7")
        idx_cfg = cfg.get("indexer", {})
        exp_cfg = cfg.get("expansion", {})
        route_cfg = cfg.get("routing", {})
        vram_cfg = cfg.get("vram", {})

        # Phase 1: High-Speed Indexing & Shared IDF Matrix Build (V7 Parity & Coverage Pool)
        t0 = time.time()
        from .indexer.analyzer import EdgeRAGAnalyzer
        self.analyzer = EdgeRAGAnalyzer(
            stemmer=idx_cfg.get("stemmer", "kstem"),
            use_wordnet_override=idx_cfg.get("use_wordnet_override", True)
        )
        self.indexer = BM25LuceneIndexer(
            corpus,
            chunk_ids=self.chunk_ids,
            mode=idx_cfg.get("mode", "parity"),
            analyzer=self.analyzer
        )
        self.idf_registry = self.indexer.idf_registry

        pool_size = idx_cfg.get("max_vocab_pool_size", 2500)
        vocab_builder = CorpusVocabBuilder(
            self.idf_registry,
            max_vocab_size=pool_size,
            analyzer=self.analyzer
        )
        vocab_selection = idx_cfg.get("vocab_selection", "coverage")

        self.vocab_matrix = DenseVocabMatrix(model_name=exp_cfg.get("bge_model_name", "BAAI/bge-small-en-v1.5"))

        if vocab_selection == "coverage":
            candidate_stems, surface_forms = vocab_builder.extract_candidates_with_surface_forms(corpus)
            self.vocab_matrix.build_with_fps(candidate_stems, surface_forms=surface_forms, target_pool_size=pool_size)
        else:
            pool_stems, full_stems, full_surfaces = vocab_builder.build_pool_with_full(corpus, strategy=vocab_selection)
            self.vocab_matrix.build_matrix(pool_stems, full_stems=full_stems, full_surfaces=full_surfaces)

        self.tti_seconds = time.time() - t0

        # Phase 2: Query Expansion & Dense Aspect Extractor
        self.extractor = BM25DenseAspectExtractor(
            idf_registry=self.idf_registry,
            vocab_matrix=self.vocab_matrix,
            schema=schema,
            p=exp_cfg.get("p", 0.50),
            C_exp=exp_cfg.get("C_exp", 2),
            tau_sim=exp_cfg.get("tau_sim", 0.55),
            beta=exp_cfg.get("beta", 1.0),
            tau_base=exp_cfg.get("tau_base", 0.55),
            delta_tau=exp_cfg.get("delta_tau", 0.0),
            mu_ceil=exp_cfg.get("mu_ceil", 0.50),
            eta=exp_cfg.get("eta", 0.0),
            pos_ratios=exp_cfg.get("pos_ratios"),
            bailout_tau_idf=exp_cfg.get("bailout_tau_idf", 3.0),
            min_len_rescue=exp_cfg.get("min_len_rescue", 3),
            analyzer=self.analyzer
        )

        # Phase 3: Cascade Router
        self.router = BM25CascadeRouter(
            tau_bypass=route_cfg.get("tau_bypass", 0.75),
            tau_discard=route_cfg.get("tau_discard", 0.15)
        )

        # Phase 4: Listwise Reranker
        self.reranker = ListwiseLLMRerankerV2(llm_client=self.llm_client)

        # Phase 5: Late Expansion
        self.late_expansion = LateExpansionV2(llm_client=self.llm_client, N_max=vram_cfg.get("N_max", 10))

    def run(self, query: str, top_k_retrieval: int = 30) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline for a given user query.
        Returns detailed timing & results payload.
        """
        metrics = {}
        t_start = time.time()

        # Step A: Aspect Expansion
        t_exp_0 = time.time()
        initial_top_chunks = None
        if self.extractor.schema == "BM25Dense_LocalCascade":
            raw_cands = self.indexer.retrieve(query.lower().split(), top_k=top_k_retrieval)
            initial_top_chunks = [c["text"] for c in raw_cands]

        aspect_payload = self.extractor.extract(query, top_candidate_chunks=initial_top_chunks)
        metrics["expansion_latency_ms"] = (time.time() - t_exp_0) * 1000.0

        # Step B: BM25 Retrieval w/ Weighted Term Scoring
        t_ret_0 = time.time()
        term_weights = aspect_payload.get("term_weights")
        query_input = term_weights if term_weights is not None else aspect_payload.get("augmented_token_list", query.lower().split())
        candidates = self.indexer.retrieve(query_input, top_k=top_k_retrieval)
        metrics["retrieval_latency_ms"] = (time.time() - t_ret_0) * 1000.0

        # Step C: Cascade Routing
        t_route_0 = time.time()
        triage = self.router.route(candidates, aspect_payload)
        metrics["routing_latency_ms"] = (time.time() - t_route_0) * 1000.0

        # Step D: Listwise LLM Reranking (if Rerank Queue has items)
        t_rerank_0 = time.time()
        rerank_queue = triage.get("rerank", [])
        if rerank_queue:
            reranked_chunks = self.reranker.rerank(query, rerank_queue)
        else:
            reranked_chunks = []
        metrics["reranker_latency_ms"] = (time.time() - t_rerank_0) * 1000.0

        # Merge Target Chunks (Bypass Queue + Reranked Chunks)
        target_chunks = triage.get("bypass", []) + reranked_chunks
        if not target_chunks: # Fallback to top retrieved candidate
            target_chunks = candidates[:1]

        # Step E: Late Expansion & Answer Generation
        t_gen_0 = time.time()
        gen_result = self.late_expansion.generate(query, target_chunks)
        metrics["generation_latency_ms"] = (time.time() - t_gen_0) * 1000.0

        metrics["total_latency_seconds"] = time.time() - t_start
        metrics["tti_seconds"] = self.tti_seconds

        return {
            "query": query,
            "answer": gen_result["answer"],
            "aspect_payload": aspect_payload,
            "triage": {
                "num_bypass": len(triage.get("bypass", [])),
                "num_rerank": len(rerank_queue),
                "num_discarded": len(triage.get("discarded", []))
            },
            "metrics": metrics
        }
