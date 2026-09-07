"""
src/evaluation/pyterrier_harness.py

High-Performance Disk-Based Baseline Evaluation Harness using PyTerrier.
Supports standard BM25, compound-preserving analyzed BM25, and Dirichlet-smoothed
Relevance Model 3 (RM3) pseudo-relevance feedback across small and multi-million-document corpora.

Enforces:
1. Disk-backed Terrier Indexing via IterDictIndexer with MetaIndex.
2. O(K) memory scaling for PRF feedback via on-demand disk text fetching.
3. Un-confounded RM3 evaluation: Unified RM3 evaluates identical Dirichlet math
   across both default and analyzed indexes.
4. Exact metric parity with BEIR standards (ir_measures with BEIR_EXP_GAINS).
"""

import os
import time
import math
import hashlib
import inspect
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import pandas as pd
import numpy as np

import pyterrier as pt
import ir_measures
from ir_measures import nDCG, RR, R, P

from src.pipeline_v2.indexer.analyzer import EdgeRAGAnalyzer, LUCENE_STOPWORDS

# Standard BEIR Table 2 exponential gain mapping (2^rel - 1)
BEIR_EXP_GAINS = {1: 1, 2: 3, 3: 7, 4: 15}


def init_pyterrier():
    """Idempotent PyTerrier initialization."""
    if not pt.java.started():
        pt.java.init()


def get_analyzer_version_hash() -> str:
    """Computes a SHA256 hash of EdgeRAGAnalyzer to version analyzed index directories."""
    src = inspect.getsource(EdgeRAGAnalyzer)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]


def sanitize_default_query(query_text: str) -> str:
    """
    Sanitizes raw query text for standard TerrierQL retrieval.
    Removes syntax characters (?, :, <, >, {, }, ^, $, etc.) that trigger
    QueryParserException, matching EnglishTokeniser document-side behavior.
    """
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(query_text))
    tokens = [w for w in cleaned.split() if w.strip()]
    return " ".join(tokens) if tokens else "a"


class PyTerrierIndexManager:
    """Manages building and caching of dual disk indices: default and analyzed."""

    def __init__(self, cache_dir: str = "data/cache/terrier_indices"):
        init_pyterrier()
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.analyzer = EdgeRAGAnalyzer()
        self.analyzer_hash = get_analyzer_version_hash()

    def get_index_paths(self, dataset_name: str) -> Tuple[str, str]:
        """Returns (default_index_path, analyzed_index_path)."""
        safe_name = dataset_name.lower().replace("-", "_")
        default_path = os.path.abspath(os.path.join(self.cache_dir, f"{safe_name}_default"))
        analyzed_path = os.path.abspath(os.path.join(self.cache_dir, f"{safe_name}_analyzed_{self.analyzer_hash}"))
        return default_path, analyzed_path

    def build_or_load_indices(
        self,
        dataset_name: str,
        corpus_docs: List[Dict[str, str]],
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads dual disk indices with MetaIndex enabled.
        
        Args:
            dataset_name: Name of the benchmark dataset.
            corpus_docs: List of doc dicts with keys 'doc_id' and 'text'.
            overwrite: Whether to force re-indexing.
            
        Returns:
            Dict containing 'index_default', 'index_analyzed', and 'timing'.
        """
        default_path, analyzed_path = self.get_index_paths(dataset_name)
        timing = {
            "default_index_time_s": 0.0,
            "analyzed_pretokenize_time_s": 0.0,
            "analyzed_index_time_s": 0.0,
            "analyzed_total_tti_s": 0.0,
        }

        # 1. Build or Load Default Index
        if overwrite or not os.path.exists(os.path.join(default_path, "data.properties")):
            print(f"[PyTerrier] Indexing default corpus ({len(corpus_docs)} docs) -> {default_path}")
            os.makedirs(default_path, exist_ok=True)
            t0 = time.perf_counter()
            indexer = pt.IterDictIndexer(
                default_path,
                overwrite=True,
                meta={"docno": 64, "text": 32768},
            )
            indexer.setProperty("max.term.length", "256")

            def default_iter():
                for doc in corpus_docs:
                    yield {"docno": str(doc["doc_id"]), "text": doc.get("text", "")}

            ref_default = indexer.index(default_iter())
            timing["default_index_time_s"] = time.perf_counter() - t0
            index_default = pt.IndexFactory.of(ref_default)
        else:
            print(f"[PyTerrier] Loading cached default index -> {default_path}")
            index_default = pt.IndexFactory.of(default_path)

        # 2. Build or Load Analyzed Index
        if overwrite or not os.path.exists(os.path.join(analyzed_path, "data.properties")):
            print(f"[PyTerrier] Pre-tokenizing corpus with EdgeRAGAnalyzer ({len(corpus_docs)} docs)...")
            os.makedirs(analyzed_path, exist_ok=True)
            t_pre0 = time.perf_counter()

            # Pre-tokenize docs; tokens joined by whitespace, capping absurdly long non-code tokens to 200 chars
            def analyzed_iter():
                for doc in corpus_docs:
                    raw_tokens = self.analyzer.analyze(doc.get("text", ""))
                    tokens = [t[:200] for t in raw_tokens if len(t) <= 200]
                    yield {"docno": str(doc["doc_id"]), "text": " ".join(tokens)}

            timing["analyzed_pretokenize_time_s"] = time.perf_counter() - t_pre0

            print(f"[PyTerrier] Indexing analyzed corpus with WhitespaceTokeniser -> {analyzed_path}")
            t_idx0 = time.perf_counter()
            indexer_analyzed = pt.IterDictIndexer(
                analyzed_path,
                overwrite=True,
                stemmer=None,
                stopwords=None,
                tokeniser="WhitespaceTokeniser",
                meta={"docno": 64, "text": 32768},
            )
            indexer_analyzed.setProperty("termpipelines", "")
            indexer_analyzed.setProperty("max.term.length", "256")
            ref_analyzed = indexer_analyzed.index(analyzed_iter())
            timing["analyzed_index_time_s"] = time.perf_counter() - t_idx0
            timing["analyzed_total_tti_s"] = timing["analyzed_pretokenize_time_s"] + timing["analyzed_index_time_s"]
            index_analyzed = pt.IndexFactory.of(ref_analyzed)
        else:
            print(f"[PyTerrier] Loading cached analyzed index -> {analyzed_path}")
            index_analyzed = pt.IndexFactory.of(analyzed_path)

        return {
            "index_default": index_default,
            "index_analyzed": index_analyzed,
            "default_path": default_path,
            "analyzed_path": analyzed_path,
            "timing": timing,
        }


class UnifiedRM3Retriever:
    """
    Unified Dirichlet-Smoothed Relevance Model 3 (RM3) PRF Retriever.
    Runs identically across both default and analyzed indexes for un-confounded comparison.
    Memory scales at O(K) per query by reading feedback texts from disk on-demand.
    """

    def __init__(
        self,
        index: Any,
        is_analyzed: bool = False,
        fb_docs: int = 10,
        fb_terms: int = 10,
        fb_lambda: float = 0.5,
        mu: float = 1000.0,
    ):
        self.index = index
        self.is_analyzed = is_analyzed
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.fb_lambda = fb_lambda
        self.mu = mu

        self.lexicon = index.getLexicon()
        self.coll_stats = index.getCollectionStatistics()
        self.total_tokens = max(1, self.coll_stats.getNumberOfTokens())
        self.meta = index.getMetaIndex()
        self.analyzer = EdgeRAGAnalyzer() if is_analyzed else None

        # Base retriever
        if is_analyzed:
            self.base_retriever = pt.terrier.Retriever(
                self.index,
                wmodel="BM25",
                controls={"matchopql": "on"},
                properties={"termpipelines": ""},
            )
        else:
            self.base_retriever = pt.terrier.Retriever(
                self.index,
                wmodel="BM25",
            )

    def _tokenize_text(self, text: str) -> List[str]:
        """Extracts candidate tokens from a feedback document text."""
        if self.is_analyzed:
            # Document text was pre-tokenized and joined by spaces
            return text.split()
        else:
            # Default text: lowercase alphanumeric words
            raw = [w.strip() for w in text.lower().split() if w.strip()]
            cleaned = []
            for w in raw:
                # Basic token filter matching standard IR feedback
                clean_w = "".join(c for c in w if c.isalnum())
                if len(clean_w) >= 2 and clean_w not in LUCENE_STOPWORDS:
                    cleaned.append(clean_w)
            return cleaned

    def _encode_query(self, query_text: str) -> str:
        """Encodes query for first-pass retrieval."""
        if self.is_analyzed:
            tokens = self.analyzer.analyze(query_text)
            if not tokens:
                tokens = [w for w in query_text.lower().split() if w.strip()]
            return " ".join([pt.terrier.Retriever.matchop(t) for t in tokens if t.strip()])
        return sanitize_default_query(query_text)

    def search_queries(self, queries: List[Dict[str, Any]], num_results: int = 100) -> pd.DataFrame:
        """
        Executes 2-pass Unified RM3 PRF across the given queries.
        
        Args:
            queries: List of dicts with 'query_id' and 'question'.
            num_results: Final number of ranked documents per query.
            
        Returns:
            DataFrame with ['qid', 'docno', 'score', 'rank'].
        """
        # 1. Format First-Pass Queries
        encoded_queries = []
        orig_tokens_map = {}
        for q in queries:
            qid = str(q["query_id"])
            q_text = q["question"]
            encoded_q = self._encode_query(q_text)
            encoded_queries.append({"qid": qid, "query": encoded_q})
            
            if self.is_analyzed:
                t_list = self.analyzer.analyze(q_text)
                orig_tokens_map[qid] = t_list if t_list else [w.strip() for w in q_text.lower().split() if w.strip()]
            else:
                clean_q = sanitize_default_query(q_text)
                orig_tokens_map[qid] = [w.strip() for w in clean_q.lower().split() if w.strip()]

        df_queries = pd.DataFrame(encoded_queries)

        # 2. First Pass BM25 Retrieval (fetching top fb_docs per query)
        first_pass_res = self.base_retriever.transform(df_queries)

        # 3. Compute RM3 Expanded Queries
        second_pass_queries = []
        grouped = first_pass_res.groupby("qid")

        for qid, q_row in df_queries.set_index("qid").iterrows():
            orig_tokens = orig_tokens_map.get(qid, [])
            if qid not in grouped.groups:
                # No hits in first pass, retain original query
                second_pass_queries.append({"qid": qid, "query": q_row["query"]})
                continue

            q_hits = grouped.get_group(qid).head(self.fb_docs)
            if len(q_hits) == 0:
                second_pass_queries.append({"qid": qid, "query": q_row["query"]})
                continue

            # Calculate Document Probabilities P(d|Q)
            raw_scores = q_hits["score"].values
            # Non-negative score clipping and normalization
            pos_scores = np.maximum(0.0, raw_scores)
            sum_scores = np.sum(pos_scores)
            if sum_scores > 0:
                p_d = pos_scores / sum_scores
            else:
                p_d = np.ones(len(q_hits)) / len(q_hits)

            # Extract feedback doc tokens from disk on-demand (O(K) memory)
            feedback_docs_tokens = []
            for docid in q_hits["docid"].values:
                doc_text = self.meta.getItem("text", int(docid))
                doc_tokens = self._tokenize_text(doc_text) if doc_text else []
                feedback_docs_tokens.append(doc_tokens)

            # Compute Dirichlet smoothed Language Model P(w|d) and RM1
            candidate_vocab = set()
            doc_token_counts = []
            for doc_tokens in feedback_docs_tokens:
                c = Counter(doc_tokens)
                doc_token_counts.append(c)
                candidate_vocab.update(c.keys())

            if not candidate_vocab:
                second_pass_queries.append({"qid": qid, "query": q_row["query"]})
                continue

            # Accumulate P(w|R) across feedback documents
            rm1_scores = {}
            for w in candidate_vocab:
                lex_entry = self.lexicon.getLexiconEntry(w)
                cf = lex_entry.getFrequency() if lex_entry else 1
                bg_prob = cf / self.total_tokens

                p_w_R = 0.0
                for d_idx, doc_tokens in enumerate(feedback_docs_tokens):
                    doc_len = len(doc_tokens)
                    tf = doc_token_counts[d_idx].get(w, 0)
                    p_w_d = (tf + self.mu * bg_prob) / (doc_len + self.mu)
                    p_w_R += p_w_d * p_d[d_idx]

                rm1_scores[w] = p_w_R

            # Select Top fb_terms
            # Filter out terms that are stopwords
            sorted_terms = sorted(
                [(w, s) for w, s in rm1_scores.items() if w not in LUCENE_STOPWORDS],
                key=lambda x: x[1],
                reverse=True,
            )[:self.fb_terms]

            if not sorted_terms:
                second_pass_queries.append({"qid": qid, "query": q_row["query"]})
                continue

            # Normalize RM1 feedback term weights to sum to 1.0
            sum_rm1 = sum(s for _, s in sorted_terms)
            norm_rm1 = {w: (s / sum_rm1) for w, s in sorted_terms}

            # Combine with original query terms (RM3 Interpolation)
            # Original term base weight = (1 - lambda) / |Q|
            q_term_weight = (1.0 - self.fb_lambda) / max(1, len(orig_tokens))
            combined_weights = {}
            for t in orig_tokens:
                combined_weights[t] = combined_weights.get(t, 0.0) + q_term_weight

            for w, s in norm_rm1.items():
                combined_weights[w] = combined_weights.get(w, 0.0) + (self.fb_lambda * s)

            # Format 2nd-pass query string
            if self.is_analyzed:
                # Format MatchOpQL with weights
                parts = [
                    pt.terrier.Retriever.matchop(term, w=round(float(weight), 6))
                    for term, weight in combined_weights.items()
                    if term.strip()
                ]
                q2_str = " ".join(parts)
            else:
                # Default index: matchop or term^weight
                parts = [
                    pt.terrier.Retriever.matchop(term, w=round(float(weight), 6))
                    for term, weight in combined_weights.items()
                    if term.strip()
                ]
                q2_str = " ".join(parts)

            second_pass_queries.append({"qid": qid, "query": q2_str})

        # 4. Second Pass BM25 Retrieval
        df_q2 = pd.DataFrame(second_pass_queries)
        res_pass2 = self.base_retriever.transform(df_q2)

        # Truncate to num_results per query
        if num_results:
            res_pass2 = res_pass2.groupby("qid").head(num_results).reset_index(drop=True)
            res_pass2["rank"] = res_pass2.groupby("qid").cumcount()

        return res_pass2


class PyTerrierBaselineHarness:
    """
    Orchestrates the Phase 1 Baseline Evaluation Suite across all 5 configurations:
    1. BM25_Default
    2. BM25_Analyzed
    3. BM25_RM3_Terrier_Default
    4. BM25_RM3_Unified_Default
    5. BM25_RM3_Unified_Analyzed
    """

    def __init__(self, index_dict: Dict[str, Any]):
        self.index_default = index_dict["index_default"]
        self.index_analyzed = index_dict["index_analyzed"]
        self.timing = index_dict["timing"]
        self.analyzer = EdgeRAGAnalyzer()

        # Build Base Retrievers
        self.bm25_default = pt.terrier.Retriever(self.index_default, wmodel="BM25")
        self.bm25_analyzed = pt.terrier.Retriever(
            self.index_analyzed,
            wmodel="BM25",
            controls={"matchopql": "on"},
            properties={"termpipelines": ""},
        )

        # Build Native Terrier RM3 Pipeline (Literature Anchor)
        self.bm25_rm3_native = (
            self.bm25_default
            >> pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
            >> self.bm25_default
        )

        # Build Unified RM3 Retrievers (Un-confounded Pair)
        self.unified_rm3_default = UnifiedRM3Retriever(
            self.index_default,
            is_analyzed=False,
            fb_docs=10,
            fb_terms=10,
            fb_lambda=0.5,
            mu=1000.0,
        )
        self.unified_rm3_analyzed = UnifiedRM3Retriever(
            self.index_analyzed,
            is_analyzed=True,
            fb_docs=10,
            fb_terms=10,
            fb_lambda=0.5,
            mu=1000.0,
        )

    def warmup(self, num_queries: int = 10):
        """Warm up JVM JIT compiler with dummy queries before measurement."""
        dummy_q = [{"qid": f"warmup_{i}", "question": "retrieval search machine learning algorithm"} for i in range(num_queries)]
        df_dummy = pd.DataFrame([{"qid": q["qid"], "query": q["question"]} for q in dummy_q])
        try:
            self.bm25_default.transform(df_dummy)
            self.unified_rm3_analyzed.search_queries(dummy_q[:2], num_results=10)
        except Exception:
            pass

    def _chunk_transform(self, transformer: Any, df_q: pd.DataFrame, chunk_size: int = 200) -> pd.DataFrame:
        """Executes transformer.transform in bounded query chunks to cap memory footprint."""
        if len(df_q) <= chunk_size:
            return transformer.transform(df_q)
        
        chunks = [df_q.iloc[i : i + chunk_size] for i in range(0, len(df_q), chunk_size)]
        res_list = []
        for c in chunks:
            res_c = transformer.transform(c)
            res_list.append(res_c)
        return pd.concat(res_list, ignore_index=True)

    def _chunk_search(self, retriever_obj: UnifiedRM3Retriever, queries: List[Dict[str, Any]], chunk_size: int = 200) -> pd.DataFrame:
        """Executes UnifiedRM3 in bounded query chunks to cap memory footprint."""
        if len(queries) <= chunk_size:
            return retriever_obj.search_queries(queries, num_results=100)
        
        res_list = []
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i : i + chunk_size]
            res_c = retriever_obj.search_queries(chunk, num_results=100)
            res_list.append(res_c)
        return pd.concat(res_list, ignore_index=True)

    def evaluate_pipeline(
        self,
        pipeline_name: str,
        queries: List[Dict[str, Any]],
        qrels_ir: List[ir_measures.Qrel],
        gold_map: Dict[str, Dict[str, float]],
        chunk_size: int = 200,
    ) -> Dict[str, Any]:
        """Runs retrieval and computes IR measures and latency for a given pipeline."""
        t0 = time.perf_counter()

        if pipeline_name == "BM25_Default":
            df_q = pd.DataFrame([{"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])} for q in queries])
            res = self._chunk_transform(self.bm25_default, df_q, chunk_size=chunk_size)
        elif pipeline_name == "BM25_Analyzed":
            encoded = []
            for q in queries:
                tokens = self.analyzer.analyze(q["question"])
                if not tokens:
                    tokens = [w for w in q["question"].lower().split() if w.strip()]
                q_enc = " ".join([pt.terrier.Retriever.matchop(t) for t in tokens if t.strip()])
                encoded.append({"qid": str(q["query_id"]), "query": q_enc})
            df_q = pd.DataFrame(encoded)
            res = self._chunk_transform(self.bm25_analyzed, df_q, chunk_size=chunk_size)
        elif pipeline_name == "BM25_RM3_Terrier_Default":
            df_q = pd.DataFrame([{"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])} for q in queries])
            res = self._chunk_transform(self.bm25_rm3_native, df_q, chunk_size=chunk_size)
        elif pipeline_name == "BM25_RM3_Unified_Default":
            res = self._chunk_search(self.unified_rm3_default, queries, chunk_size=chunk_size)
        elif pipeline_name == "BM25_RM3_Unified_Analyzed":
            res = self._chunk_search(self.unified_rm3_analyzed, queries, chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        total_latency_s = time.perf_counter() - t0
        avg_latency_ms = (total_latency_s / max(1, len(queries))) * 1000.0

        # Convert result DataFrame to ir_measures ScoredDoc list
        run_ir = []
        retrieved_by_qid = {}
        for _, row in res.iterrows():
            qid = str(row["qid"])
            docno = str(row["docno"])
            score = float(row["score"])
            run_ir.append(ir_measures.ScoredDoc(qid, docno, score))
            retrieved_by_qid.setdefault(qid, []).append(docno)

        # Standard IR Measures (with pinned BEIR exponential gains)
        measures = [
            nDCG(gains=BEIR_EXP_GAINS) @ 10,
            nDCG(gains=BEIR_EXP_GAINS) @ 50,
            RR @ 10,
            R @ 10,
            R @ 50,
            P @ 10,
        ]
        ir_metrics = ir_measures.calc_aggregate(measures, qrels_ir, run_ir)

        # Custom Metrics (Strict@K / Hit Rate, DocRec@K)
        strict_10_hits = 0
        strict_50_hits = 0
        for qid, ret_list in retrieved_by_qid.items():
            golds = gold_map.get(qid, {})
            gold_set = {did for did, s in golds.items() if s > 0}
            if any(doc in gold_set for doc in ret_list[:10]):
                strict_10_hits += 1
            if any(doc in gold_set for doc in ret_list[:50]):
                strict_50_hits += 1

        total_q = max(1, len(queries))
        strict_at_10 = strict_10_hits / total_q
        strict_at_50 = strict_50_hits / total_q

        return {
            "pipeline": pipeline_name,
            "ndcg_10": float(ir_metrics[nDCG(gains=BEIR_EXP_GAINS) @ 10]),
            "ndcg_50": float(ir_metrics[nDCG(gains=BEIR_EXP_GAINS) @ 50]),
            "mrr_10": float(ir_metrics[RR @ 10]),
            "recall_10": float(ir_metrics[R @ 10]),
            "recall_50": float(ir_metrics[R @ 50]),
            "p_10": float(ir_metrics[P @ 10]),
            "strict_10": strict_at_10,
            "strict_50": strict_at_50,
            "avg_latency_ms": avg_latency_ms,
        }

    def run_all(self, queries: List[Dict[str, Any]], chunk_size: int = 200) -> pd.DataFrame:
        """Runs and evaluates all 5 baselines across the query set."""
        # Convert queries and gold qrels to ir_measures Qrel list
        qrels_ir = []
        gold_map = {}
        for q in queries:
            qid = str(q["query_id"])
            golds = q.get("qrels") or {str(did): 1.0 for did in q.get("gold_doc_ids", [])}
            gold_map[qid] = golds
            for doc_id, rel in golds.items():
                qrels_ir.append(ir_measures.Qrel(qid, str(doc_id), int(rel)))

        # Warm up
        self.warmup(num_queries=10)

        pipelines = [
            "BM25_Default",
            "BM25_Analyzed",
            "BM25_RM3_Terrier_Default",
            "BM25_RM3_Unified_Default",
            "BM25_RM3_Unified_Analyzed",
        ]

        results = []
        for p in pipelines:
            print(f"[PyTerrier] Evaluating {p}...")
            res_dict = self.evaluate_pipeline(p, queries, qrels_ir, gold_map, chunk_size=chunk_size)
            results.append(res_dict)

        return pd.DataFrame(results)
