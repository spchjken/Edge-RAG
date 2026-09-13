"""
src/evaluation/pyterrier_harness.py

High-Performance Disk-Based Baseline Evaluation Harness using PyTerrier.
Evaluates standard Terrier default indexing baselines:
  1. BM25_Default (Standard Terrier BM25)
  2. BM25_RM3_Terrier_Default (Native Java Relevance Model 3 PRF)
  3. BM25_Bo1_Terrier_Default (Native Java Bose-Einstein DFR PRF)
  4. DPH (Terrier Divergence From Randomness Model)

Features:
- Full BEIR metric parity using ir_measures standard linear gain nDCG@10.
- Deep candidate-funnel diagnostics: R@10-1000, Completeness@100/500/1000, Strict@10-1000, Oracle-nDCG@10.
- Strict BRIGHT exclusion semantics: Pre- and post-filtering with adaptive depth padding.
- Memory-safe streaming execution in bounded query chunks (chunk_size=200).
- Isolated PyTerrier single-query API latency benchmarking (P50/P90/P99) and resource tracking.
- Optional compressed Parquet candidate run persistence (snappy).
"""

import os
import sys
import time
import math
import random
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd
import numpy as np
import psutil

import pyterrier as pt
import ir_measures
from ir_measures import nDCG, RR, R, P, AP

# Standard literature Table 2 exponential gain mapping (2^rel - 1) for supplemental comparison
EXP_GAINS = {1: 1, 2: 3, 3: 7, 4: 15}
BEIR_EXP_GAINS = EXP_GAINS  # Backward-compatible alias

# Primary Benchmark Metrics (Official BEIR Linear Gains + Candidate Funnel)
PRIMARY_MEASURES = [
    nDCG @ 10,   # Standard linear gain (official BEIR publication parity)
    nDCG @ 50,
    nDCG @ 100,
    AP @ 100,    # MAP@100
    RR @ 10,
    R @ 10,
    R @ 50,
    R @ 100,
    R @ 200,
    R @ 500,
    R @ 1000,
    P @ 10,
    P @ 100,
    nDCG(gains=EXP_GAINS) @ 10,  # Supplemental exp_ndcg_10
]


def init_pyterrier(mem: int = 3072):
    """Idempotent PyTerrier initialization with bounded JVM heap and OOM protection."""
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("500\n")
        except Exception:
            pass

    if not pt.java.started():
        try:
            pt.java.set_memory_limit(mem)
        except Exception:
            pass
        pt.java.init()


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


def get_directory_size_mb(path: str) -> float:
    """Computes total disk footprint of a directory in megabytes."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024.0 * 1024.0), 2)


def create_exclusion_filter(excluded_map: Optional[Dict[str, Set[str]]] = None, max_docs: Optional[int] = None):
    """
    Creates a PyTerrier transformer that drops documents in excluded_map[qid],
    slices each query to top max_docs, and recomputes contiguous 0-indexed ranks.
    Preserves all PyTerrier metadata columns (qid, query, docno, score, docid, etc.).
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
        if excluded_map:
            mask = [
                str(row["docno"]) not in excluded_map.get(str(row["qid"]), set())
                for _, row in df.iterrows()
            ]
            df = df[mask]
        
        if max_docs is not None:
            df = df.groupby("qid").head(max_docs)
            
        df = df.reset_index(drop=True)
        df["rank"] = df.groupby("qid").cumcount()
        return df

    return pt.apply.generic(_filter)


def calc_oracle_ndcg_10(retrieved_docs: List[str], gold_map: Dict[str, float]) -> float:
    """
    Computes Oracle nDCG@10 of an ideal reranking of the retrieved candidate documents,
    normalized by the full ground-truth ideal DCG (IDCG@10) using standard linear gain:
    DCG = sum_{i=1}^10 rel_i / log2(i + 1).
    """
    if not gold_map:
        return 0.0
        
    retrieved_rels = [gold_map.get(str(doc_id), 0.0) for doc_id in retrieved_docs]
    oracle_sorted_rels = sorted([r for r in retrieved_rels if r > 0], reverse=True)[:10]
    if not oracle_sorted_rels:
        return 0.0
        
    dcg_oracle = sum(r / math.log2(i + 2) for i, r in enumerate(oracle_sorted_rels))
    
    all_gold_rels = sorted([r for r in gold_map.values() if r > 0], reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_gold_rels))
    
    if idcg <= 0.0:
        return 0.0
        
    return dcg_oracle / idcg


def benchmark_single_query_api_latency(
    transformer: Any,
    queries: List[Dict[str, Any]],
    sample_size: int = 1000,
    warmup_size: int = 30,
    seed: int = 42,
    llm_rewriter: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Measures PyTerrier single-query API latency (P50, P90, P99, mean in ms).
    Evaluates on all queries if len(queries) <= sample_size, or a seeded sample of sample_size.
    Directly benchmarks the online API path: Python DataFrame -> JNI -> JVM -> index matching -> DataFrame.
    If llm_rewriter is provided, adds per-query generation latency for true end-to-end measurement.
    """
    if not queries:
        return {
            "retrieval_api_p50_ms": 0.0,
            "retrieval_api_p90_ms": 0.0,
            "retrieval_api_p99_ms": 0.0,
            "retrieval_api_mean_ms": 0.0,
        }

    # Deterministic sampling
    if len(queries) <= sample_size:
        eval_queries = list(queries)
    else:
        rng = random.Random(seed)
        eval_queries = rng.sample(queries, sample_size)

    # Prepare single-query dataframes
    single_dfs = [
        pd.DataFrame([{"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])}])
        for q in eval_queries
    ]

    # Warmup
    for q_df in single_dfs[: min(warmup_size, len(single_dfs))]:
        try:
            _ = transformer.transform(q_df)
        except Exception:
            pass

    # Timed individual queries
    latencies_ms = []
    for q, q_df in zip(eval_queries, single_dfs):
        q_gen_ms = 0.0
        if llm_rewriter is not None and hasattr(llm_rewriter, "get_query_gen_time_ms"):
            q_text = q_df["query"].iloc[0] if "query" in q_df.columns else (q.get("question") or q.get("query", ""))
            q_gen_ms = llm_rewriter.get_query_gen_time_ms(q_text)
            if q_gen_ms == 0.0:
                raw_text = q.get("question") or q.get("query", "")
                q_gen_ms = llm_rewriter.get_query_gen_time_ms(raw_text)

        t0 = time.perf_counter()
        _ = transformer.transform(q_df)
        latencies_ms.append(((time.perf_counter() - t0) * 1000.0) + q_gen_ms)

    return {
        "retrieval_api_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "retrieval_api_p90_ms": round(float(np.percentile(latencies_ms, 90)), 2),
        "retrieval_api_p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "retrieval_api_mean_ms": round(float(np.mean(latencies_ms)), 2),
    }


class PyTerrierIndexManager:
    """Manages building and caching of standard default Terrier disk indices."""

    def __init__(self, cache_dir: str = "data/cache/terrier_indices"):
        init_pyterrier()
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_index_path(self, dataset_name: str) -> str:
        """Returns path to the default Terrier index."""
        safe_name = dataset_name.lower().replace("-", "_")
        return os.path.abspath(os.path.join(self.cache_dir, f"{safe_name}_default"))

    def build_or_load_indices(
        self,
        dataset_name: str,
        corpus_docs: Optional[Any] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads standard default disk index with MetaIndex enabled.
        Supports streaming disk generators with zero raw-corpus RAM retention.
        """
        default_path = self.get_index_path(dataset_name)
        timing = {
            "default_build_time_s": 0.0,
            "default_load_time_s": 0.0,
        }

        need_default = overwrite or not os.path.exists(os.path.join(default_path, "data.properties"))

        if not need_default:
            print(f"[PyTerrier] Loading cached default index -> {default_path}")
            t_ld = time.perf_counter()
            index_default = pt.IndexFactory.of(default_path)
            timing["default_load_time_s"] = round(time.perf_counter() - t_ld, 2)
            return {
                "index_default": index_default,
                "default_path": default_path,
                "default_disk_mb": get_directory_size_mb(default_path),
                "timing": timing,
            }

        from src.evaluation.benchmark_loader import BenchmarkLoader

        def _get_raw_stream():
            if corpus_docs is not None:
                for doc in corpus_docs:
                    if isinstance(doc, dict):
                        yield str(doc["doc_id"]), doc.get("text", "")
                    else:
                        yield str(doc[0]), str(doc[1])
            else:
                for did, text in BenchmarkLoader.stream_corpus(dataset_name):
                    yield str(did), text

        print(f"[PyTerrier] Indexing default corpus via stream -> {default_path}")
        os.makedirs(default_path, exist_ok=True)
        t0 = time.perf_counter()
        indexer = pt.IterDictIndexer(
            default_path,
            overwrite=True,
            meta={"docno": 512, "text": 4096},
        )
        indexer.setProperty("max.term.length", "512")
        indexer.setProperty("indexing.max.memory", "1073741824")  # 1 GiB flush threshold

        def default_iter():
            for did, text in _get_raw_stream():
                yield {"docno": did, "text": text}

        ref_default = indexer.index(default_iter())
        timing["default_build_time_s"] = round(time.perf_counter() - t0, 2)
        index_default = pt.IndexFactory.of(ref_default)

        return {
            "index_default": index_default,
            "default_path": default_path,
            "default_disk_mb": get_directory_size_mb(default_path),
            "timing": timing,
        }

    def get_dense_index_path(self, dataset_name: str, model_tag: str = "bge_small") -> str:
        """Returns path to the FlexIndex dense index."""
        safe_name = dataset_name.lower().replace("-", "_")
        return os.path.abspath(os.path.join("data/cache/dense_indices", f"{safe_name}_{model_tag}"))

    def get_splade_index_path(self, dataset_name: str, model_tag: str = "splade_v3") -> str:
        """Returns path to the PisaIndex SPLADE index."""
        safe_name = dataset_name.lower().replace("-", "_")
        return os.path.abspath(os.path.join("data/cache/splade_indices", f"{safe_name}_{model_tag}"))

    def build_or_load_dense_index(
        self,
        dataset_name: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Builds or loads disk-backed FlexIndex for dense BGE retrieval."""
        import pyterrier_dr as pt_dr
        dense_path = self.get_dense_index_path(dataset_name)
        timing = {"dense_build_time_s": 0.0, "dense_load_time_s": 0.0}
        bge_model = pt_dr.HgfBiEncoder.from_pretrained(model_name, device="cuda", batch_size=batch_size)
        dense_indexer = pt_dr.FlexIndex(dense_path)

        if not overwrite and dense_indexer.built():
            print(f"[PyTerrier-DR] Loading cached FlexIndex -> {dense_path}")
            return {
                "bge_model": bge_model,
                "dense_indexer": dense_indexer,
                "dense_path": dense_path,
                "dense_disk_mb": get_directory_size_mb(dense_path),
                "timing": timing,
            }

        from src.evaluation.benchmark_loader import BenchmarkLoader
        print(f"[PyTerrier-DR] Building FlexIndex via stream -> {dense_path}")
        t0 = time.perf_counter()

        def stream_dict():
            for did, text in BenchmarkLoader.stream_corpus(dataset_name):
                yield {"docno": str(did), "text": str(text)}

        mode = "overwrite" if (overwrite or os.path.exists(dense_path)) else "create"
        (bge_model >> dense_indexer.indexer(mode=mode)).index(stream_dict())
        timing["dense_build_time_s"] = round(time.perf_counter() - t0, 2)
        return {
            "bge_model": bge_model,
            "dense_indexer": dense_indexer,
            "dense_path": dense_path,
            "dense_disk_mb": get_directory_size_mb(dense_path),
            "timing": timing,
        }

    def build_or_load_splade_index(
        self,
        dataset_name: str,
        model_name: str = "naver/splade-v3-distilbert",
        batch_size: int = 64,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Builds or loads disk-backed PisaIndex for neural sparse SPLADE retrieval."""
        import pyterrier_splade as pt_splade
        from pyterrier_pisa import PisaIndex
        splade_path = self.get_splade_index_path(dataset_name)
        timing = {"splade_build_time_s": 0.0, "splade_load_time_s": 0.0}
        splade_model = pt_splade.Splade(model=model_name, device="cuda")
        splade_index = PisaIndex(splade_path, stemmer="none")

        if not overwrite and splade_index.built():
            print(f"[PyTerrier-SPLADE] Loading cached PisaIndex -> {splade_path}")
            return {
                "splade_model": splade_model,
                "splade_index": splade_index,
                "splade_path": splade_path,
                "splade_disk_mb": get_directory_size_mb(splade_path),
                "timing": timing,
            }

        from src.evaluation.benchmark_loader import BenchmarkLoader
        print(f"[PyTerrier-SPLADE] Building PisaIndex via stream -> {splade_path}")
        t0 = time.perf_counter()

        def stream_dict():
            for did, text in BenchmarkLoader.stream_corpus(dataset_name):
                yield {"docno": str(did), "text": str(text)}

        mode = "overwrite" if (overwrite or os.path.exists(splade_path)) else "create"
        (splade_model.doc_encoder(batch_size=batch_size) >> splade_index.toks_indexer(mode=mode)).index(stream_dict())
        timing["splade_build_time_s"] = round(time.perf_counter() - t0, 2)
        return {
            "splade_model": splade_model,
            "splade_index": splade_index,
            "splade_path": splade_path,
            "splade_disk_mb": get_directory_size_mb(splade_path),
            "timing": timing,
        }


class PyTerrierBaselineHarness:
    """
    Orchestrates the 4 Standard Terrier Default Baselines:
      1. BM25_Default
      2. BM25_RM3_Terrier_Default
      3. BM25_Bo1_Terrier_Default
      4. DPH
    """

    def __init__(self, index_dict: Optional[Dict[str, Any]] = None):
        index_dict = index_dict or {}
        self.index_dict = index_dict
        self.index_default = index_dict.get("index_default")
        self.bge_model = index_dict.get("bge_model")
        self.dense_indexer = index_dict.get("dense_indexer")
        self.splade_model = index_dict.get("splade_model")
        self.splade_index = index_dict.get("splade_index")
        self.timing = index_dict.get("timing", {})
        self.default_disk_mb = index_dict.get("default_disk_mb", 0.0)
        self.dense_disk_mb = index_dict.get("dense_disk_mb", 0.0)
        self.splade_disk_mb = index_dict.get("splade_disk_mb", 0.0)

    def warmup(self, num_queries: int = 10):
        """Warm up JVM JIT compiler with dummy queries before measurement."""
        if self.index_default is None:
            return
        dummy_q = [{"qid": f"warmup_{i}", "query": "retrieval search machine learning algorithm"} for i in range(num_queries)]
        df_dummy = pd.DataFrame(dummy_q)
        try:
            bm25 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
            _ = bm25.transform(df_dummy)
        except Exception:
            pass

    def build_pipeline(self, pipeline_name: str, excluded_map: Optional[Dict[str, Set[str]]] = None) -> Any:
        """
        Builds the specified retrieval pipeline enforcing strict BRIGHT pre/post exclusion filtering.
        """
        max_ex = max([len(s) for s in excluded_map.values()] or [0]) if excluded_map else 0

        if pipeline_name == "BM25_Default":
            if excluded_map:
                retriever = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retriever >> filter_post
            return pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)

        elif pipeline_name == "DPH":
            if excluded_map:
                retriever = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retriever >> filter_post
            return pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)

        elif pipeline_name == "BM25_RM3_Terrier_Default":
            if excluded_map:
                # First pass: request K1 = min(max(100, fb_docs + max_ex), 300), filter excluded, slice to fb_docs=10
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                # Second pass: request K2 = min(1000 + max_ex, 3000), filter excluded, slice to 1000
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> rm3 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)
                return pass1 >> rm3 >> pass2

        elif pipeline_name == "BM25_Bo1_Terrier_Default":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> bo1 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=1000)
                return pass1 >> bo1 >> pass2

        elif pipeline_name == "DPH_Bo1_Terrier_Default":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> bo1 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=10)
                bo1 = pt.rewrite.Bo1QueryExpansion(self.index_default, fb_terms=10, fb_docs=10)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)
                return pass1 >> bo1 >> pass2

        elif pipeline_name == "DPH_RM3_Terrier_Default":
            if excluded_map:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(max(100, 10 + max_ex), 300))
                filter1 = create_exclusion_filter(excluded_map, max_docs=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=min(1000 + max_ex, 3000))
                filter2 = create_exclusion_filter(excluded_map, max_docs=1000)
                return pass1 >> filter1 >> rm3 >> pass2 >> filter2
            else:
                pass1 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=10)
                rm3 = pt.rewrite.RM3(self.index_default, fb_terms=10, fb_docs=10, fb_lambda=0.5)
                pass2 = pt.terrier.Retriever(self.index_default, wmodel="DPH", num_results=1000)
                return pass1 >> rm3 >> pass2

        elif pipeline_name == "BGE_Small_Dense":
            if self.bge_model is None or self.dense_indexer is None:
                raise ValueError("BGE_Small_Dense requires bge_model and dense_indexer in index_dict")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.bge_model.query_encoder() >> self.dense_indexer.retriever(num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "SPLADE_v3_PISA":
            if self.splade_model is None or self.splade_index is None:
                raise ValueError("SPLADE_v3_PISA requires splade_model and splade_index in index_dict")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.splade_model.query_encoder() >> self.splade_index.quantized(num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "BGE_Vocab_QE":
            if not hasattr(self, "bge_vocab_rewriter") or self.bge_vocab_rewriter is None:
                raise ValueError("BGE_Vocab_QE requires bge_vocab_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.bge_vocab_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif pipeline_name == "LLM_Q2E_ZS":
            if not hasattr(self, "llm_qe_rewriter") or self.llm_qe_rewriter is None:
                raise ValueError("LLM_Q2E_ZS requires llm_qe_rewriter to be registered on harness")
            k = min(1000 + max_ex, 3000) if excluded_map else 1000
            retr = self.llm_qe_rewriter >> pt.terrier.Retriever(self.index_default, wmodel="BM25", num_results=k)
            if excluded_map:
                filter_post = create_exclusion_filter(excluded_map, max_docs=1000)
                return retr >> filter_post
            return retr

        elif hasattr(self, "pipelines") and pipeline_name in self.pipelines:
            base = self.pipelines[pipeline_name]
            if excluded_map:
                filter_tf = create_exclusion_filter(excluded_map, max_docs=1000)
                class _FilteredWrapper:
                    def transform(self, df):
                        return filter_tf.transform(base.transform(df))
                return _FilteredWrapper()
            return base


        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

    def evaluate_pipeline(
        self,
        pipeline_name: str,
        queries: List[Dict[str, Any]],
        qrels_ir: List[ir_measures.Qrel],
        gold_map: Dict[str, Dict[str, float]],
        chunk_size: int = 200,
        save_runs_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs retrieval and computes standard IR measures, candidate-funnel diagnostics,
        and latency in bounded query chunks.
        """
        t0 = time.perf_counter()

        # 1. Query formatting & exclusion extraction
        excluded_map = {str(q["query_id"]): set(q.get("excluded_doc_ids", [])) for q in queries if q.get("excluded_doc_ids")}
        formatted_queries = [
            {"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])}
            for q in queries
        ]

        # 2. Build Pipeline
        transformer = self.build_pipeline(pipeline_name, excluded_map=excluded_map)

        # 3. Stream & Accumulate Metrics in Bounded Query Chunks
        total_queries = len(queries)
        metric_sums = {m: 0.0 for m in PRIMARY_MEASURES}
        strict_10_hits = 0
        strict_50_hits = 0
        strict_100_hits = 0
        strict_1000_hits = 0
        completeness_100_hits = 0
        completeness_500_hits = 0
        completeness_1000_hits = 0
        oracle_ndcg_sum = 0.0
        total_retrieval_time_s = 0.0

        # Pre-group qrels by qid for fast chunk lookup
        qrels_by_qid: Dict[str, List[ir_measures.Qrel]] = {}
        for qrel in qrels_ir:
            qrels_by_qid.setdefault(str(qrel.query_id), []).append(qrel)

        persisted_chunks = []

        for i in range(0, total_queries, chunk_size):
            chunk_queries = queries[i : i + chunk_size]
            chunk_df_q = pd.DataFrame(formatted_queries[i : i + chunk_size])

            t_ret_c0 = time.perf_counter()
            res_c = transformer.transform(chunk_df_q)
            total_retrieval_time_s += (time.perf_counter() - t_ret_c0)

            # Build chunk ScoredDoc list & retrieved candidate map
            chunk_run_ir = []
            chunk_retrieved_by_qid: Dict[str, List[str]] = {}
            for _, row in res_c.iterrows():
                qid = str(row["qid"])
                docno = str(row["docno"])
                # Defensive check against exclusions
                if qid in excluded_map and docno in excluded_map[qid]:
                    continue
                score = float(row["score"])
                chunk_run_ir.append(ir_measures.ScoredDoc(qid, docno, score))
                chunk_retrieved_by_qid.setdefault(qid, []).append(docno)

            # Compute custom candidate-funnel metrics for queries in this chunk
            for q in chunk_queries:
                qid = str(q["query_id"])
                ret_list = chunk_retrieved_by_qid.get(qid, [])
                golds = gold_map.get(qid, {})
                gold_set = {did for did, s in golds.items() if s > 0}

                # Strict@K hits
                if any(doc in gold_set for doc in ret_list[:10]):
                    strict_10_hits += 1
                if any(doc in gold_set for doc in ret_list[:50]):
                    strict_50_hits += 1
                if any(doc in gold_set for doc in ret_list[:100]):
                    strict_100_hits += 1
                if any(doc in gold_set for doc in ret_list[:1000]):
                    strict_1000_hits += 1

                # Completeness@K hits (100% of gold docs in top K)
                if gold_set:
                    if gold_set.issubset(set(ret_list[:100])):
                        completeness_100_hits += 1
                    if gold_set.issubset(set(ret_list[:500])):
                        completeness_500_hits += 1
                    if gold_set.issubset(set(ret_list[:1000])):
                        completeness_1000_hits += 1

                # Oracle-nDCG@10 (linear gain against full-qrel IDCG)
                oracle_ndcg_sum += calc_oracle_ndcg_10(ret_list[:1000], golds)

            # Build chunk qrels
            chunk_qrels = []
            for q in chunk_queries:
                qid = str(q["query_id"])
                chunk_qrels.extend(qrels_by_qid.get(qid, []))

            # Compute chunk IR metrics via iter_calc
            if chunk_run_ir and chunk_qrels:
                for mv in ir_measures.iter_calc(PRIMARY_MEASURES, chunk_qrels, chunk_run_ir):
                    metric_sums[mv.measure] += float(mv.value)

            if save_runs_dir and dataset_name:
                persisted_chunks.append(res_c[["qid", "docno", "score", "rank"]].copy())

            # Release chunk memory
            del res_c, chunk_run_ir, chunk_retrieved_by_qid, chunk_qrels

            if total_queries > 500 and (min(i + chunk_size, total_queries) % 1000 < chunk_size or (i + chunk_size) >= total_queries):
                print(f"  [{pipeline_name}] Progress: {min(i + chunk_size, total_queries)}/{total_queries} queries evaluated...", flush=True)

        # Optional: persist candidate run as compressed Parquet
        if save_runs_dir and dataset_name and persisted_chunks:
            os.makedirs(save_runs_dir, exist_ok=True)
            safe_ds = dataset_name.lower().replace("-", "_")
            parquet_path = os.path.join(save_runs_dir, f"{safe_ds}_{pipeline_name}.parquet")
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                writer = None
                for chunk_df in persisted_chunks:
                    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
                    writer.write_table(table)
                if writer is not None:
                    writer.close()
            except Exception:
                run_df = pd.concat(persisted_chunks, ignore_index=True)
                run_df.to_parquet(parquet_path, compression="snappy", index=False)
                del run_df
            print(f"  [{pipeline_name}] Persisted candidate run -> {parquet_path}", flush=True)
            del persisted_chunks

        is_neural = pipeline_name in ("BGE_Small_Dense", "SPLADE_v3_PISA", "LLM_Q2E_ZS")
        llm_rewriter = getattr(self, "llm_qe_rewriter", None) if pipeline_name == "LLM_Q2E_ZS" else None
        latency_metrics = benchmark_single_query_api_latency(
            transformer,
            queries,
            sample_size=100 if is_neural else 1000,
            warmup_size=5 if is_neural else 30,
            seed=42,
            llm_rewriter=llm_rewriter,
        )

        total_wall_time_s = time.perf_counter() - t0
        if llm_rewriter is not None and hasattr(llm_rewriter, "get_query_gen_time_ms"):
            total_wall_time_s += sum(
                llm_rewriter.get_query_gen_time_ms(q.get("question") or q.get("query", "")) / 1000.0
                for q in queries
            )
        total_q = max(1, total_queries)
        harness_per_query_ms = round((total_wall_time_s / total_q) * 1000.0, 2)
        batch_throughput_qps = round(total_queries / max(0.001, total_retrieval_time_s), 2)
        host_ram_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 2)

        return {
            "pipeline": pipeline_name,
            # Primary quality (linear gains)
            "ndcg_10": round(metric_sums[PRIMARY_MEASURES[0]] / total_q, 4),
            "ndcg_50": round(metric_sums[PRIMARY_MEASURES[1]] / total_q, 4),
            "exp_ndcg_10": round(metric_sums[PRIMARY_MEASURES[13]] / total_q, 4),
            "ndcg_100": round(metric_sums[PRIMARY_MEASURES[2]] / total_q, 4),
            "map_100": round(metric_sums[PRIMARY_MEASURES[3]] / total_q, 4),
            "mrr_10": round(metric_sums[PRIMARY_MEASURES[4]] / total_q, 4),
            "p_10": round(metric_sums[PRIMARY_MEASURES[11]] / total_q, 4),
            "p_100": round(metric_sums[PRIMARY_MEASURES[12]] / total_q, 4),
            "strict_10": round(strict_10_hits / total_q, 4),
            "strict_50": round(strict_50_hits / total_q, 4),
            # Candidate funnel
            "recall_10": round(metric_sums[PRIMARY_MEASURES[5]] / total_q, 4),
            "recall_50": round(metric_sums[PRIMARY_MEASURES[6]] / total_q, 4),
            "recall_100": round(metric_sums[PRIMARY_MEASURES[7]] / total_q, 4),
            "recall_200": round(metric_sums[PRIMARY_MEASURES[8]] / total_q, 4),
            "recall_500": round(metric_sums[PRIMARY_MEASURES[9]] / total_q, 4),
            "recall_1000": round(metric_sums[PRIMARY_MEASURES[10]] / total_q, 4),
            "completeness_100": round(completeness_100_hits / total_q, 4),
            "completeness_500": round(completeness_500_hits / total_q, 4),
            "completeness_1000": round(completeness_1000_hits / total_q, 4),
            "strict_100": round(strict_100_hits / total_q, 4),
            "strict_1000": round(strict_1000_hits / total_q, 4),
            "oracle_ndcg_10": round(oracle_ndcg_sum / total_q, 4),
            # Latency & Throughput
            "retrieval_api_p50_ms": latency_metrics["retrieval_api_p50_ms"],
            "retrieval_api_p90_ms": latency_metrics["retrieval_api_p90_ms"],
            "retrieval_api_p99_ms": latency_metrics["retrieval_api_p99_ms"],
            "retrieval_api_mean_ms": latency_metrics["retrieval_api_mean_ms"],
            "batch_throughput_qps": batch_throughput_qps,
            "harness_per_query_ms": harness_per_query_ms,
            # Resources
            "index_disk_mb": (
                self.dense_disk_mb if pipeline_name == "BGE_Small_Dense"
                else self.splade_disk_mb if pipeline_name == "SPLADE_v3_PISA"
                else self.default_disk_mb
            ),
            "host_ram_peak_mb": host_ram_mb,
            "index_build_s": round(
                self.timing.get("dense_build_time_s", 0.0) if pipeline_name == "BGE_Small_Dense"
                else self.timing.get("splade_build_time_s", 0.0) if pipeline_name == "SPLADE_v3_PISA"
                else self.timing.get("default_build_time_s", 0.0),
                2
            ),
            "cache_load_s": round(
                self.timing.get("dense_load_time_s", 0.0) if pipeline_name == "BGE_Small_Dense"
                else self.timing.get("splade_load_time_s", 0.0) if pipeline_name == "SPLADE_v3_PISA"
                else self.timing.get("default_load_time_s", 0.0),
                2
            ),
        }

    def run_all(
        self,
        queries: List[Dict[str, Any]],
        chunk_size: int = 200,
        save_runs_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        pipelines: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Runs and evaluates the specified baselines across the query set."""
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

        if pipelines is None:
            pipelines = [
                "BM25_Default",
                "BM25_RM3_Terrier_Default",
                "BM25_Bo1_Terrier_Default",
                "DPH",
            ]

        results = []
        for p in pipelines:
            print(f"[PyTerrier] Evaluating {p}...", flush=True)
            res_dict = self.evaluate_pipeline(
                p,
                queries,
                qrels_ir,
                gold_map,
                chunk_size=chunk_size,
                save_runs_dir=save_runs_dir,
                dataset_name=dataset_name,
            )
            results.append(res_dict)
            import gc
            gc.collect()

        return pd.DataFrame(results)
