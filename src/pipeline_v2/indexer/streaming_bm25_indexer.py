"""
src/pipeline_v2/indexer/streaming_bm25_indexer.py

High-Performance Streaming BM25 Indexer wrapping StreamingPostingIndex.
Provides full drop-in compatibility with BM25LuceneIndexer and Edge-RAG V7.
"""

import os
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Union, Generator

from .streaming_posting_index import StreamingPostingIndex
from .corpus_idf_registry import CorpusIDFRegistry
from .analyzer import EdgeRAGAnalyzer


class StreamingBM25Indexer:
    """
    Disk-backed, memory-mapped streaming BM25 Indexer.
    Computes Lucene non-negative BM25 scores with Krovetz stemming & WordNet suppletion overrides.
    Directly compatible with CorpusVocabBuilder and V7AspectExtractor.
    """

    def __init__(
        self,
        index_dir: str,
        analyzer: Optional[EdgeRAGAnalyzer] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.index_dir = os.path.abspath(index_dir)
        self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
        self.k1 = float(k1)
        self.b = float(b)

        self.index = StreamingPostingIndex(index_dir=self.index_dir, k1=self.k1, b=self.b)
        self.idf_registry: Optional[CorpusIDFRegistry] = None
        self.stem_to_surface: Dict[str, str] = {}

        self._surface_map_path = os.path.join(self.index_dir, "stem_to_surface.json")
        self._doc_freqs_path = os.path.join(self.index_dir, "doc_freqs.json")

        # If index was already built previously, auto-load metadata
        if os.path.exists(self.index.meta_path) and os.path.exists(self._doc_freqs_path):
            self.load()

    def build_from_stream(
        self,
        doc_stream: Generator[Tuple[str, str], None, None],
        batch_size: int = 50000,
        num_buckets: int = 16,
    ) -> "StreamingBM25Indexer":
        """
        Builds the memory-mapped streaming inverted index from a (doc_id, text) generator.
        """
        res = self.index.build_from_stream(
            doc_stream=doc_stream,
            analyzer=self.analyzer,
            batch_size=batch_size,
            num_buckets=num_buckets,
        )

        self.stem_to_surface = res["stem_to_surface"]
        doc_freqs = res["doc_freqs"]
        num_docs = res["num_docs"]

        # Cache stem_to_surface and doc_freqs for instant reloading
        with open(self._surface_map_path, "w", encoding="utf-8") as f:
            json.dump(self.stem_to_surface, f)

        with open(self._doc_freqs_path, "w", encoding="utf-8") as f:
            json.dump(doc_freqs, f)

        self.idf_registry = CorpusIDFRegistry(
            doc_freqs=doc_freqs,
            num_docs=num_docs,
            analyzer=self.analyzer,
        )

        return self

    def load(self):
        """Loads an existing memory-mapped index from disk."""
        self.index.open()

        if os.path.exists(self._surface_map_path):
            with open(self._surface_map_path, "r", encoding="utf-8") as f:
                self.stem_to_surface = json.load(f)

        if os.path.exists(self._doc_freqs_path):
            with open(self._doc_freqs_path, "r", encoding="utf-8") as f:
                doc_freqs = json.load(f)
            self.idf_registry = CorpusIDFRegistry(
                doc_freqs=doc_freqs,
                num_docs=self.index.num_docs,
                analyzer=self.analyzer,
            )

    def retrieve(
        self, query_input: Union[List[str], str, Dict[str, float]], top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates given tokenized query terms, raw query string, or term weights dict.
        """
        if isinstance(query_input, dict):
            return self.retrieve_weighted(term_weights=query_input, top_k=top_k)

        if isinstance(query_input, str):
            if hasattr(self.analyzer, "analyze"):
                analyzed = self.analyzer.analyze(query_input)
            else:
                analyzed = query_input.lower().split()
        else:
            analyzed = query_input

        term_weights = {term: float(count) for term, count in Counter(analyzed).items()}
        return self.retrieve_weighted(term_weights=term_weights, top_k=top_k)

    def retrieve_weighted(
        self, term_weights: Dict[str, float], top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates using weighted term scoring.
        Returns list of dicts with {"chunk_id": doc_id, "score": float(score)}.
        """
        scored_pairs = self.index.retrieve_weighted(term_weights=term_weights, top_k=top_k)
        return [
            {"chunk_id": doc_id, "score": float(score)}
            for doc_id, score in scored_pairs
        ]
