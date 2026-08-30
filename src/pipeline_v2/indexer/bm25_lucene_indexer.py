from typing import List, Dict, Any, Optional, Union
from collections import Counter
from src.baselines.bm25 import LuceneBM25Baseline
from .corpus_idf_registry import CorpusIDFRegistry
from .analyzer import EdgeRAGAnalyzer
from .posting_index import InvertedPostingIndex


class BM25LuceneIndexer:
    """
    High-Speed Inverted Indexer for Edge-RAG (V7 Design).
    
    Supports:
    - mode="parity": Uses EdgeRAGAnalyzer + InvertedPostingIndex (compact postings, analyzed KStem parity).
    - mode="legacy": Wraps LuceneBM25Baseline.
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        idf_registry: Optional[CorpusIDFRegistry] = None,
        k1: float = 1.2,
        b: float = 0.75,
        mode: str = "parity",
        analyzer: Optional[EdgeRAGAnalyzer] = None
    ):
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [f"c_{i}" for i in range(len(corpus))]
        self._doc_id_to_idx = {did: idx for idx, did in enumerate(self.chunk_ids)}
        self.mode = mode
        self.k1 = k1
        self.b = b

        if self.mode == "parity":
            self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
            self.index = InvertedPostingIndex(k1=k1, b=b)
            analyzed_corpus = [self.analyzer.analyze(doc) for doc in self.corpus]
            self.index.build_from_analyzed_corpus(analyzed_corpus, doc_ids=self.chunk_ids)

            if idf_registry is not None:
                self.idf_registry = idf_registry
            else:
                doc_freqs = {term: plist.df for term, plist in self.index.postings.items()}
                self.idf_registry = CorpusIDFRegistry(
                    doc_freqs=doc_freqs,
                    num_docs=len(corpus),
                    analyzer=self.analyzer
                )
        else:
            self.baseline = LuceneBM25Baseline(corpus=corpus, chunk_ids=self.chunk_ids, k1=k1, b=b)
            if idf_registry is not None:
                self.idf_registry = idf_registry
            else:
                self.idf_registry = CorpusIDFRegistry(
                    doc_freqs=self.baseline.nd,
                    num_docs=len(corpus)
                )

    def retrieve(
        self, query_input: Union[List[str], str, Dict[str, float]], top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates given tokenized query terms, raw query string, or term weights dict.
        """
        if isinstance(query_input, dict):
            return self.retrieve_weighted(term_weights=query_input, top_k=top_k)

        if self.mode == "parity":
            if isinstance(query_input, str):
                analyzed = self.analyzer.analyze(query_input)
            else:
                analyzed = query_input
            return self.retrieve_weighted(dict(Counter(analyzed)), top_k=top_k)
        else:
            if isinstance(query_input, str):
                tokens = query_input.lower().split()
            else:
                tokens = query_input
            return self.baseline.retrieve_weighted(dict(Counter(tokens)), top_k=top_k)

    def retrieve_weighted(self, term_weights: Dict[str, float], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates using weighted term scoring.
        """
        if self.mode == "parity":
            scored_pairs = self.index.retrieve_weighted(term_weights, top_k=top_k)
            results = []
            for chunk_id, score in scored_pairs:
                idx = self._doc_id_to_idx.get(chunk_id)
                if idx is not None:
                    results.append({
                        "chunk_id": chunk_id,
                        "text": self.corpus[idx],
                        "score": float(score),
                    })
            return results
        else:
            return self.baseline.retrieve_weighted(term_weights=term_weights, top_k=top_k)

