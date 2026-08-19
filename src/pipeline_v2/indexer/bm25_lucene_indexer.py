from typing import List, Dict, Any, Optional
from src.baselines.bm25 import LuceneBM25Baseline
from .corpus_idf_registry import CorpusIDFRegistry


class BM25LuceneIndexer:
    """
    High-Speed Inverted Indexer wrapping LuceneBM25Baseline.
    Passes pre-computed BM25 doc_freqs directly to CorpusIDFRegistry for 0ms IDF setup.
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        idf_registry: Optional[CorpusIDFRegistry] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [f"c_{i}" for i in range(len(corpus))]
        self.baseline = LuceneBM25Baseline(corpus=corpus, chunk_ids=self.chunk_ids, k1=k1, b=b)
        
        # Share pre-computed Lucene doc_freqs directly if idf_registry is not passed
        if idf_registry is not None:
            self.idf_registry = idf_registry
        else:
            self.idf_registry = CorpusIDFRegistry(
                doc_freqs=self.baseline.nd,
                num_docs=len(corpus)
            )

    def retrieve(self, tokenized_query: List[str], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidates given tokenized query terms.
        Supports token repetition weighting.
        """
        query_str = " ".join(tokenized_query)
        return self.baseline.retrieve(query=query_str, top_k=top_k)
