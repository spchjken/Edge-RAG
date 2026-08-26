"""
src/pipeline_v2/indexer/posting_index.py

Memory-Compact Inverted Posting List Index with Vectorized Weighted Retrieval.

Features:
- Compact array('I') storage for posting lists (doc_ids, term_frequencies).
- Non-negative Lucene BM25 scoring with exact length normalization.
- Pure NumPy vectorized accumulator: traverses only postings of active query terms.
- Fast top-K extraction (<5ms per query on CPU for 100k chunks).
"""

import math
from array import array
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import numpy as np


class PostingList:
    """Compact inverted posting list for a single vocabulary term."""
    __slots__ = ("df", "total_tf", "doc_ids", "tfs")

    def __init__(self):
        self.df: int = 0
        self.total_tf: int = 0
        self.doc_ids = array('I')
        self.tfs = array('I')

    def append(self, doc_id: int, tf: int):
        self.doc_ids.append(doc_id)
        self.tfs.append(tf)
        self.df += 1
        self.total_tf += tf


class InvertedPostingIndex:
    """
    Inverted Posting List Index implementing Lucene BM25 retrieval.
    
    Supports:
    - retrieve_weighted(term_weights, top_k): direct weighted vector scoring.
    - retrieve_analyzed(analyzed_tokens, top_k): tokenized query retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1: float = float(k1)
        self.b: float = float(b)
        self.num_docs: int = 0
        self.avgdl: float = 0.0
        self.doc_ids_map: List[str] = []
        self.postings: Dict[str, PostingList] = {}
        self._doc_lens: Optional[np.ndarray] = None
        self._k1_len_norm: Optional[np.ndarray] = None
        self._idf_cache: Dict[str, float] = {}

    def build_from_analyzed_corpus(
        self,
        analyzed_corpus: List[List[str]],
        doc_ids: Optional[List[str]] = None
    ):
        """
        Builds inverted index from pre-analyzed corpus token streams.
        
        Args:
            analyzed_corpus: List of analyzed token lists per document.
            doc_ids: Optional string IDs for documents. Defaults to string indices.
        """
        self.num_docs = len(analyzed_corpus)
        if doc_ids is not None and len(doc_ids) == self.num_docs:
            self.doc_ids_map = [str(did) for did in doc_ids]
        else:
            self.doc_ids_map = [str(i) for i in range(self.num_docs)]

        self.postings.clear()
        self._idf_cache.clear()

        doc_lens = np.zeros(self.num_docs, dtype=np.float32)
        total_tokens = 0

        # Build posting lists
        for doc_idx, tokens in enumerate(analyzed_corpus):
            doc_len = len(tokens)
            doc_lens[doc_idx] = doc_len
            total_tokens += doc_len

            if not tokens:
                continue

            tf_counts = Counter(tokens)
            for term, tf in tf_counts.items():
                if term not in self.postings:
                    self.postings[term] = PostingList()
                self.postings[term].append(doc_idx, tf)

        self.avgdl = (total_tokens / self.num_docs) if self.num_docs > 0 else 1.0
        self._doc_lens = doc_lens

        # Precompute k1 * (1 - b + b * (doc_len / avgdl)) array for instant vectorized scoring
        len_norm = (1.0 - self.b) + self.b * (self._doc_lens / max(self.avgdl, 1e-6))
        self._k1_len_norm = (self.k1 * len_norm).astype(np.float32)

    def idf(self, term: str) -> float:
        """Computes non-negative Lucene IDF for a term with caching."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        plist = self.postings.get(term)
        n = plist.df if plist is not None else 0
        N = self.num_docs

        # Lucene formula: ln(1.0 + (N - n + 0.5) / (n + 0.5))
        val = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        idf_val = max(0.0, val)
        self._idf_cache[term] = idf_val
        return idf_val

    def retrieve_weighted(
        self,
        term_weights: Dict[str, float],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top_k documents by scoring query terms with explicit weights.
        
        Score(D, Q) = sum_{t in w_Q} w_Q(t) * IDF(t) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl)))
        
        Args:
            term_weights: Mapping from analyzed term to non-negative weight.
            top_k: Number of highest-scoring documents to return.
            
        Returns:
            List of (chunk_id, score) tuples sorted descending by score.
        """
        if not term_weights or self.num_docs == 0 or self._k1_len_norm is None:
            return []

        # Vectorized accumulator across active postings
        scores = np.zeros(self.num_docs, dtype=np.float32)
        k1_p1 = self.k1 + 1.0

        for term, weight in term_weights.items():
            if weight <= 0:
                continue
            plist = self.postings.get(term)
            if plist is None or plist.df == 0:
                continue

            term_idf = self.idf(term)
            boost = float(weight * term_idf * k1_p1)

            # Extract arrays
            d_ids = np.frombuffer(plist.doc_ids, dtype=np.uint32)
            d_tfs = np.frombuffer(plist.tfs, dtype=np.uint32).astype(np.float32)

            # Vectorized BM25 addition
            scores[d_ids] += (boost * d_tfs) / (d_tfs + self._k1_len_norm[d_ids])

        # Find top-K positive scores
        active_mask = scores > 0
        num_positive = np.count_nonzero(active_mask)
        if num_positive == 0:
            return []

        k = min(top_k, num_positive)
        if k == self.num_docs:
            top_indices = np.argsort(-scores)
        else:
            partitioned = np.argpartition(-scores, k - 1)[:k]
            top_indices = partitioned[np.argsort(-scores[partitioned])]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s <= 0:
                break
            results.append((self.doc_ids_map[idx], s))

        return results

    def retrieve_analyzed(
        self,
        analyzed_tokens: List[str],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Convenience retrieval for an analyzed token list (unit weights with frequency count).
        """
        if not analyzed_tokens:
            return []
        term_counts = Counter(analyzed_tokens)
        weights = {t: float(count) for t, count in term_counts.items()}
        return self.retrieve_weighted(weights, top_k=top_k)
