import math
from typing import List, Dict, Any, Optional
from collections import Counter
from rank_bm25 import BM25Okapi, BM25Plus, BM25L


class BM25Baseline:
    """
    BM25 Sparse Retrieval Baseline (Okapi BM25).
    Fully isolated, CPU-bound, implementing the Okapi BM25 algorithm.

    The inverted index (BM25Okapi) is built once in the constructor. Index-build
    time (TTI) is therefore the constructor wall-clock time, measured by the
    caller. Per-query latency is the ``retrieve`` call only.

    Pass a parallel ``chunk_ids`` list to recover real benchmark chunk IDs;
    otherwise positional ``c_<idx>`` ids are used as a fallback.
    """

    def __init__(self, corpus: List[str], chunk_ids: Optional[List[str]] = None):
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using BM25Okapi.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": scores[idx],
            })
        return results


class BM25PlusBaseline:
    """
    BM25+ Sparse Retrieval Baseline.
    Fixes over-penalization of long documents via lower-bound delta parameter (default delta = 1.0).
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ):
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Plus(self.tokenized_corpus, k1=k1, b=b, delta=delta)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using BM25+.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": scores[idx],
            })
        return results


class BM25LBaseline:
    """
    BM25L Sparse Retrieval Baseline.
    Softens length penalty for longer text chunks.
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ):
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25L(self.tokenized_corpus, k1=k1, b=b, delta=delta)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using BM25L.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": scores[idx],
            })
        return results


class LuceneBM25Baseline:
    """
    Lucene BM25 Sparse Retrieval Baseline.
    Uses Lucene's non-negative IDF formula:
    IDF = ln(1 + (N - n + 0.5) / (n + 0.5))
    Default parameters: k1 = 1.2, b = 0.75
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.doc_freqs = []
        self.nd = {}
        self.num_docs = len(corpus)

        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        for doc in self.tokenized_corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)

            for word in frequencies.keys():
                self.nd[word] = self.nd.get(word, 0) + 1

        self.avgdl = sum(self.doc_len) / self.num_docs if self.num_docs > 0 else 0.0

        # Precompute Lucene IDF dictionary for indexed vocabulary
        self.idf = {}
        for word, freq in self.nd.items():
            self.idf[word] = math.log(
                1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5)
            )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using Lucene BM25 scoring.
        """
        tokenized_query = query.lower().split()
        scores = [0.0] * self.num_docs

        for q in tokenized_query:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (self.doc_len[i] / self.avgdl if self.avgdl > 0 else 1.0)
                )
                score = idf_val * (freq * (self.k1 + 1.0)) / denom
                scores[i] += score

        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx]),
            })
        return results
