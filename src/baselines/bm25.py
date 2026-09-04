import math
import numpy as np
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


def tokenize_standard_unstemmed(text: str) -> List[str]:
    """
    Standard word-boundary tokenization without stemming or stopword removal.
    Strips trailing/leading punctuation while preserving alphanumeric words.
    """
    import re
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())


class LuceneBM25Baseline:
    """
    Lucene BM25 Sparse Retrieval Baseline (Standard Lucene, unstemmed).
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

        self.tokenized_corpus = [tokenize_standard_unstemmed(doc) for doc in self.corpus]
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

        # --- Vectorized Inverted Posting Lists (NumPy) ---
        # Precompute document length normalization vector:
        #   doc_norm[i] = k1 * (1 - b + b * doc_len[i] / avgdl)
        doc_len_arr = np.array(self.doc_len, dtype=np.float32)
        if self.avgdl > 0:
            self.doc_norm = (self.k1 * (1.0 - self.b + self.b * (doc_len_arr / self.avgdl))).astype(np.float32)
        else:
            self.doc_norm = np.full(self.num_docs, self.k1, dtype=np.float32)

        # Build inverted posting lists: term -> (doc_ids_array, tf_array)
        postings_raw = {}  # type: Dict[str, List[tuple]]
        for doc_idx, freq_counter in enumerate(self.doc_freqs):
            for term, tf in freq_counter.items():
                if term not in postings_raw:
                    postings_raw[term] = []
                postings_raw[term].append((doc_idx, tf))

        self.postings = {}  # type: Dict[str, tuple]
        for term, pairs in postings_raw.items():
            dids = np.array([p[0] for p in pairs], dtype=np.int32)
            tfs = np.array([p[1] for p in pairs], dtype=np.float32)
            self.postings[term] = (dids, tfs)

    def retrieve_weighted(
        self, term_weights: Dict[str, float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using vectorized Lucene BM25 scoring
        with weighted term scoring.

        Args:
            term_weights: Dict mapping each unique query term to its weight multiplier.
                          E.g. {"retrieval": 3.0, "rag": 2.0, "generation": 1.0}
            top_k: Number of top results to return.
        """
        scores = np.zeros(self.num_docs, dtype=np.float32)

        for term, weight in term_weights.items():
            if term not in self.postings:
                continue
            dids, tfs = self.postings[term]
            idf_val = self.idf[term]
            denom = tfs + self.doc_norm[dids]
            scores[dids] += weight * idf_val * (tfs * (self.k1 + 1.0)) / denom

        # O(N) top-k selection via argpartition
        if top_k >= self.num_docs:
            top_k_indices = np.argsort(-scores)[:top_k]
        else:
            partitioned = np.argpartition(scores, -top_k)[-top_k:]
            top_k_indices = partitioned[np.argsort(-scores[partitioned])]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx]),
            })
        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using standard unstemmed Lucene BM25 scoring.
        Handles repeated tokens via Counter-based weight conversion.
        """
        tokenized_query = tokenize_standard_unstemmed(query)
        term_weights = dict(Counter(tokenized_query))
        return self.retrieve_weighted(term_weights, top_k=top_k)


class AnalyzedLuceneBM25:
    """
    Analyzed Lucene BM25 Baseline (Pure-Python Lucene Parity).

    Uses:
    1. EdgeRAGAnalyzer (Canonical tokenization, possessive filter, technical keyword exemptions, Lucene stopwords, KStem).
    2. InvertedPostingIndex (Memory-compact inverted posting list with vectorized weighted scoring).

    Default parameters: k1 = 1.5, b = 0.75
    """

    def __init__(
        self,
        corpus: List[str],
        chunk_ids: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        stemmer: str = "kstem",
        use_stopwords: bool = True,
        exempt_technical: bool = True,
    ):
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = corpus
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.k1 = k1
        self.b = b
        self._doc_id_to_idx = {did: idx for idx, did in enumerate(self.chunk_ids)}

        from src.pipeline_v2.indexer.analyzer import EdgeRAGAnalyzer
        from src.pipeline_v2.indexer.posting_index import InvertedPostingIndex

        self.analyzer = EdgeRAGAnalyzer(
            stemmer=stemmer,
            use_stopwords=use_stopwords,
            exempt_technical=exempt_technical,
        )
        self.index = InvertedPostingIndex(k1=k1, b=b)

        # Analyze corpus and build posting index
        analyzed_corpus = [self.analyzer.analyze(doc) for doc in self.corpus]
        self.index.build_from_analyzed_corpus(analyzed_corpus, doc_ids=self.chunk_ids)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using Analyzed Lucene BM25 scoring.
        """
        analyzed_tokens = self.analyzer.analyze(query)
        scored_pairs = self.index.retrieve_analyzed(analyzed_tokens, top_k=top_k)

        results = []
        for chunk_id, score in scored_pairs:
            idx = self._doc_id_to_idx[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": self.corpus[idx],
                "score": score,
            })
        return results

    def retrieve_weighted(self, term_weights: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks scoring directly on pre-analyzed term weights.
        """
        scored_pairs = self.index.retrieve_weighted(term_weights, top_k=top_k)

        results = []
        for chunk_id, score in scored_pairs:
            idx = self._doc_id_to_idx[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": self.corpus[idx],
                "score": score,
            })
        return results

