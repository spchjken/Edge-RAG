import math
import string
from typing import List, Dict, Union, Optional


class CorpusIDFRegistry:
    """
    Unified Shared IDF Registry.
    Computes and stores corpus-wide Inverse Document Frequency (IDF) scores
    using Lucene's non-negative formula:
        IDF(term) = ln(1.0 + (N - n + 0.5) / (n + 0.5))
    Can accept pre-computed doc_freqs from LuceneBM25Baseline for zero-overhead TTI.
    """

    TRANS_TABLE = str.maketrans('', '', string.punctuation.replace('-', '').replace('_', ''))

    def __init__(self, corpus: Optional[List[str]] = None, doc_freqs: Optional[Dict[str, int]] = None, num_docs: Optional[int] = None):
        self.doc_freqs: Dict[str, int] = doc_freqs if doc_freqs is not None else {}
        self.num_docs: int = num_docs if num_docs is not None else (len(corpus) if corpus else 0)
        self.idf_table: Dict[str, float] = {}
        self.tokenized_corpus: Optional[List[List[str]]] = None

        if not self.doc_freqs and corpus:
            # Tokenize corpus only if doc_freqs was not provided
            self.tokenized_corpus = [
                [w for w in doc.lower().translate(self.TRANS_TABLE).split() if len(w) >= 2]
                for doc in corpus
            ]
            for doc_tokens in self.tokenized_corpus:
                for word in set(doc_tokens):
                    self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        # Compute Lucene IDF scores
        if self.num_docs > 0:
            for word, freq in self.doc_freqs.items():
                self.idf_table[word] = math.log(
                    1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5)
                )

        self.max_idf = max(self.idf_table.values()) if self.idf_table else 1.0
        self.median_idf = self._compute_median_idf()

    def _compute_median_idf(self) -> float:
        """Calculates median IDF across corpus vocabulary."""
        if not self.idf_table:
            return 0.0
        sorted_idfs = sorted(self.idf_table.values())
        n = len(sorted_idfs)
        if n % 2 == 1:
            return sorted_idfs[n // 2]
        else:
            return (sorted_idfs[n // 2 - 1] + sorted_idfs[n // 2]) / 2.0

    def get_idf(self, term: Union[str, List[str]]) -> float:
        """
        Returns IDF score for a unigram or bigram term.
        For bigrams, computes the mean constituent word IDF:
            IDF("word1 word2") = (IDF("word1") + IDF("word2")) / 2
        """
        if isinstance(term, str):
            words = term.lower().split()
        else:
            words = [w.lower() for w in term]

        if not words:
            return 0.0

        if len(words) == 1:
            return self.idf_table.get(words[0], self.max_idf * 0.5)

        # Bigram constituent mean IDF
        word_idfs = [self.idf_table.get(w, self.max_idf * 0.5) for w in words]
        return sum(word_idfs) / len(word_idfs)

    def get_normalized_idf(self, term: Union[str, List[str]]) -> float:
        """Returns term IDF normalized to [0, 1]."""
        raw_idf = self.get_idf(term)
        return min(1.0, max(0.0, raw_idf / self.max_idf)) if self.max_idf > 0 else 0.0
