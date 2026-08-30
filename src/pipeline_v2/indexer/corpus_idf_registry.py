import math
from typing import List, Dict, Union, Optional
from .analyzer import EdgeRAGAnalyzer


class CorpusIDFRegistry:
    """
    Unified Shared IDF Registry.
    Computes and stores corpus-wide Inverse Document Frequency (IDF) scores
    using Lucene's non-negative formula:
        IDF(term) = ln(1.0 + (N - n + 0.5) / (n + 0.5))
    Can accept pre-computed doc_freqs from InvertedPostingIndex or AnalyzedLuceneBM25.
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        doc_freqs: Optional[Dict[str, int]] = None,
        num_docs: Optional[int] = None,
        analyzer: Optional[EdgeRAGAnalyzer] = None
    ):
        self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
        self.doc_freqs: Dict[str, int] = doc_freqs if doc_freqs is not None else {}
        self.num_docs: int = num_docs if num_docs is not None else (len(corpus) if corpus else 0)
        self.idf_table: Dict[str, float] = {}
        self.tokenized_corpus: Optional[List[List[str]]] = None

        if not self.doc_freqs and corpus:
            # Tokenize and analyze corpus with EdgeRAGAnalyzer
            self.tokenized_corpus = [
                self.analyzer.analyze(doc)
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

        # Build Index-Time Boundary Map for O(1) compound & bailout lookup
        import re
        self.boundary_prefix_map: Dict[str, List[str]] = {}
        for word in self.doc_freqs.keys():
            if len(word) < 2:
                continue
            # Delimited sub-tokens (e.g., "rclcpp-debug", "nav2_bringup", "simulation.py")
            parts = re.split(r'[-_.]', word)
            if len(parts) > 1:
                for p in parts:
                    p_clean = p.strip().lower()
                    if len(p_clean) >= 2:
                        if p_clean not in self.boundary_prefix_map:
                            self.boundary_prefix_map[p_clean] = []
                        self.boundary_prefix_map[p_clean].append(word)
            # Base prefix with trailing digits (e.g. "qwen2", "bert7b", "rqt2")
            m = re.match(r'^([a-zA-Z]+)(\d+[a-zA-Z0-9]*)$', word)
            if m:
                base_p = m.group(1).lower()
                if len(base_p) >= 2:
                    if base_p not in self.boundary_prefix_map:
                        self.boundary_prefix_map[base_p] = []
                    self.boundary_prefix_map[base_p].append(word)

    def get_boundary_candidates(self, anchor: str) -> List[str]:
        """Returns pre-indexed compound candidate matches in O(1) time."""
        return self.boundary_prefix_map.get(anchor.lower(), [])

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
            w = words[0]
            if w in self.idf_table:
                return self.idf_table[w]
            # Try analyzer on the word if not found directly
            analyzed = self.analyzer.analyze(w)
            if analyzed and analyzed[0] in self.idf_table:
                return self.idf_table[analyzed[0]]
            # OOV fallback: maximum corpus IDF
            return self.max_idf

        # Bigram constituent mean IDF
        word_idfs = []
        for w in words:
            if w in self.idf_table:
                word_idfs.append(self.idf_table[w])
            else:
                analyzed = self.analyzer.analyze(w)
                if analyzed and analyzed[0] in self.idf_table:
                    word_idfs.append(self.idf_table[analyzed[0]])
                else:
                    word_idfs.append(self.max_idf)

        return sum(word_idfs) / len(word_idfs) if word_idfs else 0.0

    def get_normalized_idf(self, term: Union[str, List[str]]) -> float:
        """Returns term IDF normalized to [0, 1] with OOV clamping."""
        raw_idf = self.get_idf(term)
        return min(1.0, max(0.0, raw_idf / self.max_idf)) if self.max_idf > 0 else 0.0

