import math
import re
import random
from collections import Counter
from typing import List, Set, Optional, Tuple, Dict, Any
from .corpus_idf_registry import CorpusIDFRegistry
from .analyzer import EdgeRAGAnalyzer, LUCENE_STOPWORDS
from .tokenizer import EdgeRAGTokenizer


class CorpusVocabBuilder:
    """
    O(N) High-Speed Analyzed Token & Salience Pool Builder (V7 Design).
    
    Features:
    1. Analyzes corpus via EdgeRAGAnalyzer to ensure 1:1 stem parity.
    2. Retains surface-form mapping (maps analyzed stem -> highest-frequency surface form in corpus)
       to ensure BGE embedding is evaluated on natural words rather than truncated stem artifacts.
    3. Operates across the full stem vocabulary without legacy pre-truncation.
    4. Supports selection strategies: 'coverage' (FPS), 'salience' (IDF * ln(1+DF)), 'idf', 'random'.
    """

    CLEAN_PATTERN = re.compile(r'^[a-zA-Z0-9\-_]+$')

    def __init__(
        self,
        idf_registry: CorpusIDFRegistry,
        max_vocab_size: int = 50000,
        analyzer: Optional[EdgeRAGAnalyzer] = None
    ):
        self.idf_registry = idf_registry
        self.max_vocab_size = max_vocab_size
        self.analyzer = analyzer if analyzer is not None else EdgeRAGAnalyzer()
        self.stem_to_surface: Dict[str, str] = {}

    def extract_candidates_with_surface_forms(
        self,
        corpus: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts all valid analyzed candidate stems along with their canonical surface forms.
        
        Returns:
            Tuple of (candidate_stems, canonical_surface_forms).
        """
        num_docs = self.idf_registry.num_docs if self.idf_registry.num_docs > 0 else len(corpus)
        max_doc_freq = max(5, int(0.15 * num_docs))

        # Track surface token frequencies per stem
        stem_surface_counts: Dict[str, Counter] = {}

        # Scan documents (sample up to 5000 chunks with deterministic seed for representative surface form extraction)
        rng = random.Random(42)
        sample_corpus = rng.sample(corpus, min(len(corpus), 5000)) if len(corpus) > 5000 else corpus
        for doc in sample_corpus:
            raw_tokens = EdgeRAGTokenizer.tokenize(doc)
            for raw_tok in raw_tokens:
                if len(raw_tok) < 2 or raw_tok in LUCENE_STOPWORDS:
                    continue
                # Analyze single token
                analyzed = self.analyzer.analyze(raw_tok)
                if not analyzed:
                    continue
                stem = analyzed[0]
                if stem not in stem_surface_counts:
                    stem_surface_counts[stem] = Counter()
                stem_surface_counts[stem][raw_tok] += 1

        candidate_tuples = []

        for stem, df in self.idf_registry.doc_freqs.items():
            if df > max_doc_freq or df < 1:
                continue
            if len(stem) < 2 or stem.isdigit() or stem in LUCENE_STOPWORDS:
                continue
            if not self.CLEAN_PATTERN.match(stem):
                continue

            # Pick highest-frequency surface form if available, else fallback to stem
            if stem in stem_surface_counts and stem_surface_counts[stem]:
                best_surface = stem_surface_counts[stem].most_common(1)[0][0]
            else:
                best_surface = stem

            self.stem_to_surface[stem] = best_surface
            idf_val = self.idf_registry.get_idf(stem)
            salience = idf_val * math.log(1.0 + df)
            candidate_tuples.append((stem, best_surface, salience))

        if self.max_vocab_size is not None and len(candidate_tuples) > self.max_vocab_size:
            candidate_tuples.sort(key=lambda c: -c[2])
            candidate_tuples = candidate_tuples[:self.max_vocab_size]

        candidate_stems = [c[0] for c in candidate_tuples]
        canonical_surfaces = [c[1] for c in candidate_tuples]

        return candidate_stems, canonical_surfaces

    def extract_candidates_from_indexer(
        self,
        indexer: Any
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts all valid analyzed candidate stems along with their canonical surface forms
        in O(V) time using the 1-pass surface map already collected by BM25LuceneIndexer.
        """
        num_docs = self.idf_registry.num_docs if self.idf_registry.num_docs > 0 else 100
        max_doc_freq = max(5, int(0.15 * num_docs))
        stem_to_surface = getattr(indexer, "stem_to_surface", {})

        candidate_tuples = []
        for stem, df in self.idf_registry.doc_freqs.items():
            if df > max_doc_freq or df < 1:
                continue
            if len(stem) < 2 or stem.isdigit() or stem in LUCENE_STOPWORDS:
                continue
            if not self.CLEAN_PATTERN.match(stem):
                continue

            best_surface = stem_to_surface.get(stem, stem)
            self.stem_to_surface[stem] = best_surface
            idf_val = self.idf_registry.get_idf(stem)
            salience = idf_val * math.log(1.0 + df)
            candidate_tuples.append((stem, best_surface, salience))

        if self.max_vocab_size is not None and len(candidate_tuples) > self.max_vocab_size:
            candidate_tuples.sort(key=lambda c: -c[2])
            candidate_tuples = candidate_tuples[:self.max_vocab_size]

        candidate_stems = [c[0] for c in candidate_tuples]
        canonical_surfaces = [c[1] for c in candidate_tuples]
        return candidate_stems, canonical_surfaces

    def build_clean_vocabulary(
        self,
        corpus: List[str],
        strategy: str = "coverage",
        seed: int = 42,
        vocab_matrix: Optional[Any] = None
    ) -> List[str]:
        """
        Builds candidate vocabulary pool using the requested strategy.
        
        Strategies:
        - 'coverage': FPS semantic hub selection on DenseVocabMatrix
        - 'salience': Score(t) = IDF(t) * ln(1 + DF(t))
        - 'idf': Pure IDF descending
        - 'random': Uniform random sample
        """
        candidate_stems, canonical_surfaces = self.extract_candidates_with_surface_forms(corpus)
        if not candidate_stems:
            return []

        if len(candidate_stems) <= self.max_vocab_size:
            return candidate_stems

        if strategy == "coverage" and vocab_matrix is not None:
            vocab_matrix.build_with_fps(
                candidate_stems,
                surface_forms=canonical_surfaces,
                target_pool_size=self.max_vocab_size
            )
            return vocab_matrix.vocab_terms

        if strategy == "idf":
            scored = [(s, self.idf_registry.get_idf(s)) for s in candidate_stems]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]

        elif strategy == "random":
            rng = random.Random(seed)
            return rng.sample(candidate_stems, self.max_vocab_size)

        else:  # 'salience' default fallback
            scored = [
                (s, self.idf_registry.get_idf(s) * math.log(1.0 + self.idf_registry.doc_freqs.get(s, 1)))
                for s in candidate_stems
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]

    def _select_pool(self, candidate_stems: List[str], strategy: str, seed: int) -> List[str]:
        """Selects the top-N pool from candidate_stems by strategy (salience/idf/random)."""
        if strategy == "idf":
            scored = [(s, self.idf_registry.get_idf(s)) for s in candidate_stems]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]
        elif strategy == "random":
            rng = random.Random(seed)
            return rng.sample(candidate_stems, self.max_vocab_size)
        else:  # salience
            scored = [
                (s, self.idf_registry.get_idf(s) * math.log(1.0 + self.idf_registry.doc_freqs.get(s, 1)))
                for s in candidate_stems
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]

    def build_pool_with_full(
        self,
        corpus: List[str],
        strategy: str = "salience",
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Builds the N-selected pool AND returns the full filtered corpus vocabulary.

        Returns:
            (pool_stems, full_stems, full_surfaces):
              - pool_stems: top-N selection by strategy (salience/idf/random).
              - full_stems / full_surfaces: the complete filtered candidate vocabulary
                (aligned) for full-corpus storage in DenseVocabMatrix.
        """
        candidate_stems, canonical_surfaces = self.extract_candidates_with_surface_forms(corpus)
        if not candidate_stems:
            return [], [], []

        if len(candidate_stems) <= self.max_vocab_size:
            pool_stems = list(candidate_stems)
        else:
            pool_stems = self._select_pool(candidate_stems, strategy, seed)
        return pool_stems, candidate_stems, canonical_surfaces
