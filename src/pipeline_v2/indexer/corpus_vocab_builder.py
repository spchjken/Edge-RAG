import math
import re
import random
from collections import Counter
from typing import List, Set, Optional, Tuple, Dict
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
        max_vocab_size: int = 2500,
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

        # Scan documents (sample up to 2000 chunks for fast surface form extraction)
        sample_corpus = corpus[:2000] if len(corpus) > 2000 else corpus
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

        # Sort by salience descending and cap candidate pool to 2x target pool size (max 5000) for sub-second BGE encoding
        max_candidates = max(self.max_vocab_size * 2, 5000)
        candidate_tuples.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidate_tuples[:max_candidates]

        candidate_stems = [c[0] for c in top_candidates]
        canonical_surfaces = [c[1] for c in top_candidates]

        return candidate_stems, canonical_surfaces

    def build_clean_vocabulary(
        self,
        corpus: List[str],
        strategy: str = "salience",
        seed: int = 42
    ) -> List[str]:
        """
        Builds candidate vocabulary pool using the requested strategy.
        
        Strategies:
        - 'salience': Score(t) = IDF(t) * ln(1 + DF(t))
        - 'idf': Pure IDF descending
        - 'random': Uniform random sample
        """
        candidate_stems, _ = self.extract_candidates_with_surface_forms(corpus)
        if not candidate_stems:
            return []

        if len(candidate_stems) <= self.max_vocab_size:
            return candidate_stems

        if strategy == "idf":
            scored = [(s, self.idf_registry.get_idf(s)) for s in candidate_stems]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]

        elif strategy == "random":
            rng = random.Random(seed)
            return rng.sample(candidate_stems, min(self.max_vocab_size, len(candidate_stems)))

        else: # Default: 'salience'
            scored = []
            for s in candidate_stems:
                idf_val = self.idf_registry.get_idf(s)
                df = self.idf_registry.doc_freqs.get(s, 1)
                salience = idf_val * math.log(1.0 + df)
                scored.append((s, salience))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in scored[:self.max_vocab_size]]

