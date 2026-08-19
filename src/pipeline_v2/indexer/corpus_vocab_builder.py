import string
from collections import Counter
from typing import List, Set, Optional
from .corpus_idf_registry import CorpusIDFRegistry


class CorpusVocabBuilder:
    """
    O(N) High-Speed Token Counter & Median IDF Pre-Filter (V5 Design).
    Uses corpus sampling for bigram extraction (<0.05s) and reads unigrams from idf_registry.
    """

    ENGLISH_STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
        "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
        "through", "about", "against", "between", "into", "throughout", "during", "before",
        "after", "above", "below", "to", "from", "up", "upon", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
    }

    TRANS_TABLE = str.maketrans('', '', string.punctuation.replace('-', '').replace('_', ''))

    def __init__(self, idf_registry: CorpusIDFRegistry, max_vocab_size: int = 1000):
        self.idf_registry = idf_registry
        self.max_vocab_size = max_vocab_size

    def build_clean_vocabulary(self, corpus: List[str]) -> List[str]:
        """
        Sub-second unigram & bigram extraction (<0.05s):
        1. Pulls unigram doc_freqs directly from idf_registry.doc_freqs (0ms).
        2. Filters out generic corpus stopwords (doc_freq > 15% of docs).
        3. Samples up to 1,000 corpus chunks for fast bigram counting (<0.03s).
        4. Applies Sublinear Salience Ranking: Score(t) = IDF(t) * ln(1 + doc_freq(t)).
        """
        import math
        import re
        stopwords = self.ENGLISH_STOPWORDS
        num_docs = self.idf_registry.num_docs if self.idf_registry.num_docs > 0 else len(corpus)
        max_doc_freq = max(5, int(0.15 * num_docs))

        clean_pattern = re.compile(r'^[a-zA-Z0-9\-_ ]+$')

        # 1. Top unigrams directly from idf_registry.doc_freqs
        valid_unigrams = [
            (term, count) for term, count in self.idf_registry.doc_freqs.items()
            if 2 <= count <= max_doc_freq and term not in stopwords and not term.isdigit() and len(term) >= 3 and clean_pattern.match(term)
        ]
        valid_unigrams.sort(key=lambda x: x[1], reverse=True)
        top_unigrams = [(term, count) for term, count in valid_unigrams[:2500]]

        # 2. Fast bigram counter from sampled corpus subset (max 1000 chunks)
        sample_corpus = corpus[:1000] if len(corpus) > 1000 else corpus
        bigram_counter = Counter()

        for doc in sample_corpus:
            clean_doc = doc.lower().translate(self.TRANS_TABLE)
            clean_tokens = [w for w in clean_doc.split() if len(w) >= 3 and w not in stopwords and not w.isdigit() and clean_pattern.match(w)]
            if len(clean_tokens) > 1:
                bigram_counter.update(zip(clean_tokens, clean_tokens[1:]))

        top_bigrams = [
            (f"{t1} {t2}", count) for (t1, t2), count in bigram_counter.most_common(1000)
            if 2 <= count <= max_doc_freq and clean_pattern.match(f"{t1} {t2}")
        ]
        candidate_terms = top_unigrams + top_bigrams

        # 3. Apply Sublinear Salience Ranking
        surviving_terms = []
        for term, count in candidate_terms:
            idf_val = self.idf_registry.get_idf(term)
            if idf_val >= 1.5:  # Filter out true generic stopwords
                salience = idf_val * math.log(1.0 + count)
                surviving_terms.append((term, salience))

        surviving_terms.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in surviving_terms[:self.max_vocab_size]]
