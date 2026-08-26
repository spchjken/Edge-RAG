"""
src/pipeline_v2/indexer/analyzer.py

Pure-Python Lucene-Parity Analyzer Chain for Edge-RAG.

Implements the 5-stage pipeline:
1. EdgeRAGTokenizer (canonical tokenization)
2. PossessiveFilter (Lucene EnglishPossessiveFilter semantics: strips trailing "'s")
3. KeywordMarkerExemption (protects technical compounds, digits, and acronyms)
4. StopwordFilter (Lucene standard English stop set)
5. StemFilter (KStem / Krovetz light stemming with graceful Snowball / pure-Python fallback)
"""

import re
from typing import List, Set, Optional, Callable, Tuple
from .tokenizer import EdgeRAGTokenizer

# Standard Lucene English stopword set (~33 words)
LUCENE_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

# Regex to identify protected technical compound patterns that MUST NEVER be stemmed or stopword-dropped
TECHNICAL_PATTERN = re.compile(r'^[a-z0-9]+(?:[-._][a-z0-9]+)+$|^[a-z0-9]*\d+[a-z0-9]*$', re.IGNORECASE)


def _init_stemmer(stemmer_name: str = "kstem") -> Tuple[Callable[[str], str], str]:
    """
    Initializes the requested stemmer and returns (stem_fn, resolved_name).
    Strictly forbids silent fallback to Porter.
    """
    if stemmer_name == "kstem":
        try:
            import krovetzstemmer
            k_stem = krovetzstemmer.Stemmer()
            return (lambda w: k_stem.stem(w)), "krovetzstemmer (KStem 0.8)"
        except ImportError as e:
            raise RuntimeError(
                "CRITICAL: stemmer='kstem' requested but 'krovetzstemmer' is not installed! "
                "Silent fallback to Porter is strictly forbidden. Run: pip install KrovetzStemmer"
            ) from e
    elif stemmer_name == "snowball":
        try:
            from nltk.stem import SnowballStemmer
            snowball = SnowballStemmer("english")
            return (lambda w: snowball.stem(w)), "nltk.SnowballStemmer"
        except ImportError as e:
            raise RuntimeError("stemmer='snowball' requested but nltk is not available") from e
    elif stemmer_name == "porter":
        try:
            from nltk.stem import PorterStemmer
            porter = PorterStemmer()
            return (lambda w: porter.stem(w)), "nltk.PorterStemmer"
        except ImportError as e:
            raise RuntimeError("stemmer='porter' requested but nltk is not available") from e
    elif not stemmer_name or stemmer_name == "none":
        return (lambda w: w), "none"
    else:
        raise ValueError(f"Unknown stemmer: {stemmer_name}")


class EdgeRAGAnalyzer:
    """
    Lucene-Parity Analyzer for Edge-RAG.
    Produces identical analyzed token streams for documents and queries.
    """

    def __init__(
        self,
        stemmer: str = "kstem",
        use_stopwords: bool = True,
        exempt_technical: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.use_stopwords = use_stopwords
        self.exempt_technical = exempt_technical
        self.stopwords = custom_stopwords if custom_stopwords is not None else LUCENE_STOPWORDS
        self.stemmer_name = stemmer
        self.stem_fn, self.resolved_stemmer = _init_stemmer(stemmer)

    def is_stem_exempt(self, token: str) -> bool:
        """Determines if a token is protected from stemming."""
        if not self.exempt_technical:
            return False
        # Do not stem short tokens (len <= 3), compounds (qwen2.5-7b, gpt-4, fp16), or strings with digits
        if len(token) <= 3:
            return True
        return bool(TECHNICAL_PATTERN.match(token))

    def analyze(self, text: str) -> List[str]:
        """
        Runs the full analysis pipeline on text.
        
        Args:
            text: Raw input string.
            
        Returns:
            List of analyzed, normalized tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Canonical Tokenization
        raw_tokens = EdgeRAGTokenizer.tokenize(text)
        analyzed_tokens: List[str] = []

        for token in raw_tokens:
            # 2. Possessive Filter (strips trailing 's)
            if token.endswith("'s") and len(token) > 2:
                token = token[:-2]

            # 3. Stopword Filter (drops standard English stopwords unless it is a compound with hyphen/dot/digit)
            if self.use_stopwords and token in self.stopwords:
                if not ("-" in token or "." in token or "_" in token or any(c.isdigit() for c in token)):
                    continue

            # 4. Stem Filter (applied to non-exempt tokens)
            if not self.is_stem_exempt(token):
                token = self.stem_fn(token)

            if token:
                analyzed_tokens.append(token)

        return analyzed_tokens

