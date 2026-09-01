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
from typing import List, Set, Optional, Callable, Tuple, Dict
from .tokenizer import EdgeRAGTokenizer

# Standard Lucene English stopword set (~33 words)
LUCENE_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

# Regex to identify protected technical compound patterns that MUST NEVER be stemmed or stopword-dropped
TECHNICAL_PATTERN = re.compile(r'^[a-z0-9]+(?:[-._][a-z0-9]+)+$|^[a-z0-9]*\d+[a-z0-9]*$', re.IGNORECASE)

# Comprehensive WordNet suppletion and irregular morphological overrides
WORDNET_SUPPLETION_OVERRIDES: Dict[str, str] = {
    # Verbs (Suppletion & Irregulars)
    "went": "go", "gone": "go",
    "bought": "buy", "brought": "bring", "caught": "catch", "taught": "teach",
    "thought": "think", "sought": "seek", "fought": "fight", "sold": "sell",
    "told": "tell", "held": "hold", "stood": "stand", "understood": "understand",
    "sat": "sit", "lost": "lose", "felt": "feel", "met": "meet", "led": "lead",
    "fed": "feed", "wrote": "write", "written": "write", "drew": "draw",
    "drawn": "draw", "flew": "fly", "flown": "fly", "grew": "grow", "grown": "grow",
    "blew": "blow", "blown": "blow", "knew": "know", "known": "know", "threw": "throw",
    "thrown": "throw", "swam": "swim", "swum": "swim", "began": "begin", "begun": "begin",
    "ran": "run", "sang": "sing", "sung": "sing", "rang": "ring", "rung": "ring",
    "sank": "sink", "sunk": "sink", "spoke": "speak", "spoken": "speak", "broke": "break",
    "broken": "break", "chose": "choose", "chosen": "choose", "froze": "freeze",
    "frozen": "freeze", "woke": "wake", "woken": "wake", "stole": "steal", "stolen": "steal",
    "took": "take", "taken": "take", "shook": "shake", "shaken": "shake", "gave": "give",
    "given": "give", "saw": "see", "seen": "see", "ate": "eat", "eaten": "eat",
    "fell": "fall", "fallen": "fall", "drove": "drive", "driven": "drive", "rode": "ride",
    "ridden": "ride", "rose": "rise", "risen": "rise", "arose": "arise", "arisen": "arise",
    "hid": "hide", "hidden": "hide", "bit": "bite", "bitten": "bite", "forgave": "forgive",
    "forgiven": "forgive", "forgot": "forget", "forgotten": "forget", "bore": "bear",
    "born": "bear", "borne": "bear", "wore": "wear", "worn": "wear", "tore": "tear",
    "torn": "tear", "swore": "swear", "sworn": "swear", "laid": "lay", "paid": "pay",
    "said": "say", "sent": "send", "spent": "spend", "lent": "lend", "built": "build",
    "bent": "bend", "meant": "mean", "dealt": "deal", "slept": "sleep", "kept": "keep",
    "wept": "weep", "swept": "sweep", "left": "leave", "heard": "hear", "found": "find",
    "bound": "bind", "wound": "wind", "struck": "strike", "stricken": "strike", "dug": "dig",
    "hung": "hang", "spun": "spin", "stung": "sting", "swung": "swing", "wrung": "wring",
    "clung": "cling", "flung": "fling", "slung": "sling", "shone": "shine", "lit": "light",
    "shot": "shoot", "came": "come", "became": "become", "overcame": "overcome",

    # Adjectives & Adverbs (Suppletion)
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "farther": "far", "further": "far", "farthest": "far", "furthest": "far",
    "less": "little", "least": "little", "lesser": "little",

    # Nouns (Irregular Plurals)
    "children": "child", "men": "man", "women": "woman", "mice": "mouse",
    "feet": "foot", "teeth": "tooth", "geese": "goose", "oxen": "ox",
    "people": "person", "criteria": "criterion", "phenomena": "phenomenon",
    "corpora": "corpus", "indices": "index", "matrices": "matrix", "vertices": "vertex",
    "appendices": "appendix", "analyses": "analysis", "hypotheses": "hypothesis",
    "theses": "thesis", "syntheses": "synthesis", "parentheses": "parenthesis",
    "diagnoses": "diagnosis", "crises": "crisis"
}


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
    
    6-stage pipeline:
    1. EdgeRAGTokenizer (canonical tokenization)
    2. PossessiveFilter (Lucene EnglishPossessiveFilter semantics: strips trailing "'s")
    3. KeywordMarkerExemption (protects technical compounds, digits, and acronyms)
    4. StopwordFilter (Lucene standard English stop set)
    5. StemmerOverride (WordNet suppletion: went -> go, better -> good, children -> child)
    6. StemFilter (KStem / Krovetz light stemming)
    """

    def __init__(
        self,
        stemmer: str = "kstem",
        use_stopwords: bool = True,
        exempt_technical: bool = True,
        use_wordnet_override: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
        custom_overrides: Optional[Dict[str, str]] = None
    ):
        self.use_stopwords = use_stopwords
        self.exempt_technical = exempt_technical
        self.use_wordnet_override = use_wordnet_override
        self.stopwords = custom_stopwords if custom_stopwords is not None else LUCENE_STOPWORDS
        self.overrides = custom_overrides if custom_overrides is not None else WORDNET_SUPPLETION_OVERRIDES
        self.stemmer_name = stemmer
        self.stem_fn, self.resolved_stemmer = _init_stemmer(stemmer)

    def is_stem_exempt(self, token: str) -> bool:
        """Determines if a token is protected from stemming and overrides."""
        if not self.exempt_technical:
            return False
        # Protect compounds (qwen2.5-7b, gpt-4, fp16), strings with digits, or technical patterns
        if "-" in token or "." in token or "_" in token or any(c.isdigit() for c in token):
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

            # 4. StemmerOverride Filter (WordNet irregular suppletion overrides)
            if self.use_wordnet_override and token in self.overrides:
                token = self.overrides[token]
                analyzed_tokens.append(token)
                continue

            # 5. Technical Keyword Marker Exemption
            if self.is_stem_exempt(token):
                if token:
                    analyzed_tokens.append(token)
                continue

            # 6. Stem Filter (applied to standard words)
            token = self.stem_fn(token)

            if token:
                analyzed_tokens.append(token)

        return analyzed_tokens

    def analyze_with_surface(self, text: str) -> List[Tuple[str, str]]:
        """
        Runs the full analysis pipeline while preserving the raw surface form for each token.
        
        Returns:
            List of (analyzed_stem, raw_surface_token) tuples.
        """
        if not text or not isinstance(text, str):
            return []

        raw_tokens = EdgeRAGTokenizer.tokenize(text)
        analyzed_pairs: List[Tuple[str, str]] = []

        for token in raw_tokens:
            surface = token
            if token.endswith("'s") and len(token) > 2:
                token = token[:-2]

            if self.use_stopwords and token in self.stopwords:
                if not ("-" in token or "." in token or "_" in token or any(c.isdigit() for c in token)):
                    continue

            if self.use_wordnet_override and token in self.overrides:
                analyzed_pairs.append((self.overrides[token], surface))
                continue

            if self.is_stem_exempt(token):
                if token:
                    analyzed_pairs.append((token, surface))
                continue

            stem = self.stem_fn(token)
            if stem:
                analyzed_pairs.append((stem, surface))

        return analyzed_pairs

