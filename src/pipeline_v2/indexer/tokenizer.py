"""
src/pipeline_v2/indexer/tokenizer.py

Canonical Edge-RAG Tokenizer.
Serves as the single source of truth across:
- BM25LuceneIndexer / AnalyzedLuceneBM25
- CorpusVocabBuilder
- CorpusIDFRegistry
- BM25DenseAspectExtractor

Guarantees 1:1 token boundary parity between document indexing and query extraction.
"""

import re
from typing import List

class EdgeRAGTokenizer:
    """
    Canonical Tokenizer for Edge-RAG.
    
    1. Splits on standard delimiter punctuation (whitespace, commas, quotes, parentheses, slashes).
    2. Protects alphanumeric version strings and technical compounds intact
       (e.g., 'qwen2.5-7b', 'gpt-4', 'llama-3.1', 'fp16', 'zero-shot').
    3. Retains vital 2-letter technical acronyms ('ai', 'ml', 'db', 'kv', 'ip').
    """
    # Regex: Alphanumeric compounds with internal hyphen/dot/underscore OR single words >= 2 chars
    TOKEN_PATTERN = re.compile(r'\b[a-z0-9]+(?:[-._][a-z0-9]+)+\b|\b[a-z0-9]{2,}\b', re.IGNORECASE)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Tokenizes text into canonical lowercase tokens.
        
        Args:
            text: Raw input string.
            
        Returns:
            List of lowercased canonical tokens.
        """
        if not text or not isinstance(text, str):
            return []
        return [t.lower() for t in cls.TOKEN_PATTERN.findall(text)]
