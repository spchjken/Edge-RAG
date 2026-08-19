import re
import numpy as np
import yake
from typing import List, Dict, Any, Optional, Callable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Using scikit-learn's standard, trusted English stopword list (318 words)
STOPWORDS = set(ENGLISH_STOP_WORDS)

class YAKEAspectExtractor:
    """
    Extracts a list of orthogonal aspects and their weights from a query using YAKE.
    Implements Acronym Recovery to prevent dropping critical constraints (like "AI").
    Supports two weighting variants: 'statistical' (Global IDF) and 'vector' (Semantic Distance).
    """
    def __init__(self, 
                 weighting_strategy: str = "statistical",
                 idf_dict: Optional[Dict[str, float]] = None,
                 embedding_fn: Optional[Callable[[str], np.ndarray]] = None):
        """
        weighting_strategy: 'statistical' or 'vector'
        idf_dict: Dictionary mapping terms to IDF scores (required for 'statistical')
        embedding_fn: Function mapping a string to a numpy array (required for 'vector')
        """
        self.weighting_strategy = weighting_strategy
        self.idf_dict = idf_dict
        self.embedding_fn = embedding_fn
        
        if weighting_strategy == "statistical" and idf_dict is None:
            raise ValueError("idf_dict must be provided for statistical weighting")
        if weighting_strategy == "vector" and embedding_fn is None:
            raise ValueError("embedding_fn must be provided for vector weighting")

        # We configure YAKE but bypass its internal stopword logic 
        # by pre-filtering the query to ensure maximum purity.
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=3, 
            dedupLim=0.9, 
            dedupFunc='seqm', 
            windowsSize=1, 
            top=10, 
            features=None
        )

    def extract(self, query: str) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # Step 1: Pre-processing & Acronym Recovery
        # Extract fully capitalized words of 2+ characters (e.g. AI, API, SDK, CMO)
        acronyms = set(re.findall(r'\b[A-Z]{2,}\b', query))
        must_include = [acr.lower() for acr in acronyms]

        # Filter stopwords from the query *before* feeding to YAKE 
        # to guarantee no stopwords leak into the final n-grams
        words = query.split()
        filtered_words = [w for w in words if w.lower() not in STOPWORDS]
        clean_query = " ".join(filtered_words)
        
        if not clean_query.strip():
            clean_query = query # Fallback if the query was entirely stopwords

        # Step 2: YAKE N-Gram Extraction
        yake_results = self.kw_extractor.extract_keywords(clean_query)
        yake_aspects = [kw.lower() for kw, score in yake_results]

        # Step 3: Aspect Merging
        # Combine recovered acronyms with YAKE results
        merged_aspects = list(set(must_include + yake_aspects))

        # Step 4: Global Lexical Subsumption (Orthogonalization)
        # Sort by length descending to allow longer phrases to consume substrings
        merged_aspects.sort(key=len, reverse=True)
        
        orthogonal_aspects = []
        for aspect in merged_aspects:
            # Check if this aspect is a substring of any already accepted aspect
            is_subsumed = False
            for accepted in orthogonal_aspects:
                # Add word boundaries so "ai" doesn't get subsumed by "aim"
                if re.search(r'\b' + re.escape(aspect) + r'\b', accepted):
                    is_subsumed = True
                    break
            
            if not is_subsumed:
                orthogonal_aspects.append(aspect)

        if not orthogonal_aspects:
            return []

        # Step 5: Weight Calculation
        if self.weighting_strategy == "statistical":
            return self._calculate_statistical_weights(orthogonal_aspects)
        elif self.weighting_strategy == "vector":
            return self._calculate_vector_weights(query, orthogonal_aspects)
            
        return []

    def _calculate_statistical_weights(self, aspects: List[str]) -> List[Dict[str, Any]]:
        raw_weights = []
        for aspect in aspects:
            words = aspect.split()
            # Default IDF for unknown words (assume they are relatively rare)
            default_idf = 5.0 
            
            idfs = [self.idf_dict.get(w, default_idf) for w in words]
            # Average IDF of the aspect
            avg_idf = sum(idfs) / len(idfs) if idfs else 0.0
            raw_weights.append(avg_idf)
            
        return self._normalize_weights(aspects, raw_weights)

    def _calculate_vector_weights(self, original_query: str, aspects: List[str]) -> List[Dict[str, Any]]:
        query_vec = self.embedding_fn(original_query)
        
        raw_weights = []
        for aspect in aspects:
            aspect_vec = self.embedding_fn(aspect)
            
            # Cosine similarity
            dot_product = np.dot(query_vec, aspect_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_a = np.linalg.norm(aspect_vec)
            
            if norm_q == 0 or norm_a == 0:
                sim = 0.0
            else:
                sim = float(dot_product / (norm_q * norm_a))
                
            raw_weights.append(sim)
            
        return self._normalize_weights(aspects, raw_weights)

    def _normalize_weights(self, aspects: List[str], raw_weights: List[float]) -> List[Dict[str, Any]]:
        """Min-Max scaler to normalize weights between 0.1 and 1.0"""
        if not raw_weights:
            return []
            
        min_w = min(raw_weights)
        max_w = max(raw_weights)
        
        results = []
        for aspect, weight in zip(aspects, raw_weights):
            if max_w == min_w:
                norm_w = 1.0
            else:
                norm_w = (weight - min_w) / (max_w - min_w)
                
            # Floor at 0.1 so valid constraints don't get zeroed out completely
            norm_w = max(0.1, norm_w)
            
            # Ensure the top aspect is exactly 1.0
            if weight == max_w:
                norm_w = 1.0
                
            results.append({
                "name": aspect,
                "weight": round(norm_w, 2)
            })
            
        # Sort by weight descending
        results.sort(key=lambda x: x["weight"], reverse=True)
        return results
