import math
import numpy as np
from typing import Dict, Any, List
import yake
import fasttext
from rank_bm25 import BM25Okapi

class StatisticalQueryExpander:
    def __init__(self, 
                 fasttext_model_path: str = "data/models/cc.en.50.bin",
                 idf_dictionary: Dict[str, float] = None,
                 corpus_documents: List[str] = None,
                 idf_dict: Dict[str, float] = None):
        """
        Initializes the Pathway A (Statistical & Lexical) query expander.
        """
        # Load YAKE for Aspect Extraction (Step 1)
        self.yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=20)
        
        # Load FastText for Semantic Clustering (Step 2)
        try:
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
        except Exception:
            self.fasttext_model = None
            
        self.idf_dictionary = idf_dictionary if idf_dictionary is not None else (idf_dict or {})
        self.corpus_documents = corpus_documents or []
        
        # Initialize BM25 for PRF (Step 3)
        if self.corpus_documents:
            tokenized_corpus = [doc.lower().split() for doc in self.corpus_documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
        
    def _get_fasttext_phrase_vector(self, phrase: str) -> np.ndarray:
        """Helper to get a bag-of-words vector for a phrase using FastText."""
        if not self.fasttext_model:
            return np.random.rand(300) # Mock vector if model failed to load
        
        words = phrase.split()
        vectors = [self.fasttext_model.get_word_vector(w) for w in words]
        return np.mean(vectors, axis=0)
        
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
        
    def expand(self, query: str, K: int = 5) -> Dict[str, Any]:
        """
        Executes the 6-step Pathway A expansion logic.
        """
        # 1. Aspect Extraction
        extracted = self.yake_extractor.extract_keywords(query)
        raw_aspects = [kw[0].lower() for kw in extracted]
        
        # 2. Orthogonalization (Lexical Subsumption + Semantic Clustering)
        
        # 2a. Global Lexical Subsumption
        # YAKE often extracts overlapping n-grams (e.g., "adobe creative cloud", "creative cloud").
        # We sort by length descending and drop any phrase that is a substring of a longer kept phrase.
        lexically_filtered = []
        sorted_raw = sorted(raw_aspects, key=len, reverse=True)
        for aspect in sorted_raw:
            is_subsumed = False
            for existing in lexically_filtered:
                if aspect in existing:
                    is_subsumed = True
                    break
            if not is_subsumed:
                lexically_filtered.append(aspect)
        
        # 2b. Semantic Clustering
        aspect_vectors = {asp: self._get_fasttext_phrase_vector(asp) for asp in lexically_filtered}
        
        semantic_clusters = []
        assigned = set()
        
        for i, asp1 in enumerate(lexically_filtered):
            if asp1 in assigned:
                continue
                
            current_cluster = [asp1]
            assigned.add(asp1)
            
            for j, asp2 in enumerate(lexically_filtered[i+1:]):
                if asp2 in assigned:
                    continue
                
                sim = self._cosine_similarity(aspect_vectors[asp1], aspect_vectors[asp2])
                if sim > 0.85:
                    current_cluster.append(asp2)
                    assigned.add(asp2)
                    
            semantic_clusters.append(current_cluster)
            
        # 2c. Representative Selection
        orthogonal_aspects = []
        for cluster in semantic_clusters:
            # Keep the longest phrase as the cluster representative
            best_aspect = max(cluster, key=len)
            orthogonal_aspects.append(best_aspect)
            
        if not orthogonal_aspects and query.strip():
            orthogonal_aspects = [query]
            
        # 3. Keyword Exploration (PRF via BM25)
        aspect_data = []
        for aspect in orthogonal_aspects:
            expanded_keywords = [{"term": aspect, "weight": 1.0}]
            
            if self.corpus_documents:
                tokenized_query = aspect.split()
                scores = self.bm25.get_scores(tokenized_query)
                
                # Only consider documents with a positive BM25 score
                top_doc_indices = np.argsort(scores)[::-1]
                top_docs = []
                for idx in top_doc_indices:
                    if scores[idx] > 0:
                        top_docs.append(self.corpus_documents[idx])
                    if len(top_docs) == 3:
                        break
                
                if top_docs:
                    # Extract PRF terms from these top docs (simplified TF extraction)
                    term_freqs = {}
                    for doc in top_docs:
                        for token in doc.lower().split():
                            if token not in aspect.split() and len(token) > 3: # Basic filter
                                term_freqs[token] = term_freqs.get(token, 0) + 1
                    
                    if term_freqs:
                        # Take top 3 PRF terms, normalize frequencies to [0,1] for weights
                        sorted_prf = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)[:3]
                        max_freq = sorted_prf[0][1]
                        
                        for term, freq in sorted_prf:
                            expanded_keywords.append({"term": term, "weight": round(freq / max_freq, 2)})
                    
            aspect_data.append({"term": aspect, "keywords": expanded_keywords})
            
        # 4. Weight Assignment (IDF)
        max_idf = max(self.idf_dictionary.values()) if self.idf_dictionary else 1.0
        
        for item in aspect_data:
            # We look up the IDF. For multi-word aspects, we might take the max or average IDF of its words.
            # Here we just do a simple lookup, falling back to a default low weight.
            words = item["term"].split()
            idfs = [self.idf_dictionary.get(w, 0.1) for w in words]
            avg_idf = sum(idfs) / len(idfs) if idfs else 0.1
            
            # Normalize against max_idf in corpus
            normalized_weight = round(avg_idf / max_idf, 2)
            item["aspect_weight"] = normalized_weight
            
        # 5. Top-K Pruning
        aspect_data.sort(key=lambda x: x["aspect_weight"], reverse=True)
        top_k = aspect_data[:K]
        
        # 6. Formatting
        final_payload = {"aspects": []}
        for item in top_k:
            final_payload["aspects"].append({
                "name": item["term"],
                "aspect_weight": item["aspect_weight"],
                "keywords": item["keywords"]
            })
            
        return final_payload
