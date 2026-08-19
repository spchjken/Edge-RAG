import re
import numpy as np
import yake
import json
from typing import List, Dict, Any, Callable
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

class DenseVocabularyExtractor:
    """
    Version 1: Extracts a list of grounded aspects (keywords) using Dense Vocabulary Routing.
    It builds a global vocabulary from the corpus, embeds it, and semantically matches 
    it against the user query using cosine similarity.
    Includes Heuristic Entity Injection (Regex) for critical keywords.
    """
    def __init__(self, embedding_fn: Callable[[str], np.ndarray], embedding_fn_batch: Callable[[List[str]], np.ndarray] = None):
        self.embedding_fn = embedding_fn
        self.embedding_fn_batch = embedding_fn_batch or (lambda terms: np.array([embedding_fn(t) for t in terms]))
        self.vocab_vectors: Dict[str, np.ndarray] = {}
        
        # Configure YAKE for extraction of top unigrams and bigrams
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=2, 
            dedupLim=0.9, 
            dedupFunc='seqm', 
            windowsSize=1, 
            top=3000, 
            features=None
        )

    def build_vocabulary(self, corpus_text: str, limit: int = 5000):
        if not corpus_text or not corpus_text.strip():
            return
            
        # 1. Clean text and filter stopwords
        words = corpus_text.split()
        filtered_words = [w for w in words if w.lower() not in STOPWORDS]
        clean_corpus = " ".join(filtered_words)
        
        if not clean_corpus.strip():
            return
            
        # 2. Extract n-grams using YAKE
        yake_results = self.kw_extractor.extract_keywords(clean_corpus)
        yake_terms = [kw.lower() for kw, score in yake_results]
        
        # 3. Extract high-frequency unigrams as a fallback baseline
        word_counts = Counter([w.lower() for w in filtered_words if re.match(r'^[a-zA-Z]+$', w)])
        top_freq_words = [w for w, c in word_counts.most_common(limit)]
        
        # 4. Combine and deduplicate
        combined_vocab = set(yake_terms + top_freq_words)
        combined_vocab_list = list(combined_vocab)[:limit]
        
        # 5. Embed the vocabulary
        self.vocab_vectors = {}
        for term in combined_vocab_list:
            self.vocab_vectors[term] = self.embedding_fn(term)
            
    def _extract_heuristics(self, query: str) -> List[str]:
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        hyphenated = re.findall(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b', query)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)
        quoted = re.findall(r'"([^"]+)"', query)
        quoted.extend(re.findall(r"'([^']+)'", query))
        
        entities = []
        seen = set()
        for ent in acronyms + hyphenated + proper_nouns + quoted:
            ent_clean = ent.strip().lower()
            if ent_clean and ent_clean not in seen:
                entities.append(ent_clean)
                seen.add(ent_clean)
        return entities

    def extract(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip() or not self.vocab_vectors:
            return []
            
        # Extract forced entities
        heuristic_entities = self._extract_heuristics(query)
        
        # Embed user query
        words = query.split()
        clean_query = " ".join([w for w in words if w.lower() not in STOPWORDS])
        if not clean_query.strip():
            clean_query = query
            
        query_vec = self.embedding_fn(clean_query)
        norm_q = np.linalg.norm(query_vec)
        
        if norm_q == 0:
            return []
            
        # Calculate cosine similarity
        scored_terms = []
        for term, term_vec in self.vocab_vectors.items():
            norm_t = np.linalg.norm(term_vec)
            if norm_t == 0:
                continue
            dot_product = np.dot(query_vec, term_vec)
            sim = float(dot_product / (norm_q * norm_t))
            scored_terms.append((term, sim))
            
        # Sort and select Top-K
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = scored_terms[:top_k]
        
        # Combine
        results = []
        for ent in heuristic_entities:
            results.append({
                "name": ent,
                "weight": 1.0
            })
            
        for term, weight in top_terms:
            if not any(a["name"] == term for a in results):
                results.append({
                    "name": term,
                    "weight": round(weight, 4)
                })
                
        return results[:top_k]


class DenseVocabularyAdvancedExtractor(DenseVocabularyExtractor):
    """
    Version 2: Heuristics + Generic Term Prevention (IDF Scaling).
    """
    def __init__(self, embedding_fn: Callable[[str], np.ndarray], idf_dict: Dict[str, float] = None):
        super().__init__(embedding_fn)
        self.idf_dict = idf_dict

    def extract(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip() or not self.vocab_vectors:
            return []
            
        # Extract forced entities
        heuristic_entities = self._extract_heuristics(query)
        
        # Embed user query
        words = query.split()
        clean_query = " ".join([w for w in words if w.lower() not in STOPWORDS])
        if not clean_query.strip():
            clean_query = query
            
        query_vec = self.embedding_fn(clean_query)
        norm_q = np.linalg.norm(query_vec)
        
        if norm_q == 0:
            return []
            
        scored_terms = []
        max_idf = max(self.idf_dict.values()) if self.idf_dict else 1.0
        if max_idf == 0:
            max_idf = 1.0
            
        for term, term_vec in self.vocab_vectors.items():
            norm_t = np.linalg.norm(term_vec)
            if norm_t == 0:
                continue
            dot_product = np.dot(query_vec, term_vec)
            sim = float(dot_product / (norm_q * norm_t))
            
            # Apply IDF scaling
            if self.idf_dict:
                term_words = [w for w in term.split() if w]
                term_idf = np.mean([self.idf_dict.get(w, 1.0) for w in term_words]) if term_words else 1.0
                lmbda = 0.5
                scaled_factor = lmbda + (1.0 - lmbda) * (term_idf / max_idf)
                sim = sim * scaled_factor
                
            scored_terms.append((term, sim))
            
        # Top-K Selection by IDF-scaled score
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = scored_terms[:top_k]
                
        results = []
        for ent in heuristic_entities:
            results.append({
                "name": ent,
                "weight": 1.0
            })
            
        for term, weight in top_terms:
            if not any(a["name"] == term for a in results):
                results.append({
                    "name": term,
                    "weight": round(weight, 4)
                })
                
        return results[:top_k]


class DenseVocabularyLLMExtractor(DenseVocabularyExtractor):
    """
    Version 3: LLM Keyword Extraction + Generic Term Prevention (IDF Scaling).
    """
    def __init__(self, embedding_fn: Callable[[str], np.ndarray], model_name: str = "qwen3.5-2b", idf_dict: Dict[str, float] = None, embedding_fn_batch: Callable[[List[str]], np.ndarray] = None):
        super().__init__(embedding_fn, embedding_fn_batch=embedding_fn_batch)
        from src.utils.llm_client import LLMClient
        self.llm = LLMClient(model_name=model_name)
        self.idf_dict = idf_dict
        self.schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["keywords"],
            "additionalProperties": False
        }

    def _build_prompt(self, query: str) -> str:
        return f"""You are an expert query analyzer. Your task is to identify and extract up to 5 most critical technical keywords, named entities, acronyms, or specific nouns from the user query that are essential for matching relevant documents.
Focus on highly specific entities (e.g., model names, specific algorithms, specialized metrics).
Do not explain your choices. Return them exactly as they appear in the query.

USER QUERY:
"{query}"

Return a JSON object containing the list of keywords under the key 'keywords'.
"""

    def extract(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip() or not self.vocab_vectors:
            return []
            
        # Extract critical words using LLM
        llm_keywords = []
        prompt = self._build_prompt(query)
        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                json_schema=self.schema,
                temperature=0.0
            )
            parsed = json.loads(raw_response)
            llm_keywords = [kw.strip().lower() for kw in parsed.get("keywords", []) if kw.strip()]
        except Exception as e:
            print(f"DenseVocabularyLLMExtractor: LLM generation failed: {e}")
            
        # Embed user query
        words = query.split()
        clean_query = " ".join([w for w in words if w.lower() not in STOPWORDS])
        if not clean_query.strip():
            clean_query = query
            
        query_vec = self.embedding_fn(clean_query)
        norm_q = np.linalg.norm(query_vec)
        
        if norm_q == 0:
            return []
            
        scored_terms = []
        max_idf = max(self.idf_dict.values()) if self.idf_dict else 1.0
        if max_idf == 0:
            max_idf = 1.0
            
        for term, term_vec in self.vocab_vectors.items():
            norm_t = np.linalg.norm(term_vec)
            if norm_t == 0:
                continue
            dot_product = np.dot(query_vec, term_vec)
            sim = float(dot_product / (norm_q * norm_t))
            
            # Apply IDF scaling
            if self.idf_dict:
                term_words = [w for w in term.split() if w]
                term_idf = np.mean([self.idf_dict.get(w, 1.0) for w in term_words]) if term_words else 1.0
                lmbda = 0.5
                scaled_factor = lmbda + (1.0 - lmbda) * (term_idf / max_idf)
                sim = sim * scaled_factor
                
            scored_terms.append((term, sim))
            
        # Top-K Selection by IDF-scaled score
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = scored_terms[:top_k]
                
        results = []
        for kw in llm_keywords:
            results.append({
                "name": kw,
                "weight": 1.0
            })
            
        for term, weight in top_terms:
            if not any(a["name"] == term for a in results):
                results.append({
                    "name": term,
                    "weight": round(weight, 4)
                })
                
        return results[:top_k]


class DenseVocabularyFastExtractor(DenseVocabularyLLMExtractor):
    """
    Version 4: Fast O(N) Linear Token Frequency Pre-building + Batch BGE Embedding.
    Uses a larger vocab limit (2000) to absorb generic term slots.
    Generic term suppression is handled at query time via IDF Scaling.
    """
    def __init__(self, embedding_fn: Callable[[str], np.ndarray], model_name: str = "qwen3.5-2b", idf_dict: Dict[str, float] = None, embedding_fn_batch: Callable[[List[str]], np.ndarray] = None):
        super().__init__(embedding_fn=embedding_fn, model_name=model_name, idf_dict=idf_dict, embedding_fn_batch=embedding_fn_batch)

    def build_vocabulary(self, corpus_text: str, limit: int = 2000):
        if not corpus_text or not corpus_text.strip():
            return
            
        words = [w.lower() for w in corpus_text.split() if len(w) >= 2 and w.isalpha() and w.lower() not in STOPWORDS]
        if not words:
            return
            
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        vocab_counts = Counter(words + bigrams)
        top_terms = [t for t, c in vocab_counts.most_common(limit)]
        
        if not top_terms:
            return

        all_vectors = self.embedding_fn_batch(top_terms)
        self.vocab_vectors = {t: all_vectors[i] for i, t in enumerate(top_terms)}


class DenseVocabularyFilteredExtractor(DenseVocabularyLLMExtractor):
    """
    Version 5: Fast O(N) Linear Token Frequency + IDF Pre-Filter + Batch BGE Embedding.
    Filters out terms below median IDF BEFORE BGE embedding to save both computation time and vocab slots.
    """
    def __init__(self, embedding_fn: Callable[[str], np.ndarray], model_name: str = "qwen3.5-2b", idf_dict: Dict[str, float] = None, embedding_fn_batch: Callable[[List[str]], np.ndarray] = None):
        super().__init__(embedding_fn=embedding_fn, model_name=model_name, idf_dict=idf_dict, embedding_fn_batch=embedding_fn_batch)

    def build_vocabulary(self, corpus_text: str, limit: int = 1000):
        if not corpus_text or not corpus_text.strip():
            return
            
        words = [w.lower() for w in corpus_text.split() if len(w) >= 2 and w.isalpha() and w.lower() not in STOPWORDS]
        if not words:
            return
            
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        vocab_counts = Counter(words + bigrams)
        candidates = [t for t, c in vocab_counts.most_common(limit * 3)]

        if self.idf_dict and candidates:
            def term_idf(t):
                parts = t.split()
                return sum(self.idf_dict.get(w, 0.0) for w in parts) / len(parts) if parts else 0.0
            
            idf_scores = [term_idf(t) for t in candidates]
            idf_median = sorted(idf_scores)[len(idf_scores) // 2]
            filtered = [t for t, s in zip(candidates, idf_scores) if s >= idf_median]
        else:
            filtered = candidates

        filtered_terms = filtered[:limit]
        if not filtered_terms:
            return

        all_vectors = self.embedding_fn_batch(filtered_terms)
        self.vocab_vectors = {t: all_vectors[i] for i, t in enumerate(filtered_terms)}

