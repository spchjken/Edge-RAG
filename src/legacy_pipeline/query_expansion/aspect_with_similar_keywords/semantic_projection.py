import numpy as np
from typing import Dict, Any, List, Tuple
import yake
import faiss
from FlagEmbedding import FlagModel

class VectorQueryExpander:
    def __init__(self, 
                 bge_model_name: str = "BAAI/bge-small-en-v1.5",
                 faiss_index_path: str = None,
                 faiss_vocab_list: List[str] = None):
        """
        Initializes the Pathway B (Unified Vector) query expander.
        
        Args:
            bge_model_name: Name or path to the BGE dense embedding model.
            faiss_index_path: Path to the pre-computed FAISS index (.index file).
            faiss_vocab_list: List of strings corresponding to the vectors in the FAISS index,
                              used to map FAISS integer indices back to terms.
        """
        # Load YAKE for Aspect Extraction (Step 1)
        self.yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=20)
        
        # Load BGE Model (Step 2, 3, 4)
        print(f"Loading BGE model: {bge_model_name}")
        self.bge_model = FlagModel(bge_model_name, 
                                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                                   use_fp16=True) # Use fp16 for edge devices
        
        # Load FAISS Index (Step 3)
        self.faiss_index = None
        self.faiss_vocab = faiss_vocab_list or []
        if faiss_index_path:
            try:
                self.faiss_index = faiss.read_index(faiss_index_path)
            except Exception as e:
                print(f"Warning: Could not load FAISS index from {faiss_index_path}: {e}")
                
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
        
    def expand(self, query: str, K: int = 5) -> Dict[str, Any]:
        """
        Executes the 6-step Pathway B expansion logic.
        """
        # 1. Aspect Extraction
        extracted = self.yake_extractor.extract_keywords(query)
        raw_aspects = [kw[0].lower() for kw in extracted]
        
        # 2. Orthogonalization (Lexical Subsumption + Semantic Clustering)
        
        # 2a. Global Lexical Subsumption
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
                
        # We cache the vectors to reuse them in Step 3
        aspect_vectors = {asp: self.bge_model.encode(asp) for asp in lexically_filtered}
        
        # 2b. Semantic Clustering
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
            
        # 3. Keyword Exploration (BEFORE weight assignment)
        aspect_data = []
        for aspect in orthogonal_aspects:
            aspect_vec = aspect_vectors.get(aspect, self.bge_model.encode(aspect))
            
            # The aspect's own term is always the first keyword
            expanded_keywords = [{"term": aspect, "weight": 1.0}]
            
            # Query FAISS index
            if self.faiss_index and self.faiss_vocab:
                # FAISS expects 2D array: (1, d)
                search_vec = np.array([aspect_vec]).astype('float32')
                # We use inner product (IP) index for cosine similarity if vectors are normalized
                # Assuming index returns distances and indices
                distances, indices = self.faiss_index.search(search_vec, 3)
                
                for idx_rank in range(len(indices[0])):
                    vocab_idx = indices[0][idx_rank]
                    if vocab_idx != -1 and vocab_idx < len(self.faiss_vocab):
                        word = self.faiss_vocab[vocab_idx]
                        score = float(distances[0][idx_rank])
                        if word != aspect:
                            expanded_keywords.append({"term": word, "weight": round(score, 2)})
                            
            aspect_data.append({
                "term": aspect,
                "keywords": expanded_keywords
            })
            
        # 4. Weight Assignment (AFTER expansion — uses enriched vector)
        query_vector = self.bge_model.encode(query)
        for item in aspect_data:
            # Build enriched text: aspect name + all expanded keyword terms
            all_terms = " ".join([kw["term"] for kw in item["keywords"]])
            enriched_vector = self.bge_model.encode(all_terms)
            
            weight = self._cosine_similarity(query_vector, enriched_vector)
            item["aspect_weight"] = round(weight, 2)
            
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
