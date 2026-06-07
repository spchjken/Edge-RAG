try:
    import torch
    from FlagEmbedding import FlagModel
except ImportError:
    torch = None
    FlagModel = None

from typing import List, Dict, Any

class DenseRAGBaseline:
    """
    Dense RAG baseline using BGE-M3.
    Optimized for VRAM using fp16 precision.
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        if not FlagModel:
            raise ImportError("FlagEmbedding library is required for DenseRAGBaseline.")
            
        # Optimization: use_fp16=True halves the VRAM footprint (from ~2.2GB to ~1.1GB)
        self.model = FlagModel(model_name, 
                               query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query",
                               use_fp16=True)

    def log_vram_usage(self):
        if torch and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[Dense Baseline VRAM] Peak memory allocated: {peak_vram:.2f} GB")

    def retrieve(self, query: str, corpus: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks by cosine similarity of dense embeddings.
        """
        if not corpus:
            return []

        # Encode query
        q_embedding = self.model.encode_queries([query])
        
        # Encode corpus
        # Note: In a real benchmark, we'd cache corpus embeddings if doing multiple queries.
        c_embeddings = self.model.encode(corpus)
        
        # Cosine similarity
        # FlagEmbedding outputs normalized vectors usually, but doing inner product is standard.
        scores = c_embeddings @ q_embedding.T
        scores = scores.squeeze(-1)
        
        # Get top-k indices
        # If corpus is smaller than top_k, limit it
        k = min(top_k, len(corpus))
        
        if torch and isinstance(scores, torch.Tensor):
            # If FlagEmbedding returned torch tensors
            top_k_scores, top_k_indices = torch.topk(torch.tensor(scores), k)
            top_k_indices = top_k_indices.tolist()
            top_k_scores = top_k_scores.tolist()
        else:
            # If numpy arrays
            import numpy as np
            top_k_indices = np.argsort(scores)[::-1][:k].tolist()
            top_k_scores = [float(scores[i]) for i in top_k_indices]
            
        results = []
        for i, idx in enumerate(top_k_indices):
            results.append({
                "chunk_id": f"c_{idx}",
                "text": corpus[idx],
                "score": top_k_scores[i]
            })
            
        self.log_vram_usage()
        return results
