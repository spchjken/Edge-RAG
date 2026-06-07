from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Baseline:
    """
    BM25 Sparse Retrieval Baseline.
    Fully isolated, CPU-bound, implementing the Okapi BM25 algorithm.
    """
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        # Simple whitespace tokenization for BM25
        self.tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks using BM25.
        Returns a list of dictionaries with text and score.
        """
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": f"c_{idx}", # Mock ID or pass mapping if needed
                "text": self.corpus[idx],
                "score": scores[idx]
            })
            
        return results
