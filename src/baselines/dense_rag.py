import numpy as np

try:
    import torch
    from FlagEmbedding import FlagModel
except ImportError:
    torch = None
    FlagModel = None

from typing import List, Dict, Any, Optional

# Standard BGE retrieval query instruction for bge-small-en-v1.5 / bge-base-en-v1.5.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class DenseRAGBaseline:
    """
    Dense RAG baseline (BGE via FlagEmbedding).

    Two-phase usage so corpus embedding cost (TTI) is separated from
    per-query retrieval latency:

        dense = DenseRAGBaseline()
        dense.warmup()               # initialize CUDA/model before any timing
        dense.build_index(corpus_texts, chunk_ids)   # one-time embed (timed by caller)
        docs = dense.retrieve(query, top_k=10)       # per-query embed + search

    Corpus embeddings are cached after ``build_index`` and reused for every
    query. Per-query latency includes the query embedding time. Similarity is
    cosine (L2-normalized inner product).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        use_fp16: bool = True,
    ):
        if FlagModel is None:
            raise ImportError("FlagEmbedding library is required for DenseRAGBaseline.")
        # fp16 halves the VRAM footprint (e.g. bge-small ~0.13GB -> ~0.07GB).
        self.model = FlagModel(
            model_name,
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16,
        )
        self._corpus_texts: Optional[List[str]] = None
        self._chunk_ids: Optional[List[str]] = None
        self._corpus_embeddings: Optional[np.ndarray] = None

    def warmup(self) -> None:
        """Force CUDA/model initialization before any timing is measured.

        The first encode call performs lazy CUDA context / kernel init and is
        unrepresentatively slow; run this once before timing build_index/retrieve.
        """
        _ = np.asarray(self.model.encode(["Warmup CUDA"]))
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def build_index(self, corpus_texts: List[str],
                    chunk_ids: Optional[List[str]] = None) -> "DenseRAGBaseline":
        """Embed the corpus once (the TTI component) and cache it."""
        if chunk_ids is not None and len(chunk_ids) != len(corpus_texts):
            raise ValueError("chunk_ids must be the same length as corpus_texts")
        self._corpus_texts = list(corpus_texts)
        self._chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus_texts))
        ]
        emb = np.asarray(self.model.encode_corpus(self._corpus_texts))
        self._corpus_embeddings = self._l2_normalize(emb)
        return self

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks by cosine similarity.

        Includes the query embedding time in the call. Requires ``build_index``
        to have been called first. Returns real ``chunk_id``, ``text``, ``score``.
        """
        if self._corpus_embeddings is None:
            raise RuntimeError("build_index() must be called before retrieve()")

        q_emb = np.asarray(self.model.encode_queries([query]))
        q_emb = self._l2_normalize(q_emb)
        scores = (self._corpus_embeddings @ q_emb.T).ravel()

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k].tolist()

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": self._corpus_texts[idx],
                "score": float(scores[idx]),
            })
        return results

    def log_vram_usage(self) -> None:
        if torch is not None and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[Dense Baseline VRAM] Peak memory allocated: {peak_vram:.2f} GB")

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms
