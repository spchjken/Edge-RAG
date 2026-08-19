from typing import List, Dict, Any, Optional
import torch


class SPLADEBaseline:
    """
    SPLADE Sparse Neural Retrieval Baseline.
    Uses `naver/splade-v3-distilbert` from Hugging Face for fast sparse neural retrieval
    and contextual term expansion.

    Two-phase usage so model loading/warmup is separated from index build (TTI):
        splade = SPLADEBaseline(model_name="naver/splade-v3-distilbert")
        splade.warmup()                             # load weights & warmup CUDA
        splade.build_index(corpus_texts, chunk_ids) # timed by caller for TTI
        docs = splade.retrieve(query, top_k=10)
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        model_name: str = "naver/splade-v3-distilbert",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._encoder = None
        self.corpus_embeddings = None
        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []

        if corpus is not None:
            self.build_index(corpus, chunk_ids)

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SparseEncoder
                self._encoder = SparseEncoder(self.model_name, device=self.device)
            except Exception:
                from transformers import AutoTokenizer, AutoModelForMaskedLM

                class TransformersSPLADE:
                    def __init__(self, model_name: str, device: str):
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
                        self.model.eval()
                        self.device = device

                    def encode(self, texts: List[str]) -> List[Dict[str, float]]:
                        results = []
                        for text in texts:
                            inputs = self.tokenizer(
                                text, return_tensors="pt", truncation=True, max_length=512
                            ).to(self.device)
                            with torch.no_grad():
                                logits = self.model(**inputs).logits
                                relu_logits = torch.relu(logits)
                                log_logits = torch.log(1 + relu_logits)
                                values, _ = torch.max(log_logits, dim=1)
                                values = values.squeeze(0)

                                non_zero_indices = torch.nonzero(values).squeeze(-1)
                                doc_dict = {}
                                for idx in non_zero_indices:
                                    token_str = self.tokenizer.decode([idx.item()]).strip()
                                    weight = values[idx].item()
                                    if token_str and weight > 0:
                                        doc_dict[token_str] = weight
                                results.append(doc_dict)
                        return results

                    def encode_document(self, texts: List[str]):
                        return self.encode(texts)

                    def encode_query(self, query: str):
                        return self.encode([query])[0]

                self._encoder = TransformersSPLADE(self.model_name, self.device)
        return self._encoder

    def warmup(self) -> None:
        """Force model loading and initial CUDA kernel execution outside timing loops."""
        encoder = self._get_encoder()
        if hasattr(encoder, "encode_query"):
            _ = encoder.encode_query("Warmup query")
        elif hasattr(encoder, "encode"):
            _ = encoder.encode(["Warmup query"])
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def build_index(
        self, corpus: List[str], chunk_ids: Optional[List[str]] = None
    ) -> "SPLADEBaseline":
        """Embed document corpus to build the sparse inverted representation."""
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")
        self.corpus = list(corpus)
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        encoder = self._get_encoder()
        if hasattr(encoder, "encode_document"):
            self.corpus_embeddings = encoder.encode_document(self.corpus)
        elif hasattr(encoder, "encode"):
            self.corpus_embeddings = encoder.encode(self.corpus)
        return self

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using SPLADE sparse dot-product.
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("build_index() must be called before retrieve()")

        encoder = self._get_encoder()
        if hasattr(encoder, "encode_query"):
            query_vec = encoder.encode_query(query)
        else:
            query_vec = encoder.encode([query])[0]

        # Use encoder.similarity if supported by sentence_transformers SparseEncoder
        if hasattr(encoder, "similarity"):
            try:
                sim_matrix = encoder.similarity(query_vec, self.corpus_embeddings)
                if isinstance(sim_matrix, torch.Tensor):
                    scores = sim_matrix.flatten().cpu().tolist()
                else:
                    scores = list(sim_matrix[0])
            except Exception:
                scores = self._compute_scores_manual(query_vec)
        else:
            scores = self._compute_scores_manual(query_vec)

        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_k_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx]),
            })
        return results

    def _compute_scores_manual(self, query_vec) -> List[float]:
        scores = []
        if isinstance(query_vec, dict):
            for doc_vec in self.corpus_embeddings:
                score = 0.0
                if isinstance(doc_vec, dict):
                    for term, q_w in query_vec.items():
                        if term in doc_vec:
                            score += q_w * doc_vec[term]
                scores.append(score)
            return scores

        if isinstance(query_vec, torch.Tensor):
            q_tensor = query_vec.cpu()
            if q_tensor.is_sparse:
                q_tensor = q_tensor.coalesce()
                q_indices = q_tensor.indices().flatten().tolist()
                q_values = q_tensor.values().flatten().tolist()
                q_dict = dict(zip(q_indices, q_values))

                for doc_vec in self.corpus_embeddings:
                    d_tensor = doc_vec.cpu() if isinstance(doc_vec, torch.Tensor) else doc_vec
                    score = 0.0
                    if isinstance(d_tensor, torch.Tensor) and d_tensor.is_sparse:
                        d_tensor = d_tensor.coalesce()
                        d_indices = d_tensor.indices().flatten().tolist()
                        d_values = d_tensor.values().flatten().tolist()
                        for idx_val, val in zip(d_indices, d_values):
                            if idx_val in q_dict:
                                score += q_dict[idx_val] * val
                    scores.append(score)
                return scores

        for doc_vec in self.corpus_embeddings:
            q_t = query_vec.cpu().to_dense().flatten() if isinstance(query_vec, torch.Tensor) else torch.tensor(query_vec)
            d_t = doc_vec.cpu().to_dense().flatten() if isinstance(doc_vec, torch.Tensor) else torch.tensor(doc_vec)
            score = torch.dot(q_t, d_t).item()
            scores.append(score)
        return scores
