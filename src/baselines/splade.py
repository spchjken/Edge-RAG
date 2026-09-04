"""
src/baselines/splade.py

Standardized SPLADE-v3 Sparse Neural Retrieval Baseline.
Uses `naver/splade-v3-distilbert` with:
1. Canonical HuggingFace encoder (AutoTokenizer + AutoModelForMaskedLM).
2. Symmetrical padding masking on both document and query encoding.
3. Special token filtering ([CLS], [SEP], [PAD]).
4. Batched CUDA FP16 inference.
5. In-memory Sparse Inverted Index with compact array storage for sub-millisecond retrieval.
"""

from array import array
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SparseInvertedIndex:
    """
    Compact In-Memory Sparse Inverted Index for neural sparse representations.
    Stores postings as integer token IDs -> (doc_ids array('I'), weights array('f')).
    """
    def __init__(self):
        self.num_docs: int = 0
        self.doc_ids_map: List[str] = []
        self.postings: Dict[int, Tuple[array, array]] = {}

    def build(self, sparse_docs: List[Dict[int, float]], doc_ids: List[str]):
        self.num_docs = len(sparse_docs)
        self.doc_ids_map = [str(did) for did in doc_ids]
        self.postings.clear()

        # Temporary raw list accumulator
        temp_postings: Dict[int, Tuple[List[int], List[float]]] = {}

        for doc_idx, doc_dict in enumerate(sparse_docs):
            for token_id, weight in doc_dict.items():
                if weight <= 1e-4:
                    continue
                if token_id not in temp_postings:
                    temp_postings[token_id] = ([], [])
                temp_postings[token_id][0].append(doc_idx)
                temp_postings[token_id][1].append(float(weight))

        # Convert to compact C-arrays
        for token_id, (dids, weights) in temp_postings.items():
            arr_ids = array('I', dids)
            arr_weights = array('f', weights)
            self.postings[token_id] = (arr_ids, arr_weights)

    def retrieve(self, query_activations: Dict[int, float], top_k: int = 10) -> List[Tuple[str, float]]:
        if not query_activations or self.num_docs == 0:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float32)

        # Vectorized accumulator across active query tokens
        for token_id, q_w in query_activations.items():
            if q_w <= 1e-4 or token_id not in self.postings:
                continue
            arr_ids, arr_weights = self.postings[token_id]
            d_ids = np.frombuffer(arr_ids, dtype=np.uint32)
            d_weights = np.frombuffer(arr_weights, dtype=np.float32)
            scores[d_ids] += float(q_w) * d_weights

        active_mask = scores > 0
        num_positive = np.count_nonzero(active_mask)
        if num_positive == 0:
            return []

        k = min(top_k, num_positive)
        if k == self.num_docs:
            top_indices = np.argsort(-scores)
        else:
            partitioned = np.argpartition(-scores, k - 1)[:k]
            top_indices = partitioned[np.argsort(-scores[partitioned])]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s <= 0:
                break
            results.append((self.doc_ids_map[idx], s))
        return results


class TransformersSPLADE:
    """
    Canonical HuggingFace SPLADE Encoder using `naver/splade-v3-distilbert`.
    Applies exact attention-masked logit max-pooling and special token exclusion.
    """
    def __init__(self, model_name: str = "naver/splade-v3-distilbert", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision="main")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, revision="main").to(device)
        self.model.eval()
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    def encode_batch(self, texts: List[str], max_length: int = 512) -> List[Dict[int, float]]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = self.model(**inputs).logits
                # Apply attention mask to zero-out padding positions before max-pooling
                mask = inputs["attention_mask"].unsqueeze(-1)
                log_logits = torch.log1p(torch.relu(logits)) * mask
                sparse_vecs, _ = torch.max(log_logits, dim=1)  # [batch, vocab_size]

        sparse_vecs = sparse_vecs.cpu()
        results = []

        for i in range(len(texts)):
            vec = sparse_vecs[i]
            non_zero_indices = torch.nonzero(vec > 1e-4).squeeze(-1)
            doc_dict: Dict[int, float] = {}
            for idx in non_zero_indices:
                token_id = idx.item()
                if token_id not in self.special_token_ids:
                    doc_dict[token_id] = float(vec[idx].item())
            results.append(doc_dict)

        return results

    def encode_document(self, texts: List[str], batch_size: int = 64) -> List[Dict[int, float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self.encode_batch(batch, max_length=512))
        return results

    def encode_query(self, query: str) -> Dict[int, float]:
        batch_res = self.encode_batch([query], max_length=256)
        return batch_res[0] if batch_res else {}


class SPLADEBaseline:
    """
    Standardized SPLADE Sparse Neural Retrieval Baseline.
    Uses canonical HuggingFace `naver/splade-v3-distilbert` with in-memory inverted index.
    """
    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        model_name: str = "naver/splade-v3-distilbert",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder: Optional[TransformersSPLADE] = None
        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []
        self.corpus_map: Dict[str, str] = {}
        self.index = SparseInvertedIndex()

        if corpus is not None:
            self.build_index(corpus, chunk_ids)

    def _get_encoder(self) -> TransformersSPLADE:
        if self._encoder is None:
            self._encoder = TransformersSPLADE(self.model_name, self.device)
        return self._encoder

    def warmup(self) -> None:
        """Force model loading and initial CUDA kernel execution outside timing loops."""
        encoder = self._get_encoder()
        _ = encoder.encode_query("Warmup query")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def build_index(
        self, corpus: List[str], chunk_ids: Optional[List[str]] = None
    ) -> "SPLADEBaseline":
        """Embed document corpus using batched FP16 and build in-memory inverted index."""
        if chunk_ids is not None and len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids must be the same length as corpus")

        self.corpus = list(corpus)
        self.chunk_ids = chunk_ids if chunk_ids is not None else [
            f"c_{i}" for i in range(len(corpus))
        ]
        self.corpus_map = dict(zip(self.chunk_ids, self.corpus))

        encoder = self._get_encoder()
        sparse_docs = encoder.encode_document(self.corpus, batch_size=64)
        self.index.build(sparse_docs, self.chunk_ids)
        return self

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using sparse inverted index dot-product.
        """
        if self.index.num_docs == 0:
            raise RuntimeError("build_index() must be called before retrieve()")

        encoder = self._get_encoder()
        query_vec = encoder.encode_query(query)
        top_hits = self.index.retrieve(query_vec, top_k=top_k)

        results = []
        for did, score in top_hits:
            results.append({
                "chunk_id": did,
                "text": self.corpus_map.get(did, ""),
                "score": float(score),
            })
        return results
