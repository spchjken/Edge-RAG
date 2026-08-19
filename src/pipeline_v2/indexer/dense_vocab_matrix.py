import os
import sys
import contextlib
from typing import List, Optional, Dict
import torch
from FlagEmbedding import FlagModel

# Global model cache to avoid re-loading PyTorch weights on every TTI call
_BGE_MODEL_CACHE: Dict[str, FlagModel] = {}


@contextlib.contextmanager
def suppress_progress_bars():
    """Silences tqdm progress bars emitted by FlagEmbedding/transformers."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_cached_bge_model(model_name: str, use_gpu: bool) -> FlagModel:
    key = f"{model_name}_{use_gpu}"
    if key not in _BGE_MODEL_CACHE:
        with suppress_progress_bars():
            _BGE_MODEL_CACHE[key] = FlagModel(
                model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=use_gpu
            )
    return _BGE_MODEL_CACHE[key]


class DenseVocabMatrix:
    """
    Batched BGE Embedding Matrix for Corpus Vocabulary.
    Embeds the clean 1,000 candidate vocabulary pool in 1 GPU batch call (<0.3s TTI).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = get_cached_bge_model(model_name, self.use_gpu)
        self.vocab_terms: List[str] = []
        self.vocab_embeddings: Optional[torch.Tensor] = None

    def build_matrix(self, vocab_terms: List[str]) -> torch.Tensor:
        """
        Embeds vocab_terms in a single batch call.
        Returns normalized PyTorch Tensor matrix [N_vocab, hidden_dim].
        """
        self.vocab_terms = vocab_terms
        if not vocab_terms:
            self.vocab_embeddings = torch.empty((0, 384))
            return self.vocab_embeddings

        with suppress_progress_bars():
            embeddings_np = self.model.encode(vocab_terms, batch_size=len(vocab_terms))
        tensor_emb = torch.from_numpy(embeddings_np).float()
        # L2 normalize for cosine similarity calculation
        self.vocab_embeddings = torch.nn.functional.normalize(tensor_emb, p=2, dim=1)
        return self.vocab_embeddings

    def encode_query(self, query: str) -> torch.Tensor:
        """Encodes query string into L2 normalized PyTorch Tensor [1, hidden_dim]."""
        with suppress_progress_bars():
            emb_np = self.model.encode_queries([query])
        tensor_emb = torch.from_numpy(emb_np).float()
        return torch.nn.functional.normalize(tensor_emb, p=2, dim=1)

    def encode_terms(self, terms: List[str]) -> torch.Tensor:
        """Encodes list of short terms/anchors into L2 normalized PyTorch Tensor [N_terms, hidden_dim]."""
        if not terms:
            return torch.empty((0, 384))
        with suppress_progress_bars():
            emb_np = self.model.encode(terms, batch_size=len(terms))
        tensor_emb = torch.from_numpy(emb_np).float()
        return torch.nn.functional.normalize(tensor_emb, p=2, dim=1)
