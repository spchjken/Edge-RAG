import os
import sys
import contextlib
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
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
    Batched BGE Embedding Matrix for Corpus Vocabulary (V7 Design).
    
    Features:
    1. Embeds candidate vocabulary in single/batched GPU calls (<0.3s TTI).
    2. Supports Farthest-Point Sampling (FPS) for coverage/hub pool selection.
    3. Caches full stem embedding matrix for 0ms query-time bailout candidate assessment.
    4. Evaluates 1-pass batch GEMM anchor-vocab projection.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = get_cached_bge_model(model_name, self.use_gpu)
        self.vocab_terms: List[str] = []
        self.vocab_embeddings: Optional[torch.Tensor] = None
        self.full_stem_map: Dict[str, torch.Tensor] = {}

    def build_matrix(self, vocab_terms: List[str]) -> torch.Tensor:
        """
        Embeds vocab_terms directly in a single batch call.
        Returns normalized PyTorch Tensor matrix [N_vocab, hidden_dim].
        """
        self.vocab_terms = vocab_terms
        if not vocab_terms:
            self.vocab_embeddings = torch.empty((0, 384))
            return self.vocab_embeddings

        with suppress_progress_bars():
            embeddings_np = self.model.encode(vocab_terms, batch_size=len(vocab_terms))
        tensor_emb = torch.from_numpy(embeddings_np).float()
        self.vocab_embeddings = torch.nn.functional.normalize(tensor_emb, p=2, dim=1)
        for i, term in enumerate(vocab_terms):
            self.full_stem_map[term] = self.vocab_embeddings[i:i+1]
        return self.vocab_embeddings

    def build_with_fps(
        self,
        all_stems: List[str],
        surface_forms: Optional[List[str]] = None,
        target_pool_size: int = 2500,
        batch_size: int = 512
    ) -> torch.Tensor:
        """
        Embeds all candidate stems (via their canonical surface forms) and selects
        target_pool_size terms using Farthest-Point Sampling (FPS) coverage ranking.
        Stores the selected matrix for probing and caches all embeddings for bailout assessment.
        """
        if not all_stems:
            self.vocab_terms = []
            self.vocab_embeddings = torch.empty((0, 384))
            return self.vocab_embeddings

        texts_to_embed = surface_forms if (surface_forms and len(surface_forms) == len(all_stems)) else all_stems

        # Batched embedding encoding
        all_embs_list = []
        with suppress_progress_bars():
            for i in range(0, len(texts_to_embed), batch_size):
                chunk = texts_to_embed[i:i + batch_size]
                emb_chunk = self.model.encode(chunk, batch_size=len(chunk))
                all_embs_list.append(emb_chunk)

        if len(all_embs_list) == 1:
            all_embs_np = all_embs_list[0]
        else:
            all_embs_np = np.vstack(all_embs_list)

        all_embs = torch.from_numpy(all_embs_np).float()
        all_embs = torch.nn.functional.normalize(all_embs, p=2, dim=1)

        # Cache all embeddings into full_stem_map and full_stem_tensor for fast bailout lookup
        self.full_stem_map.clear()
        self.full_stem_tensor = all_embs
        self.stem_to_idx = {stem: idx for idx, stem in enumerate(all_stems)}
        for idx, stem in enumerate(all_stems):
            self.full_stem_map[stem] = all_embs[idx:idx+1]

        N_target = min(target_pool_size, len(all_stems))
        if len(all_stems) <= target_pool_size:
            self.vocab_terms = list(all_stems)
            self.vocab_embeddings = all_embs
            return self.vocab_embeddings

        # Greedy Farthest-Point Sampling (FPS) on PyTorch tensors without V x V pairwise matrix
        device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        embs_dev = all_embs.to(device)

        selected_indices = [0]
        # min_dists is initialized to distance (1 - cosine) from point 0
        min_dists = 1.0 - torch.mv(embs_dev, embs_dev[0])

        for _ in range(1, N_target):
            next_idx = int(torch.argmax(min_dists).item())
            selected_indices.append(next_idx)
            new_dists = 1.0 - torch.mv(embs_dev, embs_dev[next_idx])
            min_dists = torch.minimum(min_dists, new_dists)

        self.vocab_terms = [all_stems[idx] for idx in selected_indices]
        self.vocab_embeddings = all_embs[selected_indices]
        return self.vocab_embeddings

    def get_stem_embeddings_batch(self, stems: List[str]) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieves batched tensor [K, 384] for cached stems via tensor index slicing.
        Returns (valid_stems, tensor_matrix).
        """
        if not stems or not hasattr(self, "full_stem_tensor") or self.full_stem_tensor is None:
            return [], torch.empty((0, 384))

        valid_stems = []
        valid_indices = []
        for s in stems:
            if s in self.stem_to_idx:
                valid_stems.append(s)
                valid_indices.append(self.stem_to_idx[s])

        if not valid_indices:
            return [], torch.empty((0, 384))

        return valid_stems, self.full_stem_tensor[valid_indices]

    def get_stem_embedding(self, stem: str) -> Optional[torch.Tensor]:
        """
        Retrieves embedding tensor [1, 384] for an analyzed stem from the cache.
        If not cached, falls back to encoding on demand.
        """
        if stem in self.full_stem_map:
            return self.full_stem_map[stem]
        encoded = self.encode_terms([stem])
        if encoded.numel() > 0:
            self.full_stem_map[stem] = encoded
            return encoded
        return None

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

