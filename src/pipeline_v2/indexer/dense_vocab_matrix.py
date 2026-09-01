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
    5. Direct zero-overhead PyTorch FP16 forward + in-pool GPU tensor slicing for query anchors.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = get_cached_bge_model(model_name, self.use_gpu)
        self.hf_model = self.model.model
        self.hf_tokenizer = self.model.tokenizer
        self.hf_model.eval()
        if self.device == "cuda":
            self.hf_model = self.hf_model.half().to(self.device)

        self.vocab_terms: List[str] = []
        self.vocab_embeddings: Optional[torch.Tensor] = None
        self.full_stem_map: Dict[str, torch.Tensor] = {}
        self.full_stem_tensor: Optional[torch.Tensor] = None
        self.stem_to_idx: Dict[str, int] = {}

    def build(self, vocab_stems: List[str], surface_forms: Optional[List[str]] = None, batch_size: int = 512) -> torch.Tensor:
        """
        Embeds vocab_stems (optionally via their canonical surface forms) in GPU batches.
        Returns normalized PyTorch Tensor matrix [N_vocab, hidden_dim].
        """
        self.vocab_terms = vocab_stems
        if not vocab_stems:
            self.vocab_embeddings = torch.empty((0, 384), device=self.device)
            return self.vocab_embeddings

        texts = surface_forms if (surface_forms and len(surface_forms) == len(vocab_stems)) else vocab_stems
        all_embs_list = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                inputs = self.hf_tokenizer(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                outputs = self.hf_model(**inputs)
                cls_emb = outputs[0][:, 0]
                norm_chunk = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                all_embs_list.append(norm_chunk)

        if len(all_embs_list) == 1:
            self.vocab_embeddings = all_embs_list[0]
        else:
            self.vocab_embeddings = torch.cat(all_embs_list, dim=0)

        self.full_stem_map.clear()
        self.full_stem_tensor = self.vocab_embeddings
        self.stem_to_idx = {stem: idx for idx, stem in enumerate(vocab_stems)}
        for i, term in enumerate(vocab_stems):
            self.full_stem_map[term] = self.vocab_embeddings[i:i+1]
        return self.vocab_embeddings

    def build_matrix(self, vocab_terms: List[str]) -> torch.Tensor:
        """
        Embeds vocab_terms directly in a single batch call.
        Returns normalized PyTorch Tensor matrix [N_vocab, hidden_dim].
        """
        return self.build(vocab_terms)

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
            self.vocab_embeddings = torch.empty((0, 384), device=self.device)
            return self.vocab_embeddings

        texts_to_embed = surface_forms if (surface_forms and len(surface_forms) == len(all_stems)) else all_stems

        all_embs_list = []
        with torch.no_grad():
            for i in range(0, len(texts_to_embed), batch_size):
                chunk = texts_to_embed[i:i + batch_size]
                inputs = self.hf_tokenizer(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                outputs = self.hf_model(**inputs)
                cls_emb = outputs[0][:, 0]
                norm_chunk = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                all_embs_list.append(norm_chunk)

        if len(all_embs_list) == 1:
            all_embs = all_embs_list[0]
        else:
            all_embs = torch.cat(all_embs_list, dim=0)

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
        embs_dev = all_embs

        selected_indices = [0]
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
        if not stems or self.full_stem_tensor is None or self.full_stem_tensor.numel() == 0:
            return [], torch.empty((0, 384), device=self.device)

        valid_stems = []
        valid_indices = []
        for s in stems:
            if s in self.stem_to_idx:
                valid_stems.append(s)
                valid_indices.append(self.stem_to_idx[s])

        if not valid_indices:
            return [], torch.empty((0, 384), device=self.device)

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
        prompt = f"Represent this sentence for searching relevant passages: {query}"
        with torch.no_grad():
            inputs = self.hf_tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.hf_model(**inputs)
            cls_emb = outputs[0][:, 0]
            return torch.nn.functional.normalize(cls_emb, p=2, dim=1)

    def encode_terms(self, terms: List[str]) -> torch.Tensor:
        """
        Direct zero-overhead PyTorch FP16 encoding with in-pool GPU tensor slicing:
        1. Slices cached rows directly from full_stem_tensor / vocab_embeddings for in-pool terms.
        2. Direct raw PyTorch forward pass for unseen terms (bypassing FlagModel wrapper).
        3. Returns L2 normalized PyTorch Tensor on self.device [N_terms, 384].
        """
        if not terms:
            return torch.empty((0, 384), device=self.device)

        cached_indices = {}
        unseen_terms = []
        unseen_positions = []

        has_stem_tensor = self.full_stem_tensor is not None and self.full_stem_tensor.numel() > 0
        stem_dict = self.stem_to_idx

        for pos, t in enumerate(terms):
            if has_stem_tensor and t in stem_dict:
                cached_indices[pos] = stem_dict[t]
            else:
                unseen_positions.append(pos)
                unseen_terms.append(t)

        out_dtype = torch.float16 if self.device == "cuda" else torch.float32
        out_tensor = torch.empty((len(terms), 384), device=self.device, dtype=out_dtype)

        # Fill in-pool sliced rows
        if cached_indices:
            src_indices = [cached_indices[pos] for pos in sorted(cached_indices.keys())]
            target_positions = sorted(cached_indices.keys())
            sliced_embs = self.full_stem_tensor[src_indices].to(self.device, dtype=out_dtype)
            out_tensor[target_positions] = sliced_embs

        # Forward pass unseen terms (if any)
        if unseen_terms:
            with torch.no_grad():
                inputs = self.hf_tokenizer(
                    unseen_terms,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.hf_model(**inputs)
                cls_emb = outputs[0][:, 0]
                unseen_norm = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                out_tensor[unseen_positions] = unseen_norm

        return out_tensor

