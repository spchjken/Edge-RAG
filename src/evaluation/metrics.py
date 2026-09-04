import time
try:
    import torch
except ImportError:
    torch = None

from typing import List, Dict, Callable, Any, Tuple, Union

def calculate_precision_at_k(retrieved_ids: List[str], ground_truth: Union[List[str], Dict[str, float]]) -> float:
    """
    Calculates P@K: Proportion of retrieved chunks that are relevant.
    Supports both binary list of doc IDs and graded dict of {doc_id: score}.
    """
    if not retrieved_ids or not ground_truth:
        return 0.0
    gold_set = set(ground_truth) if isinstance(ground_truth, (list, set, tuple)) else {did for did, s in ground_truth.items() if s > 0}
    hits = sum(1 for cid in retrieved_ids if cid in gold_set)
    return hits / len(retrieved_ids)

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth: Union[List[str], Dict[str, float]]) -> float:
    """
    Calculates R@K: Proportion of all relevant chunks that were successfully retrieved.
    Supports both binary list of doc IDs and graded dict of {doc_id: score}.
    """
    if not ground_truth:
        return 1.0 # Trivial if no truth required
    gold_set = set(ground_truth) if isinstance(ground_truth, (list, set, tuple)) else {did for did, s in ground_truth.items() if s > 0}
    if not gold_set:
        return 1.0
    hits = sum(1 for cid in retrieved_ids if cid in gold_set)
    return hits / len(gold_set)

def calculate_mrr_at_k(retrieved_ids: List[str], ground_truth: Union[List[str], Dict[str, float]], k: int = 10) -> float:
    """
    Calculates Mean Reciprocal Rank at Rank K (MRR@K).
    Supports both binary list of doc IDs and graded dict of {doc_id: score}.
    """
    if not ground_truth or not retrieved_ids:
        return 0.0
    gold_set = set(ground_truth) if isinstance(ground_truth, (list, set, tuple)) else {did for did, s in ground_truth.items() if s > 0}
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank_idx
    return 0.0

def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    ground_truth: Union[List[str], Dict[str, float]],
    k: int = 10
) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at Rank K (nDCG@K).
    Supports both graded relevance judgments (Dict[doc_id, float]) and binary relevance (List[doc_id]).
    
    Formula for Graded Relevance (BEIR / TREC standard):
        DCG@K = sum_{i=1}^K (2^(rel_i) - 1) / log2(i + 1)
        IDCG@K = sum_{j=1}^{min(K, |G|)} (2^(rel_(j)) - 1) / log2(j + 1)
        where rel_(j) are all positive relevance scores sorted strictly descending.
        
    Formula for Binary Relevance:
        DCG@K = sum_{i=1}^K 1.0 / log2(i + 1) for hits
        IDCG@K = sum_{j=1}^{min(K, |G|)} 1.0 / log2(j + 1)
    """
    import math
    if not ground_truth or not retrieved_ids or k <= 0:
        return 0.0

    if isinstance(ground_truth, dict):
        # Graded relevance calculation (Official BEIR/TREC standard)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = float(ground_truth.get(doc_id, 0.0))
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)

        positive_gains = sorted([float(s) for s in ground_truth.values() if s > 0], reverse=True)
        if not positive_gains:
            return 0.0

        idcg = sum((2.0 ** rel - 1.0) / math.log2(j + 1) for j, rel in enumerate(positive_gains[:k], start=1))
        return dcg / idcg if idcg > 0.0 else 0.0
    else:
        # Binary relevance calculation
        gold_set = set(ground_truth)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in gold_set:
                dcg += 1.0 / math.log2(i + 1)

        idcg = sum(1.0 / math.log2(j + 1) for j in range(1, min(k, len(gold_set)) + 1))
        return dcg / idcg if idcg > 0.0 else 0.0

def calculate_compression_ratio(uncompressed_tokens: int, compressed_tokens: int) -> float:
    """
    Calculates the Context Compression Ratio (C_r).
    """
    if compressed_tokens == 0:
        return float('inf')
    return uncompressed_tokens / compressed_tokens

def measure_ttft(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Executes a function and isolates its exact wall-clock execution time.
    Synchronizes CUDA to prevent async masking of true latency.
    
    Returns: (Function Result, TTFT in seconds)
    """
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    return result, (end_time - start_time)
