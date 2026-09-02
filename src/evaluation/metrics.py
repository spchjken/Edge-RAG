import time
try:
    import torch
except ImportError:
    torch = None

from typing import List, Callable, Any, Tuple

def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Calculates P@K: Proportion of retrieved chunks that are relevant.
    """
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in ground_truth_ids)
    return hits / len(retrieved_ids)

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Calculates R@K: Proportion of all relevant chunks that were successfully retrieved.
    """
    if not ground_truth_ids:
        return 1.0 # Trivial if no truth required
    hits = sum(1 for cid in retrieved_ids if cid in ground_truth_ids)
    return hits / len(ground_truth_ids)

def calculate_mrr_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 10) -> float:
    """
    Calculates Mean Reciprocal Rank at Rank K (MRR@K).
    """
    gold_set = set(ground_truth_ids)
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank_idx
    return 0.0

def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at Rank K (nDCG@K).
    Supports binary relevance judgments.
    """
    import math
    if not ground_truth_ids or not retrieved_ids or k <= 0:
        return 0.0
    
    gold_set = set(ground_truth_ids)
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
