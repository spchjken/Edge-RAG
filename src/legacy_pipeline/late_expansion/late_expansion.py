import yaml
try:
    import torch
except ImportError:
    torch = None

from typing import List, Dict, Any

class LateExpander:
    """
    Implements the Late-Expansion mechanism to fetch uncompressed original text 
    while enforcing the VRAM Overflow Protection (N_max) limit.
    See section 3.5 of ARCHITECTURE.md.
    """
    def __init__(self, config_path: str = "configs/thresholds.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.n_max = config.get("vram", {}).get("N_max", 10)

    def log_vram_usage(self):
        """
        Optional VRAM monitor. As per module rules, this is the only place 
        torch is permitted in the pipeline.
        """
        if torch and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[VRAM Guard] Peak memory allocated: {peak_vram:.2f} GB")

    def expand(self, bypass_list: List[Dict[str, Any]], reranked_list: List[Dict[str, Any]], original_corpus: Dict[str, str]) -> List[str]:
        """
        Combines Bypass and Reranked lists, truncates to N_max, 
        and fetches the original uncompressed text for generation.
        
        original_corpus is a mapping from chunk_id to its full uncompressed string.
        """
        final_context_chunks = []
        
        # Priority 1: Bypass List
        for chunk in bypass_list:
            if len(final_context_chunks) >= self.n_max:
                break
            chunk_id = chunk.get("chunk_id")
            if chunk_id in original_corpus:
                final_context_chunks.append(original_corpus[chunk_id])
                
        # Priority 2: Reranked List (already sorted by relevance_score)
        for chunk in reranked_list:
            if len(final_context_chunks) >= self.n_max:
                break
            chunk_id = chunk.get("chunk_id")
            if chunk_id in original_corpus:
                final_context_chunks.append(original_corpus[chunk_id])
                
        self.log_vram_usage()
        
        return final_context_chunks
