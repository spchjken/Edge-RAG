import yaml
from typing import List, Dict, Any, Tuple

class AdaptiveRouter:
    """
    Implements the Aspect-Weighted Density Routing engine.
    See section 3.3 of ARCHITECTURE.md for formulas.
    """
    def __init__(self, config_path: str = "configs/thresholds.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        routing_cfg = config.get("routing", {})
        self.tau_bypass = routing_cfg.get("tau_bypass", 0.85)
        self.tau_discard = routing_cfg.get("tau_discard", 0.15)
        self.top_k = routing_cfg.get("top_k", 20)

    def route_chunks(self, chunks_data: List[Dict[str, Any]], query_aspects: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Calculates the density of compressed intervals within chunks and 
        routes them to either Bypass_List or Rerank_Queue.
        
        chunks_data format expected:
        [
            {
                "chunk_id": "...",
                "chunk_length": int,
                "compressed_samples": [
                    {
                        "length": int,
                        "max_keyword_weight": float,
                        "aspects": {"Aspect1": 1.0, ...}
                    }, ...
                ]
            }, ...
        ]
        """
        bypass_list = []
        candidate_pool = []
        
        # Total sum of all query aspect weights
        total_query_aspect_weight = sum(a.get("aspect_weight", 1.0) for a in query_aspects)
        if total_query_aspect_weight == 0:
            total_query_aspect_weight = 1.0 # Prevent division by zero
            
        for chunk in chunks_data:
            samples = chunk.get("compressed_samples", [])
            chunk_length = chunk.get("chunk_length", 1)
            if chunk_length == 0:
                continue
                
            if not samples:
                continue
                
            # 1. Contiguous & Scattered Density
            max_weighted_length = 0.0
            sum_weighted_length = 0.0
            chunk_aspects_found = {}
            
            for m in samples:
                m_len = m["length"]
                w_k = m["max_keyword_weight"]
                weighted_len = m_len * w_k
                
                max_weighted_length = max(max_weighted_length, weighted_len)
                sum_weighted_length += weighted_len
                
                # Collect aspects found in this chunk
                for asp_name, asp_weight in m.get("aspects", {}).items():
                    chunk_aspects_found[asp_name] = asp_weight
                    
            rho_cont_weighted = max_weighted_length / chunk_length
            rho_scat_weighted = sum_weighted_length / chunk_length
            
            # 2. Aspect Coverage (alpha)
            chunk_aspect_weight_sum = sum(chunk_aspects_found.values())
            alpha = chunk_aspect_weight_sum / total_query_aspect_weight
            
            # 3. Final Score
            score = alpha * (rho_cont_weighted + rho_scat_weighted)
            chunk["score"] = score
            
            # 4. Routing Decision
            if score > self.tau_bypass:
                bypass_list.append(chunk)
            elif score >= self.tau_discard:
                candidate_pool.append(chunk)
            # else: discarded
                
        # Sort candidates descending and take top-K for the Rerank Queue
        candidate_pool.sort(key=lambda x: x["score"], reverse=True)
        rerank_queue = candidate_pool[:self.top_k]
        
        return bypass_list, rerank_queue
