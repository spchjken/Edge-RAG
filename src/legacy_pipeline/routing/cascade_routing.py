import yaml
from typing import List, Dict, Any, Tuple

class CascadeRouter:
    """
    Implements the Cost-Aware Cascade Routing engine.
    Separates the density score into orthogonal Mass (μ) and Focus (φ) metrics
    for a three-way triage (Bypass, Rerank, Discard).
    """
    def __init__(
        self,
        config_path: str = "configs/thresholds.yaml",
        tau_bypass: float = None,
        tau_discard: float = None
    ):
        # Provide fallback defaults in case config is not updated yet
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}
            
        routing_cfg = config.get("routing", {})
        
        # Bypass thresholds
        self.tau_alpha = tau_bypass if tau_bypass is not None else routing_cfg.get("tau_bypass", routing_cfg.get("tau_alpha", 0.85))
        self.tau_mu = routing_cfg.get("tau_mu", 0.3)
        self.tau_phi = routing_cfg.get("tau_phi", 0.8)
        
        # Discard thresholds
        self.tau_alpha_low = tau_discard if tau_discard is not None else routing_cfg.get("tau_discard", routing_cfg.get("tau_alpha_low", 0.15))
        self.tau_mu_low = routing_cfg.get("tau_mu_low", 0.02)
        
        # Budget limits
        self.n_max = routing_cfg.get("N_max", 3)
        self.top_k = routing_cfg.get("top_k", 20)

    def route_chunks(self, chunks_data: List[Dict[str, Any]], query_aspects: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Calculates Mass, Focus, and Aspect Coverage to triage chunks.
        Returns: (Bypass_List, Rerank_Queue)
        """
        # Supports both new "weight" and old "aspect_weight" keys
        total_query_weight = sum(a.get("weight", a.get("aspect_weight", 1.0)) for a in query_aspects)
        if total_query_weight == 0:
            total_query_weight = 1.0 # Prevent division by zero
            
        scored_chunks = []
        
        for chunk in chunks_data:
            samples = chunk.get("compressed_samples", [])
            chunk_length = chunk.get("chunk_length", 1)
            
            if chunk_length == 0 or not samples:
                continue # Implicitly discarded
                
            sum_weighted_len = 0.0
            max_weighted_len = 0.0
            aspects_found = {}
            
            for m in samples:
                w_bar = m["max_keyword_weight"]
                wl = m["length"] * w_bar
                
                sum_weighted_len += wl
                max_weighted_len = max(max_weighted_len, wl)
                
                # Dedup aspects found (store highest weight)
                for asp_name, asp_weight in m.get("aspects", {}).items():
                    if asp_name in aspects_found:
                        aspects_found[asp_name] = max(aspects_found[asp_name], asp_weight)
                    else:
                        aspects_found[asp_name] = asp_weight
            
            # Compute Mass (μ)
            mu = sum_weighted_len / chunk_length
            
            # Compute Focus (φ)
            if sum_weighted_len > 0:
                phi = max_weighted_len / sum_weighted_len
            else:
                phi = 0.0
                
            # Compute Aspect Coverage (α)
            alpha = sum(aspects_found.values()) / total_query_weight
            
            # Compute Soft-OR Score
            score = alpha * (mu + phi - (mu * phi))
            chunk["score"] = score
            chunk["metrics"] = {"alpha": alpha, "mu": mu, "phi": phi}
            
            # Three-Way Triage Decision
            if alpha < self.tau_alpha_low or mu < self.tau_mu_low:
                decision = "DISCARD"
            elif alpha >= self.tau_alpha and (mu >= self.tau_mu or phi >= self.tau_phi):
                decision = "BYPASS"
            else:
                decision = "RERANK"
                
            scored_chunks.append({"chunk": chunk, "score": score, "decision": decision})
            
        # Budget-Coupled Admission
        bypass_candidates = [c["chunk"] for c in scored_chunks if c["decision"] == "BYPASS"]
        rerank_candidates = [c["chunk"] for c in scored_chunks if c["decision"] == "RERANK"]
        
        # Sort bypass candidates by score descending
        bypass_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        bypass_list = []
        if len(bypass_candidates) > self.n_max:
            # Demote excess to rerank
            bypass_list = bypass_candidates[:self.n_max]
            rerank_candidates.extend(bypass_candidates[self.n_max:])
        else:
            bypass_list = bypass_candidates
            
        # Sort rerank candidates by score descending and take top-K
        rerank_candidates.sort(key=lambda x: x["score"], reverse=True)
        rerank_queue = rerank_candidates[:self.top_k]
        
        return bypass_list, rerank_queue
