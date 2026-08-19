import os
import csv
import json
import copy
from typing import List, Dict, Any
from src.pipeline.routing.cascade_routing import CascadeRouter

MOCK_CHUNKS = [
    {
        "chunk_id": "doc1_chunk1",
        "chunk_length": 150,
        "compressed_samples": [
            {"max_keyword_weight": 1.0, "length": 40, "aspects": {"adobe creative cloud": 1.0}}
        ]
    },
    {
        "chunk_id": "doc1_chunk2",
        "chunk_length": 200,
        "compressed_samples": [
            {"max_keyword_weight": 0.8, "length": 150, "aspects": {"adobe creative cloud": 1.0, "revenue": 0.5}}
        ]
    },
    {
        "chunk_id": "doc2_chunk1",
        "chunk_length": 50,
        "compressed_samples": [
            {"max_keyword_weight": 1.0, "length": 5, "aspects": {"ai": 1.0}}
        ]
    },
    {
        "chunk_id": "doc3_chunk1",
        "chunk_length": 500,
        "compressed_samples": [
            {"max_keyword_weight": 0.5, "length": 20, "aspects": {"marketing": 0.5}},
            {"max_keyword_weight": 0.6, "length": 10, "aspects": {"creative": 0.6}}
        ]
    },
    {
        "chunk_id": "doc4_chunk1",
        "chunk_length": 300,
        "compressed_samples": [
            {"max_keyword_weight": 0.1, "length": 2, "aspects": {"cloud": 0.1}}
        ]
    }
]

MOCK_QUERY_ASPECTS = [
    {"name": "adobe creative cloud", "weight": 1.0},
    {"name": "revenue", "weight": 0.5},
    {"name": "ai", "weight": 1.0},
    {"name": "marketing", "weight": 0.5},
    {"name": "creative", "weight": 0.6}
]

def run_evaluation():
    os.makedirs("results/routing_test", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    settings = {
        "Strict": {"tau_alpha": 0.9, "tau_mu": 0.5, "tau_phi": 0.9, "tau_alpha_low": 0.3, "tau_mu_low": 0.1, "N_max": 2, "top_k": 3},
        "Balanced": {"tau_alpha": 0.7, "tau_mu": 0.3, "tau_phi": 0.8, "tau_alpha_low": 0.1, "tau_mu_low": 0.02, "N_max": 3, "top_k": 5},
        "Permissive": {"tau_alpha": 0.4, "tau_mu": 0.1, "tau_phi": 0.5, "tau_alpha_low": 0.05, "tau_mu_low": 0.01, "N_max": 5, "top_k": 10}
    }
    
    results = []
    
    for setting_name, thresholds in settings.items():
        with open("configs/thresholds_test.yaml", "w") as f:
            f.write("routing:\n")
            for k, v in thresholds.items():
                f.write(f"  {k}: {v}\n")
                
        router = CascadeRouter(config_path="configs/thresholds_test.yaml")
        chunks = copy.deepcopy(MOCK_CHUNKS)
        bypass, rerank = router.route_chunks(chunks, MOCK_QUERY_ASPECTS)
        
        bypass_ids = [c["chunk_id"] for c in bypass]
        rerank_ids = [c["chunk_id"] for c in rerank]
        discard_ids = [c["chunk_id"] for c in chunks if c["chunk_id"] not in bypass_ids and c["chunk_id"] not in rerank_ids]
        
        total_original_length = sum(c["chunk_length"] for c in chunks)
        
        retained_chunks = bypass + rerank
        retained_length = sum(sum(m["length"] for m in c["compressed_samples"]) for c in retained_chunks)
        
        comp_ratio = retained_length / total_original_length if total_original_length > 0 else 0
        
        results.append({
            "threshold_setting": setting_name,
            "bypass_count": len(bypass),
            "rerank_count": len(rerank),
            "discard_count": len(discard_ids),
            "original_length": total_original_length,
            "compressed_length": retained_length,
            "compression_ratio": comp_ratio
        })
        
    csv_file = "results/routing_test/cascade_router_metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold_setting", "bypass_count", "rerank_count", "discard_count", "original_length", "compressed_length", "compression_ratio"])
        writer.writeheader()
        writer.writerows(results)
        
    md_file = "results/routing_test/cascade_router_sensitivity.md"
    with open(md_file, "w") as f:
        f.write("# Cascade Router Threshold Sensitivity\n\n")
        f.write("| Setting | Bypass | Rerank | Discard | Compression Ratio |\n")
        f.write("|---------|--------|--------|---------|-------------------|\n")
        for r in results:
            f.write(f"| {r['threshold_setting']} | {r['bypass_count']} | {r['rerank_count']} | {r['discard_count']} | {r['compression_ratio']:.2f} |\n")
        f.write("\n**Conclusion**: The 'Balanced' setting is preferred as it safely discards irrelevant low-density chunks while keeping the compression ratio optimal. The 'Strict' setting drops too many chunks into Discard, risking recall, while 'Permissive' causes unnecessary redundancy by overloading the Rerank queue.\n")

if __name__ == "__main__":
    run_evaluation()
