import os
import sys
import json
import shutil
import glob
from typing import List, Dict, Any

# Ensure src & scripts/benchmark_creation are in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.benchmark_creation.step1_chunking import process_document, get_tokenizer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ai_stress_dir = os.path.join(base_dir, "data", "benchmarks", "ai", "corpus_stress_1")
    fintech_stress_dir = os.path.join(base_dir, "data", "benchmarks", "fintech", "corpus_stress_1")
    
    out_dir = os.path.join(base_dir, "data", "benchmarks", "fused", "corpus_stress_50")
    out_chunks_dir = os.path.join(out_dir, "step1_chunks")
    os.makedirs(out_chunks_dir, exist_ok=True)

    print("Building 50-paper fused cross-domain stress benchmark...")
    
    # 1. Load ground truth benchmark JSONs
    ai_benchmark_path = os.path.join(ai_stress_dir, "final_benchmark_corpus_stress_1.json")
    fintech_benchmark_path = os.path.join(fintech_stress_dir, "final_benchmark_corpus_stress_1.json")
    
    with open(ai_benchmark_path, "r", encoding="utf-8") as f:
        ai_queries = json.load(f)
    with open(fintech_benchmark_path, "r", encoding="utf-8") as f:
        fintech_queries = json.load(f)
        
    ai_chunks_files = sorted(glob.glob(os.path.join(ai_stress_dir, "step1_chunks", "*_chunks.json")))
    fintech_chunks_files = sorted(glob.glob(os.path.join(fintech_stress_dir, "step1_chunks", "*_chunks.json")))
    
    # Select top 10 target papers from AI and top 10 target papers from Fintech
    target_ai_files = ai_chunks_files[:10]
    target_fintech_files = fintech_chunks_files[:10]
    
    target_ai_doc_ids = set()
    target_fintech_doc_ids = set()
    
    # Copy chunk JSONs for target papers
    for filepath in target_ai_files:
        filename = os.path.basename(filepath)
        doc_id = filename.replace("_chunks.json", "")
        target_ai_doc_ids.add(doc_id)
        shutil.copy(filepath, os.path.join(out_chunks_dir, filename))
        
    for filepath in target_fintech_files:
        filename = os.path.basename(filepath)
        doc_id = filename.replace("_chunks.json", "")
        target_fintech_doc_ids.add(doc_id)
        shutil.copy(filepath, os.path.join(out_chunks_dir, filename))

    print(f"Copied {len(target_ai_files)} AI target papers and {len(target_fintech_files)} Fintech target papers.")
    
    # Filter queries for selected target papers
    selected_queries = []
    for q in ai_queries:
        if q.get("doc_id_source") in target_ai_doc_ids:
            selected_queries.append(q)
            
    for q in fintech_queries:
        if q.get("doc_id_source") in target_fintech_doc_ids:
            selected_queries.append(q)
            
    print(f"Selected {len(selected_queries)} total queries ({sum(1 for q in selected_queries if q.get('doc_id_source') in target_ai_doc_ids)} AI + {sum(1 for q in selected_queries if q.get('doc_id_source') in target_fintech_doc_ids)} Fintech).")

    # 2. Select 15 unused distractor papers from AI and 15 unused from Fintech
    all_ai_stress_doc_ids = set(os.path.basename(f).replace("_chunks.json", "") for f in ai_chunks_files)
    all_fintech_stress_doc_ids = set(os.path.basename(f).replace("_chunks.json", "") for f in fintech_chunks_files)
    
    raw_ai_dir = os.path.join(base_dir, "data", "raw", "latest_arxiv", "ai")
    raw_fintech_dir = os.path.join(base_dir, "data", "raw", "latest_arxiv", "fintech")
    
    raw_ai_files = sorted(glob.glob(os.path.join(raw_ai_dir, "*.txt")))
    raw_fintech_files = sorted(glob.glob(os.path.join(raw_fintech_dir, "*.txt")))
    
    tokenizer = get_tokenizer()
    
    distractor_ai_count = 0
    for txt_path in raw_ai_files:
        doc_id = os.path.basename(txt_path).replace(".txt", "")
        if doc_id not in all_ai_stress_doc_ids and distractor_ai_count < 15:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            doc_data = process_document(content, doc_id=doc_id, tokenizer=tokenizer)
            with open(os.path.join(out_chunks_dir, f"{doc_id}_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2)
            distractor_ai_count += 1

    distractor_fintech_count = 0
    for txt_path in raw_fintech_files:
        doc_id = os.path.basename(txt_path).replace(".txt", "")
        if doc_id not in all_fintech_stress_doc_ids and distractor_fintech_count < 15:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            doc_data = process_document(content, doc_id=doc_id, tokenizer=tokenizer)
            with open(os.path.join(out_chunks_dir, f"{doc_id}_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2)
            distractor_fintech_count += 1
            
    print(f"Processed {distractor_ai_count} AI distractor papers and {distractor_fintech_count} Fintech distractor papers.")
    
    # 3. Save final benchmark JSON
    final_benchmark_path = os.path.join(out_dir, "final_benchmark_corpus_stress_50.json")
    with open(final_benchmark_path, "w", encoding="utf-8") as f:
        json.dump(selected_queries, f, indent=2)
        
    total_files = len(glob.glob(os.path.join(out_chunks_dir, "*_chunks.json")))
    print(f"SUCCESS: Built 50-paper fused benchmark in '{out_dir}' with {total_files} chunk files and {len(selected_queries)} ground-truth queries!")

if __name__ == "__main__":
    main()
