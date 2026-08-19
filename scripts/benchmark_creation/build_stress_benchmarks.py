import os
import sys
import json
import shutil
import glob
from typing import List, Dict, Any, Set

# Ensure project root & scripts/benchmark_creation are in Python path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from scripts.benchmark_creation.step1_chunking import process_document, get_tokenizer

def build_single_domain_stress(domain: str, target_count: int = 100):
    """Builds a single-domain stress benchmark (e.g. ai/corpus_stress_100 or fintech/corpus_stress_100)."""
    raw_dir = os.path.join(base_dir, "data", "raw", "latest_arxiv", domain)
    if not os.path.exists(raw_dir):
        print(f"[WARNING] Raw directory for '{domain}' not found at {raw_dir}")
        return

    src_stress_1_dir = os.path.join(base_dir, "data", "benchmarks", domain, "corpus_stress_1")
    out_dir = os.path.join(base_dir, "data", "benchmarks", domain, f"corpus_stress_{target_count}")
    out_chunks_dir = os.path.join(out_dir, "step1_chunks")
    os.makedirs(out_chunks_dir, exist_ok=True)

    print(f"\n--- Building {domain.upper()} Corpus Stress {target_count} ---")

    # Load queries from stress_1 if available
    queries = []
    stress_1_json = os.path.join(src_stress_1_dir, "final_benchmark_corpus_stress_1.json")
    if os.path.exists(stress_1_json):
        with open(stress_1_json, "r", encoding="utf-8") as f:
            queries = json.load(f)

    # Copy existing chunk files from stress_1
    existing_chunks = glob.glob(os.path.join(src_stress_1_dir, "step1_chunks", "*_chunks.json"))
    processed_doc_ids = set()
    for filepath in existing_chunks:
        filename = os.path.basename(filepath)
        doc_id = filename.replace("_chunks.json", "")
        processed_doc_ids.add(doc_id)
        shutil.copy(filepath, os.path.join(out_chunks_dir, filename))

    # Chunk remaining raw text files
    raw_txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    tokenizer = get_tokenizer()
    added_count = 0

    for txt_path in raw_txt_files:
        if len(processed_doc_ids) >= target_count:
            break
        doc_id = os.path.basename(txt_path).replace(".txt", "")
        if doc_id in processed_doc_ids:
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            continue

        doc_data = process_document(content, doc_id=doc_id, tokenizer=tokenizer)
        out_chunk_path = os.path.join(out_chunks_dir, f"{doc_id}_chunks.json")
        with open(out_chunk_path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2)

        processed_doc_ids.add(doc_id)
        added_count += 1

    # Save final benchmark JSON
    final_json_path = os.path.join(out_dir, f"final_benchmark_corpus_stress_{target_count}.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    total_chunks = len(glob.glob(os.path.join(out_chunks_dir, "*_chunks.json")))
    print(f"[OK] {domain.upper()} stress_{target_count} complete: {total_chunks} chunk files, {len(queries)} queries.")


def build_fused_stress(target_count: int, domain_counts: Dict[str, int]):
    """Builds a fused multi-domain stress benchmark (e.g. fused/corpus_stress_200 or fused/corpus_stress_500)."""
    out_dir = os.path.join(base_dir, "data", "benchmarks", "fused", f"corpus_stress_{target_count}")
    out_chunks_dir = os.path.join(out_dir, "step1_chunks")
    os.makedirs(out_chunks_dir, exist_ok=True)

    print(f"\n--- Building FUSED Corpus Stress {target_count} ---")

    selected_queries = []
    for d in domain_counts.keys():
        d_stress_dir = os.path.join(base_dir, "data", "benchmarks", d, "corpus_stress_1")
        d_json = os.path.join(d_stress_dir, "final_benchmark_corpus_stress_1.json")
        if os.path.exists(d_json):
            with open(d_json, "r", encoding="utf-8") as f:
                domain_queries = json.load(f)
                selected_queries.extend(domain_queries)
                print(f" -> Loaded {len(domain_queries)} gold queries for domain '{d}'.")

    processed_doc_ids: Set[str] = set()
    tokenizer = get_tokenizer()

    # Process documents for each specified domain
    for domain, count in domain_counts.items():
        raw_dir = os.path.join(base_dir, "data", "raw", "latest_arxiv", domain)
        src_stress_1_dir = os.path.join(base_dir, "data", "benchmarks", domain, "corpus_stress_1")
        
        # 1. First copy existing target chunk files from corpus_stress_1
        existing_chunks = glob.glob(os.path.join(src_stress_1_dir, "step1_chunks", "*_chunks.json"))
        domain_added = 0
        for filepath in existing_chunks:
            if domain_added >= count:
                break
            filename = os.path.basename(filepath)
            doc_id = filename.replace("_chunks.json", "")
            if doc_id not in processed_doc_ids:
                processed_doc_ids.add(doc_id)
                shutil.copy(filepath, os.path.join(out_chunks_dir, filename))
                domain_added += 1

        # 2. Fill remaining paper count from raw text files
        if os.path.exists(raw_dir):
            raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
            for txt_path in raw_files:
                if domain_added >= count:
                    break
                doc_id = os.path.basename(txt_path).replace(".txt", "")
                if doc_id in processed_doc_ids:
                    continue

                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                doc_data = process_document(content, doc_id=doc_id, tokenizer=tokenizer)
                out_chunk_path = os.path.join(out_chunks_dir, f"{doc_id}_chunks.json")
                with open(out_chunk_path, "w", encoding="utf-8") as f:
                    json.dump(doc_data, f, indent=2)

                processed_doc_ids.add(doc_id)
                domain_added += 1

        print(f" -> Category '{domain}': Added {domain_added}/{count} paper chunk files.")

    # Save final benchmark JSON
    final_json_path = os.path.join(out_dir, f"final_benchmark_corpus_stress_{target_count}.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(selected_queries, f, indent=2)

    total_chunks = len(glob.glob(os.path.join(out_chunks_dir, "*_chunks.json")))
    print(f"[OK] Fused stress_{target_count} complete: {total_chunks} chunk files, {len(selected_queries)} queries.")


def main():
    print("=== Edge-RAG Stress Benchmark Creator ===")
    
    # 1. Build single-domain stress_100 benchmarks for all 4 fields
    build_single_domain_stress("ai", 100)
    build_single_domain_stress("fintech", 100)
    build_single_domain_stress("biomedical", 100)
    build_single_domain_stress("systems_security", 100)
    
    # 2. Build fused stress_200 benchmark (100 AI + 100 Fintech)
    build_fused_stress(200, {"ai": 100, "fintech": 100})
    
    # 3. Build fused stress_500 benchmark (125 AI + 125 Fintech + 125 Biomedical + 125 Systems/Security)
    build_fused_stress(500, {
        "ai": 125,
        "fintech": 125,
        "biomedical": 125,
        "systems_security": 125
    })

if __name__ == "__main__":
    main()
