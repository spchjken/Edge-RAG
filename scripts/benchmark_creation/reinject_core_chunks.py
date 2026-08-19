import os
import sys
import glob
import json
import re
import argparse
from typing import List, Dict, Any

def compute_chunk_relevance_score(chunk_text: str, golden_answer: str, evidence_quotes: List[str], raw_question: str) -> float:
    """
    Computes a multi-tier relevance score between a child chunk text
    and the golden answer, evidence quotes, and raw question.
    """
    chunk_clean = " ".join(chunk_text.split()).lower()
    chunk_words = set(re.findall(r'\w+', chunk_clean))
    score = 0.0

    # Tier 1: Evidence Quote Match (Weight: up to +10.0)
    if evidence_quotes:
        for eq in evidence_quotes:
            eq_clean = " ".join(eq.split()).lower()
            if not eq_clean:
                continue
            if eq_clean in chunk_clean:
                score += 10.0
            elif len(eq_clean) > 20 and (eq_clean[:25] in chunk_clean or eq_clean[-25:] in chunk_clean):
                score += 8.0
            else:
                eq_words = set(re.findall(r'\w+', eq_clean))
                if eq_words:
                    overlap = len(eq_words.intersection(chunk_words)) / len(eq_words)
                    score += overlap * 5.0

    # Tier 2: Golden Answer Word & Bi-gram Overlap (Weight: up to +5.0)
    golden_clean = " ".join(golden_answer.split()).lower()
    g_words = set(re.findall(r'\w+', golden_clean))
    if g_words and chunk_words:
        jaccard = len(g_words.intersection(chunk_words)) / len(g_words)
        score += jaccard * 5.0

    # Tier 3: Question Word Overlap (Weight: up to +2.0)
    q_words = set(re.findall(r'\w+', raw_question.lower()))
    if q_words and chunk_words:
        q_overlap = len(q_words.intersection(chunk_words)) / len(q_words)
        score += q_overlap * 2.0

    return score


def reinject_core_chunks_file(benchmark_json_path: str, chunks_dir: str) -> int:
    """
    Inspects a benchmark JSON file. Uses multi-tier scoring to select the exact,
    most relevant child chunk(s) derived from parent_id_source that contain the golden answer,
    with a guaranteed top-scoring fallback.
    """
    if not os.path.exists(benchmark_json_path):
        print(f"[SKIP] Benchmark file not found: {benchmark_json_path}")
        return 0

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Pre-cache child chunks by parent_id from step1_chunks/
    chunk_files = glob.glob(os.path.join(chunks_dir, "*_chunks.json"))
    parent_to_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for cfile in chunk_files:
        with open(cfile, "r", encoding="utf-8") as f:
            data = json.load(f)
            child_chunks = data.get("child_chunks", [])
            for chunk in child_chunks:
                p_id = chunk.get("parent_id")
                if p_id:
                    if p_id not in parent_to_chunks:
                        parent_to_chunks[p_id] = []
                    parent_to_chunks[p_id].append({
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "rank": 1
                    })

    fixed_count = 0
    for q in queries:
        p_source = q.get("parent_id_source")
        if not p_source or p_source not in parent_to_chunks:
            continue

        candidate_core_chunks = parent_to_chunks[p_source]
        golden_answer = q.get("golden_answer", "")
        evidence_quotes = q.get("evidence_quotes", [])
        raw_question = q.get("raw_question", "")

        # Score all child chunks derived from parent_id_source
        scored_chunks = []
        for c in candidate_core_chunks:
            s = compute_chunk_relevance_score(c["text"], golden_answer, evidence_quotes, raw_question)
            scored_chunks.append((s, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_chunks[0][0]

        # Select all high-scoring child chunks, or fallback guarantee to top-1
        verified_core_chunks = [c for s, c in scored_chunks if s > 0.5 and (s >= best_score * 0.7 or s >= 5.0)]
        if not verified_core_chunks:
            # Fallback Guarantee: Always take the top-1 highest scoring child chunk from parent_id_source
            verified_core_chunks = [scored_chunks[0][1]]

        existing_gt_chunks = q.get("ground_truth_child_chunks", [])
        existing_ids = {c["chunk_id"] for c in existing_gt_chunks}

        missing_core_chunks = [c for c in verified_core_chunks if c["chunk_id"] not in existing_ids]

        if missing_core_chunks:
            updated_gt = existing_gt_chunks + missing_core_chunks
            q["ground_truth_child_chunks"] = updated_gt
            fixed_count += 1
            added_chunk_ids = [c["chunk_id"] for c in missing_core_chunks]
            print(f" -> Query {q['query_id']}: Added {len(missing_core_chunks)} missing core chunk(s) {added_chunk_ids} (score: {best_score:.1f}) for parent '{p_source}'")

    if fixed_count > 0:
        with open(benchmark_json_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2)
        print(f"[SUCCESS] Repaired {benchmark_json_path}: Injected exact core child chunks for {fixed_count} queries.\n")
    else:
        print(f"[OK] All queries in {benchmark_json_path} already contain 100% complete core child chunks.\n")

    return fixed_count


def reinject_all_benchmarks(base_dir: str):
    """Scan and repair all final_benchmark_*.json files under data/benchmarks."""
    benchmark_files = glob.glob(os.path.join(base_dir, "data", "benchmarks", "**", "final_benchmark_*.json"), recursive=True)
    total_repaired = 0

    print(f"=== Edge-RAG Guaranteed Core Chunk Re-Injector ===")
    print(f"Found {len(benchmark_files)} benchmark JSON files to check.")

    for bfile in benchmark_files:
        parent_dir = os.path.dirname(bfile)
        chunks_dir = os.path.join(parent_dir, "step1_chunks")
        if not os.path.exists(chunks_dir):
            chunks_dir = parent_dir
        total_repaired += reinject_core_chunks_file(bfile, chunks_dir)

    print(f"=== Complete: Successfully repaired {total_repaired} queries across all benchmarks ===")


def main():
    parser = argparse.ArgumentParser(description="Re-inject missing core ground truth chunks into benchmark JSON files")
    parser.add_argument("--benchmark-file", type=str, default=None, help="Path to specific final_benchmark_*.json file")
    parser.add_argument("--chunks-dir", type=str, default=None, help="Path to step1_chunks directory")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.benchmark_file:
        bfile = os.path.abspath(args.benchmark_file)
        cdir = os.path.abspath(args.chunks_dir) if args.chunks_dir else os.path.join(os.path.dirname(bfile), "step1_chunks")
        reinject_core_chunks_file(bfile, cdir)
    else:
        reinject_all_benchmarks(project_root)

if __name__ == "__main__":
    main()
