import os
import glob
import json

def verify_paper_inclusion():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    benchmark_files = glob.glob(os.path.join(project_root, "data", "benchmarks", "**", "final_benchmark_corpus_stress_*.json"), recursive=True)

    print("=== Paper & Core Chunk Inclusion Verification ===")
    print(f"Found {len(benchmark_files)} stress benchmark JSON files to verify.\n")

    all_passed = True

    for bfile in sorted(benchmark_files):
        rel_path = os.path.relpath(bfile, project_root)
        chunks_dir = os.path.join(os.path.dirname(bfile), "step1_chunks")

        if not os.path.exists(chunks_dir):
            print(f"[FAIL] Chunks directory missing for {rel_path}: {chunks_dir}")
            all_passed = False
            continue

        existing_chunk_files = glob.glob(os.path.join(chunks_dir, "*_chunks.json"))
        existing_doc_ids = {os.path.basename(f).replace("_chunks.json", "") for f in existing_chunk_files}

        # Cache all chunk_ids present in existing chunk files
        existing_chunk_ids = set()
        for f in existing_chunk_files:
            try:
                with open(f, "r", encoding="utf-8") as cf:
                    cdata = json.load(cf)
                    for chunk in cdata.get("child_chunks", []):
                        existing_chunk_ids.add(chunk["chunk_id"])
            except Exception:
                pass

        with open(bfile, "r", encoding="utf-8") as f:
            queries = json.load(f)

        missing_docs = set()
        missing_core_chunks = set()

        for q in queries:
            doc_src = q.get("doc_id_source", "")
            if doc_src and doc_src not in existing_doc_ids:
                missing_docs.add(doc_src)

            gt_chunks = q.get("ground_truth_child_chunks", [])
            for c in gt_chunks:
                cid = c.get("chunk_id", "")
                if cid and cid not in existing_chunk_ids:
                    missing_core_chunks.add(cid)

        if missing_docs or missing_core_chunks:
            all_passed = False
            print(f"[FAIL] {rel_path}:")
            if missing_docs:
                print(f"   -> Missing {len(missing_docs)} doc_id_source papers in step1_chunks: {list(missing_docs)[:3]}")
            if missing_core_chunks:
                print(f"   -> Missing {len(missing_core_chunks)} core chunk IDs on disk: {list(missing_core_chunks)[:3]}")
        else:
            print(f"[PASS] {rel_path}: 100% of {len(queries)} queries & core chunks verified in {len(existing_doc_ids)} chunk files.")

    print("\n==========================================")
    if all_passed:
        print("🎉 ALL STRESS BENCHMARKS VERIFIED SUCCESSFULLY: 100% Paper & Core Chunk Inclusion!")
    else:
        print("❌ VERIFICATION FAILED: Some paper/chunk files are missing.")
    print("==========================================")

if __name__ == "__main__":
    verify_paper_inclusion()
