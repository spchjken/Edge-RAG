import os
import glob
import json

def check_multi_block_core():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    benchmark_files = glob.glob(os.path.join(project_root, "data", "benchmarks", "*", "corpus_stress_1", "final_benchmark_corpus_stress_1.json"))

    print("=== Multi-Block Core Chunk Inspector ===")

    total_queries_checked = 0
    multi_block_queries = []

    for bfile in benchmark_files:
        domain = bfile.split(os.sep)[-3]
        with open(bfile, "r", encoding="utf-8") as f:
            queries = json.load(f)

        domain_multi = 0
        for q in queries:
            total_queries_checked += 1
            gt_chunks = q.get("ground_truth_child_chunks", [])
            p_source = q.get("parent_id_source", "")

            # Extract parent blocks represented in ground_truth_child_chunks
            parent_blocks = set()
            for c in gt_chunks:
                cid = c.get("chunk_id", "")
                if "_chunk" in cid:
                    p_block = cid.rsplit("_chunk", 1)[0]
                elif "_bridge" in cid:
                    p_block = cid.rsplit("_bridge", 1)[0]
                else:
                    p_block = c.get("parent_id", "")
                if p_block:
                    parent_blocks.add(p_block)

            if len(parent_blocks) > 1:
                domain_multi += 1
                multi_block_queries.append({
                    "domain": domain,
                    "query_id": q["query_id"],
                    "parent_id_source": p_source,
                    "parent_blocks_found": list(parent_blocks),
                    "core_chunk_count": len(gt_chunks)
                })

        print(f"Domain '{domain}': Checked {len(queries)} queries -> {domain_multi} queries have core chunks from >1 parent block.")

    print(f"\nTotal Queries Checked: {total_queries_checked}")
    print(f"Total Multi-Block Core Queries: {len(multi_block_queries)}")

    if multi_block_queries:
        print("\nDetails of Multi-Block Core Queries:")
        for idx, item in enumerate(multi_block_queries[:10]):
            print(f" [{idx+1}] [{item['domain']}] Query {item['query_id']}: Source Parent='{item['parent_id_source']}'")
            print(f"      Core Chunks Span Blocks: {item['parent_blocks_found']}")

if __name__ == "__main__":
    check_multi_block_core()
