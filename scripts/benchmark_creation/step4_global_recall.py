import os
import glob
import json
import time
import argparse
import numpy as np

try:
    import torch
    from FlagEmbedding import FlagModel
except ImportError:
    print("Please install torch and FlagEmbedding to run this script.")
    exit(1)

from rank_bm25 import BM25Okapi
import re

def bm25_tokenize(text: str):
    return re.findall(r'\w+', text.lower())

def load_corpus(processed_dir: str):
    """
    Loads all child chunks from the processed directory into a flat list.
    Returns:
        corpus_docs: list of dicts with keys (chunk_id, doc_id, text, parent_id)
        corpus_texts: list of str (just the text for indexing)
    """
    chunk_files = glob.glob(os.path.join(processed_dir, "*_chunks.json"))
    corpus_docs = []
    corpus_texts = []
    
    for fpath in chunk_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            child_chunks = data.get("child_chunks", [])
            for chunk in child_chunks:
                # Add to corpus
                doc_payload = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "parent_id": chunk["parent_id"],
                    "text": chunk["text"]
                }
                corpus_docs.append(doc_payload)
                corpus_texts.append(chunk["text"])
                
    return corpus_docs, corpus_texts

def load_queries(paraphrased_dir: str):
    """
    Loads all paraphrased queries.
    Returns:
        queries: list of dicts (the query payloads)
    """
    query_files = glob.glob(os.path.join(paraphrased_dir, "*_paraphrased.json"))
    queries = []
    
    for fpath in query_files:
        with open(fpath, "r", encoding="utf-8") as f:
            file_queries = json.load(f)
            queries.extend(file_queries)
            
    return queries

def rrf_score(rank1, rank2, k=60):
    """Reciprocal Rank Fusion"""
    return (1.0 / (k + rank1)) + (1.0 / (k + rank2))

def compute_rrf_rankings(dense_scores, sparse_scores, top_k=100):
    """
    Given an array of dense scores and sparse scores for all chunks,
    compute ranks and merge using RRF.
    Returns indices of the top-k documents.
    """
    # Sort indices by score (descending)
    dense_ranked_indices = np.argsort(dense_scores)[::-1]
    sparse_ranked_indices = np.argsort(sparse_scores)[::-1]
    
    # Create rank lookups
    # rank_lookup[doc_idx] = rank (1-indexed)
    dense_ranks = np.empty_like(dense_ranked_indices)
    dense_ranks[dense_ranked_indices] = np.arange(1, len(dense_scores) + 1)
    
    sparse_ranks = np.empty_like(sparse_ranked_indices)
    sparse_ranks[sparse_ranked_indices] = np.arange(1, len(sparse_scores) + 1)
    
    # Compute RRF
    rrf_scores = np.zeros_like(dense_scores, dtype=float)
    for i in range(len(dense_scores)):
        rrf_scores[i] = rrf_score(dense_ranks[i], sparse_ranks[i])
        
    # Get top-k based on RRF
    top_rrf_indices = np.argsort(rrf_scores)[::-1][:top_k]
    
    return top_rrf_indices.tolist(), [rrf_scores[idx] for idx in top_rrf_indices]

def main():
    parser = argparse.ArgumentParser(description="Step 4: Offline Global Recall Pass")
    parser.add_argument("--processed-dir", type=str, default="../../tmp_test_ai/processed", 
                        help="Directory containing output child chunks from Step 1")
    parser.add_argument("--paraphrased-dir", type=str, default="../../tmp_test_ai/paraphrased", 
                        help="Directory containing paraphrased queries from Step 3")
    parser.add_argument("--output-file", type=str, default="../../tmp_test_ai/retrieval_candidates.json", 
                        help="File to save the Top-K retrieval candidates")
    parser.add_argument("--top-k", type=int, default=100, 
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-m3", 
                        help="HuggingFace model ID for dense retrieval")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, args.processed_dir))
    paraphrased_dir = os.path.abspath(os.path.join(script_dir, args.paraphrased_dir))
    out_file = os.path.abspath(os.path.join(script_dir, args.output_file))
    
    # 1. Load Data
    print("[INFO] Loading corpus and queries...")
    corpus_docs, corpus_texts = load_corpus(processed_dir)
    queries = load_queries(paraphrased_dir)
    
    if not corpus_docs:
        print("[ERROR] No chunks found in processed_dir.")
        return
    if not queries:
        print("[ERROR] No queries found in paraphrased_dir.")
        return
        
    print(f"       -> Loaded {len(corpus_docs)} child chunks.")
    print(f"       -> Loaded {len(queries)} queries.")
    
    # 2. Initialize Models
    print("\n[INFO] Initializing Retrieval Models...")
    
    # BM25
    t0 = time.time()
    tokenized_corpus = [bm25_tokenize(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"       -> BM25 Initialized in {time.time()-t0:.2f}s")
    
    # Dense (BGE-M3)
    t0 = time.time()
    # Using fp16 reduces VRAM usage significantly. 
    dense_model = FlagModel(args.dense_model, 
                            query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query",
                            use_fp16=True)
    print(f"       -> {args.dense_model} Initialized in {time.time()-t0:.2f}s")
    if torch.cuda.is_available():
        print(f"       -> VRAM Used (Post-Init): {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Pre-compute Dense Embeddings for the entire Corpus
    print("\n[INFO] Computing Dense Embeddings for the Global Corpus...")
    t0 = time.time()
    # Batch size set strictly to 32 to keep VRAM usage low.
    c_embeddings = np.asarray(dense_model.encode(corpus_texts, batch_size=32))
    print(f"       -> Corpus Embedded in {time.time()-t0:.2f}s")
    if torch.cuda.is_available():
        print(f"       -> Peak VRAM Used: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

    # 2.5 Deduplicate Queries (DC.1)
    print("\n[INFO] Deduplicating queries via embedding similarity...")
    q_texts = [q["paraphrased_question"] for q in queries]
    q_embeddings = np.asarray(dense_model.encode(q_texts))
    
    norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
    norm_embeddings = q_embeddings / np.maximum(norms, 1e-9)
    sim_matrix = norm_embeddings @ norm_embeddings.T
    
    to_drop = set()
    for i in range(len(queries)):
        if i in to_drop: continue
        for j in range(i + 1, len(queries)):
            if j in to_drop: continue
            if sim_matrix[i, j] > 0.90:
                to_drop.add(j)
                print(f"       -> Dropped query {queries[j]['query_id']} (Duplicate of {queries[i]['query_id']})")
                
    queries = [q for i, q in enumerate(queries) if i not in to_drop]
    print(f"       -> Kept {len(queries)} queries.")

    # 3. Process Queries
    print(f"\n[INFO] Retrieving Top-{args.top_k} chunks for {len(queries)} queries...")
    output_results = []
    
    t_start = time.time()
    for idx, q in enumerate(queries):
        query_text_para = q["paraphrased_question"]
        query_text_raw = q["raw_question"]
        
        # Sparse Retrieval (D4.1: Para + Raw)
        tokenized_q_para = bm25_tokenize(query_text_para)
        tokenized_q_raw = bm25_tokenize(query_text_raw)
        
        sparse_scores_para = bm25.get_scores(tokenized_q_para)
        sparse_scores_raw = bm25.get_scores(tokenized_q_raw)
        sparse_scores_combined = np.maximum(sparse_scores_para, sparse_scores_raw)
        
        # Dense Retrieval (D4.1: Para + Raw)
        q_emb_para = np.asarray(dense_model.encode_queries([query_text_para]))
        q_emb_raw = np.asarray(dense_model.encode_queries([query_text_raw]))
        
        dense_scores_para = np.asarray(c_embeddings @ q_emb_para.T).squeeze(-1)
        dense_scores_raw = np.asarray(c_embeddings @ q_emb_raw.T).squeeze(-1)
        dense_scores_combined = np.maximum(dense_scores_para, dense_scores_raw)
        
        # Fusion
        top_indices, _ = compute_rrf_rankings(dense_scores_combined, sparse_scores_combined, top_k=args.top_k)
        
        # Build candidate list
        retrieved_chunks = []
        retrieved_chunk_ids = set()
        for rank, c_idx in enumerate(top_indices):
            c_id = corpus_docs[c_idx]["chunk_id"]
            retrieved_chunk_ids.add(c_id)
            retrieved_chunks.append({
                "rank": rank + 1,
                "chunk_id": c_id,
                "parent_id": corpus_docs[c_idx]["parent_id"],
                "doc_id": corpus_docs[c_idx]["doc_id"],
                "text": corpus_docs[c_idx]["text"]
            })

        # Force-include core source chunks belonging to parent_id_source if missed by Top-K
        parent_source = q.get("parent_id")
        if parent_source:
            for c_idx, doc in enumerate(corpus_docs):
                if doc["parent_id"] == parent_source and doc["chunk_id"] not in retrieved_chunk_ids:
                    retrieved_chunks.append({
                        "rank": len(retrieved_chunks) + 1,
                        "chunk_id": doc["chunk_id"],
                        "parent_id": doc["parent_id"],
                        "doc_id": doc["doc_id"],
                        "text": doc["text"]
                    })
                    retrieved_chunk_ids.add(doc["chunk_id"])
            
        result_payload = {
            "query_id": q["query_id"],
            "query_group": q.get("query_group"),
            "query_type": q.get("query_type"),
            "raw_question": query_text_raw,
            "paraphrased_question": query_text_para,
            "golden_answer": q.get("golden_answer"),
            "evidence_quotes": q.get("evidence_quotes", []),
            "doc_id_source": q.get("doc_id"),
            "parent_id_source": q.get("parent_id"),
            "retrieved_candidates": retrieved_chunks
        }
        output_results.append(result_payload)
        
        if (idx + 1) % 5 == 0:
            print(f"       -> Processed {idx + 1}/{len(queries)} queries...")

    print(f"       -> Finished all retrievals in {time.time()-t_start:.2f}s")

    # 4. Output Results
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Saved retrieved candidates to {out_file}")

if __name__ == "__main__":
    main()
