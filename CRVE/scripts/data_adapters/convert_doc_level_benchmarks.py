#!/usr/bin/env python3
"""
Document-Level Benchmark Converter for EnterpriseRAG and LiveRAG.
Extracts original, un-chunked document collections and standardized query evaluation files
for first-stage retrieval evaluation (BM25, Dense, SPLADE, Edge-RAG Schemas).
"""

import os
import sys
import json
import glob
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_enterpriserag_doc_level(target_total_docs=50000):
    random.seed(42)
    print("=== Converting EnterpriseRAG (Document-Level) ===")
    
    docs_file = os.path.join(BASE_DIR, "data", "raw", "enterpriserag_bench", "data", "documents", "test.parquet")
    quests_file = os.path.join(BASE_DIR, "data", "raw", "enterpriserag_bench", "data", "questions", "test.parquet")
    
    if not os.path.exists(docs_file) or not os.path.exists(quests_file):
        print(f"[ERROR] EnterpriseRAG files not found at {docs_file} or {quests_file}")
        return

    print("  Loading questions parquet...")
    q_df = pd.read_parquet(quests_file)
    quest_records = q_df.to_dict(orient="records")
    print(f"  Loaded {len(quest_records)} question records.")

    out_dir = os.path.join(BASE_DIR, "data", "benchmarks", "enterpriserag_doc_level")
    os.makedirs(out_dir, exist_ok=True)

    # Collect ALL gold doc IDs across all queries
    all_gold_dids = set()
    for q in quest_records:
        expected = q.get("expected_doc_ids", [])
        if isinstance(expected, str):
            all_gold_dids.add(expected)
        elif hasattr(expected, "__iter__"):
            for e in expected:
                all_gold_dids.add(str(e))
    all_gold_dids.discard("")

    print(f"  Identified {len(all_gold_dids)} unique gold document IDs across queries.")
    print(f"  Loading documents parquet (selecting {target_total_docs} documents)...")
    d_df = pd.read_parquet(docs_file, columns=["doc_id", "content"])
    
    all_doc_ids = set(d_df["doc_id"].astype(str))
    gold_doc_ids = sorted(all_gold_dids.intersection(all_doc_ids))
    noise_pool = sorted(all_doc_ids - all_gold_dids)

    needed_noise = max(0, target_total_docs - len(gold_doc_ids))
    selected_noise = random.sample(noise_pool, min(needed_noise, len(noise_pool)))
    selected_doc_ids_set = set(gold_doc_ids + selected_noise)

    print(f"  Filtering to {len(gold_doc_ids)} gold + {len(selected_noise)} noise = {len(selected_doc_ids_set)} total docs...")
    sampled_df = d_df[d_df["doc_id"].isin(selected_doc_ids_set)].copy()

    docs_dir = os.path.join(out_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    # Clean existing docs in documents/ if any
    for existing_f in os.listdir(docs_dir):
        if existing_f.endswith(".json"):
            try:
                os.remove(os.path.join(docs_dir, existing_f))
            except Exception:
                pass

    print(f"  Writing {len(sampled_df)} document JSON files to {docs_dir}...")
    for r in sampled_df.to_dict(orient="records"):
        did = str(r["doc_id"])
        text = str(r["content"])
        doc_data = {
            "doc_id": did,
            "text": text
        }
        with open(os.path.join(docs_dir, f"{did}.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f)

    # Clean up corpus.parquet if present
    corpus_pq = os.path.join(out_dir, "corpus.parquet")
    if os.path.exists(corpus_pq):
        os.remove(corpus_pq)

    doc_map_by_id = set(sampled_df["doc_id"].astype(str))

    def format_queries(q_list, q_prefix="q_ent"):
        formatted = []
        for idx, q_rec in enumerate(q_list):
            query_id = str(q_rec.get("question_id", f"{q_prefix}_{idx}"))
            question = q_rec.get("question", "")
            golden_answer = q_rec.get("gold_answer", q_rec.get("answer", ""))
            expected_docs = q_rec.get("expected_doc_ids", [])
            if isinstance(expected_docs, str):
                expected_docs = [expected_docs]
            elif hasattr(expected_docs, "__iter__"):
                expected_docs = [str(e) for e in expected_docs]
            else:
                expected_docs = []

            # Clean gold docs that exist in corpus
            valid_gold_docs = [did for did in expected_docs if (doc_map_by_id is None or did in doc_map_by_id)]
            first_doc = valid_gold_docs[0] if valid_gold_docs else "doc_0"

            formatted.append({
                "query_id": query_id,
                "query_group": "Enterprise RAG",
                "query_type": q_rec.get("question_type", "Factual"),
                "question": question,
                "raw_question": question,
                "golden_answer": golden_answer,
                "doc_id_source": first_doc,
                "expected_doc_ids": valid_gold_docs,
                "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in valid_gold_docs]  # backward compat
            })
        return formatted

    full_formatted = format_queries(quest_records)

    with open(os.path.join(out_dir, "final_benchmark_capped.json"), "w", encoding="utf-8") as f:
        json.dump(full_formatted, f, indent=2)

    with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
        json.dump(full_formatted, f, indent=2)

    with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(full_formatted, f, indent=2)

    total_docs_count = len(sampled_df)
    print(f"  Saved EnterpriseRAG Doc-Level: {total_docs_count} docs, {len(full_formatted)} queries.")


def convert_liverag_doc_level():
    random.seed(42)
    print("\n=== Converting LiveRAG (Document-Level) ===")
    
    raw_file = os.path.join(BASE_DIR, "data", "raw", "liverag_bench", "LiveRAG_banchmark_20250910.parquet")
    if not os.path.exists(raw_file):
        print(f"[ERROR] LiveRAG file not found at {raw_file}")
        return

    print("  Loading LiveRAG parquet...")
    df = pd.read_parquet(raw_file)
    records = df.to_dict(orient="records")
    print(f"  Loaded {len(records)} LiveRAG records.")

    def _parse_supp_docs(raw_val):
        if raw_val is None:
            return []
        if isinstance(raw_val, str):
            try:
                parsed = json.loads(raw_val)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except:
                return []
        if isinstance(raw_val, dict):
            return [raw_val]
        # Handle list, numpy ndarray, or any iterable of dicts/strings
        if isinstance(raw_val, (list, np.ndarray)) or hasattr(raw_val, "__iter__"):
            res = []
            for item in raw_val:
                if isinstance(item, dict):
                    res.append(item)
                elif isinstance(item, str):
                    try:
                        p = json.loads(item)
                        if isinstance(p, dict):
                            res.append(p)
                    except:
                        pass
            return res
        return []

    doc_map_by_id = {}
    doc_sessions = {}  # doc_id -> set of collection windows ("First"/"Second"/"Both")
    for rec in records:
        session = str(rec.get("Session", "")).strip()
        supp_docs = _parse_supp_docs(rec.get("Supporting_Documents"))
        for d in supp_docs:
            did = str(d.get("doc_id", ""))
            text = d.get("content", d.get("text", ""))
            if did and text:
                doc_sessions.setdefault(did, set()).add(session)
                if did not in doc_map_by_id:
                    doc_map_by_id[did] = {
                        "doc_id": did,
                        "text": text,
                        "title": d.get("title", ""),
                        "doc_length": len(text.split())
                    }

    print(f"  LiveRAG: Extracted {len(doc_map_by_id)} unique un-chunked supporting documents.")

    # Generate full document collection (all extracted supporting documents)
    out_dir = os.path.join(BASE_DIR, "data", "benchmarks", "liverag_doc_level")
    docs_dir = os.path.join(out_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    for did, doc_data in doc_map_by_id.items():
        doc_data["sessions"] = sorted(doc_sessions.get(did, []))
        with open(os.path.join(docs_dir, f"{did}.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2)

    # Format Queries
    formatted_queries = []
    for idx, rec in enumerate(records):
        question = rec.get("Question", rec.get("question", ""))
        answer = rec.get("Answer", rec.get("answer", ""))
        session = str(rec.get("Session", "")).strip()
        supp_docs = _parse_supp_docs(rec.get("Supporting_Documents"))

        gold_dids = [str(d.get("doc_id", "")) for d in supp_docs if str(d.get("doc_id", "")) in doc_map_by_id]
        first_did = gold_dids[0] if gold_dids else f"liverag_doc_{idx}"

        formatted_queries.append({
            "query_id": f"q_live_{rec.get('Index', idx)}",
            "query_group": "LiveRAG Streaming",
            "query_type": "Factual",
            "session": session,
            "question": question,
            "raw_question": question,
            "golden_answer": answer,
            "doc_id_source": first_did,
            "expected_doc_ids": gold_dids,
            "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in gold_dids]
        })

    with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)

    with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)

    print(f"  Saved LiveRAG Doc-Level: {len(doc_map_by_id)} docs, {len(formatted_queries)} queries.")


if __name__ == "__main__":
    convert_enterpriserag_doc_level()
    convert_liverag_doc_level()
    print("\n✅ Document-level benchmark conversion completed successfully.")
