#!/usr/bin/env python3
"""
scripts/data_adapters/convert_retriever_doc_level_benchmarks.py

Standardizes raw retriever benchmark datasets into the canonical Edge-RAG 
Document-Level format (matching enterpriserag_doc_level and liverag_doc_level):

1. BEIR Subsets (beir_scifact_doc_level, beir_nfcorpus_doc_level, beir_fiqa_doc_level)
2. MultiHop-RAG (multihop_rag_doc_level)
3. FinanceBench (financebench_doc_level - Page-Level Document Units)
4. BRIGHT (bright_economics_doc_level, bright_stackoverflow_doc_level, bright_leetcode_doc_level, etc.)

Each standardized directory contains:
- documents/<doc_id>.json
- final_benchmark.json
- final_benchmark_capped.json (where applicable)
- final_benchmark_full.json
"""

import os
import sys
import json
import random
import re
import glob
import argparse
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BENCHMARKS_DIR = os.path.join(BASE_DIR, "data", "benchmarks")


# ============================================================================
# 1. BEIR Subsets Converter
# ============================================================================

def convert_beir(dataset_name: str, max_capped_queries: int = 500):
    random.seed(42)
    print(f"\n=== Converting BEIR: {dataset_name} (Document-Level) ===")
    
    src_dir = os.path.join(RAW_DIR, "beir", dataset_name)
    if not os.path.exists(src_dir):
        print(f"[ERROR] Source directory not found at {src_dir}")
        return False
        
    # Determine primary corpus and queries file paths
    corpus_file = os.path.join(src_dir, "corpus.jsonl")
    if not os.path.exists(corpus_file):
        corpus_file = os.path.join(src_dir, "corpus_corpus.jsonl")
        
    queries_file = os.path.join(src_dir, "queries.jsonl")
    if not os.path.exists(queries_file):
        queries_file = os.path.join(src_dir, "queries_queries.jsonl")
        
    qrels_file = os.path.join(src_dir, "qrels", "test.tsv")
    
    if not os.path.exists(corpus_file) or not os.path.exists(queries_file) or not os.path.exists(qrels_file):
        print(f"[ERROR] Missing required BEIR files in {src_dir}")
        print(f"  corpus: {os.path.exists(corpus_file)}, queries: {os.path.exists(queries_file)}, qrels: {os.path.exists(qrels_file)}")
        return False

    clean_name = dataset_name.replace("-", "_")
    out_dir = os.path.join(BENCHMARKS_DIR, f"beir_{clean_name}_doc_level")
    docs_dir = os.path.join(out_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    # 1. Load Corpus
    print(f"  [1/3] Ingesting documents from {os.path.basename(corpus_file)}...")
    doc_map = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            did = str(rec.get("_id", ""))
            title = rec.get("title", "")
            text = rec.get("text", "")
            full_text = f"{title}\n\n{text}".strip() if title and title != text else text
            
            if did and full_text:
                doc_record = {
                    "doc_id": did,
                    "title": title,
                    "text": full_text,
                    "source_type": f"beir_{dataset_name}",
                    "doc_length": len(full_text.split())
                }
                doc_map[did] = doc_record
                safe_fname = re.sub(r'[^\w\-_\.]', '_', did) + ".json"
                with open(os.path.join(docs_dir, safe_fname), "w", encoding="utf-8") as df:
                    json.dump(doc_record, df)

    print(f"  -> Saved {len(doc_map)} document JSONs to {docs_dir}")

    # 2. Load Queries
    print(f"  [2/3] Ingesting queries from {os.path.basename(queries_file)}...")
    query_map = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec.get("_id", ""))
            qtext = rec.get("text", "")
            if qid and qtext:
                query_map[qid] = qtext
    print(f"  -> Loaded {len(query_map)} query texts.")

    # 3. Load Qrels & Map Relevance
    print(f"  [3/3] Parsing test qrels from {os.path.basename(qrels_file)}...")
    q_to_gold_docs: Dict[str, List[str]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = str(parts[0]), str(parts[1]), float(parts[2])
                if score >= 1.0 and did in doc_map:
                    q_to_gold_docs.setdefault(qid, []).append(did)

    # Format Benchmark Records
    formatted_queries = []
    for qid, gold_dids in q_to_gold_docs.items():
        if qid in query_map and len(gold_dids) > 0:
            question_text = query_map[qid]
            first_did = gold_dids[0]
            formatted_queries.append({
                "query_id": f"q_{dataset_name}_{qid}",
                "query_group": f"BEIR {dataset_name.upper()}",
                "query_type": "Retrieval",
                "question": question_text,
                "raw_question": question_text,
                "golden_answer": "",
                "doc_id_source": first_did,
                "expected_doc_ids": sorted(list(set(gold_dids))),
                "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in sorted(list(set(gold_dids)))]
            })

    # Sort queries deterministically by ID
    formatted_queries.sort(key=lambda x: x["query_id"])
    
    # Save full benchmark
    with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)
    with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)

    # Save capped benchmark for large query sets (e.g. FiQA)
    if len(formatted_queries) > max_capped_queries:
        capped_queries = random.sample(formatted_queries, max_capped_queries)
        capped_queries.sort(key=lambda x: x["query_id"])
        with open(os.path.join(out_dir, "final_benchmark_capped.json"), "w", encoding="utf-8") as f:
            json.dump(capped_queries, f, indent=2)
        print(f"  -> Generated capped benchmark with {len(capped_queries)} queries.")

    print(f"  [OK] Successfully converted BEIR {dataset_name}: {len(doc_map)} docs, {len(formatted_queries)} queries.")
    return True


# ============================================================================
# 2. MultiHop-RAG Converter
# ============================================================================

def convert_multihop_rag(max_capped_queries: int = 250):
    random.seed(42)
    print("\n=== Converting MultiHop-RAG (Document-Level) ===")
    
    src_dir = os.path.join(RAW_DIR, "multihop_rag")
    corpus_file = os.path.join(src_dir, "corpus.json")
    queries_file = os.path.join(src_dir, "MultiHopRAG.json")
    
    if not os.path.exists(corpus_file) or not os.path.exists(queries_file):
        print(f"[ERROR] MultiHop-RAG files not found at {src_dir}")
        return False

    out_dir = os.path.join(BENCHMARKS_DIR, "multihop_rag_doc_level")
    docs_dir = os.path.join(out_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    # 1. Ingest Corpus Articles
    print(f"  [1/2] Ingesting 609 news articles from {os.path.basename(corpus_file)}...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    doc_map_by_key: Dict[str, str] = {}  # title/url/norm_title -> doc_id
    doc_map: Dict[str, Dict[str, Any]] = {}

    for idx, art in enumerate(articles):
        did = f"mhop_doc_{idx:04d}"
        title = art.get("title", "")
        url = art.get("url", "")
        body = art.get("body", "")
        author = art.get("author") or ""
        source = art.get("source") or "News"
        published_at = art.get("published_at") or ""
        category = art.get("category") or "general"

        full_text = f"{title}\n\n{body}".strip() if title and title != body else body

        doc_record = {
            "doc_id": did,
            "title": title,
            "text": full_text,
            "author": author,
            "source": source,
            "published_at": published_at,
            "category": category,
            "url": url,
            "source_type": "multihop_news_article",
            "doc_length": len(full_text.split())
        }
        doc_map[did] = doc_record

        # Map lookup keys
        if title:
            doc_map_by_key[title.strip().lower()] = did
        if url:
            doc_map_by_key[url.strip().lower()] = did

        with open(os.path.join(docs_dir, f"{did}.json"), "w", encoding="utf-8") as df:
            json.dump(doc_record, df, indent=2)

    print(f"  -> Saved {len(doc_map)} document JSONs to {docs_dir}")

    # 2. Ingest MultiHop Queries & Map Evidence Chains
    print(f"  [2/2] Ingesting multi-hop queries from {os.path.basename(queries_file)}...")
    with open(queries_file, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    formatted_queries = []
    categorized_queries: Dict[str, List[Dict[str, Any]]] = {}

    for idx, qrec in enumerate(raw_queries):
        query_text = qrec.get("query", "")
        answer = qrec.get("answer", "")
        qtype = qrec.get("question_type", "inference_query")
        evidence_list = qrec.get("evidence_list", [])

        gold_dids = set()
        for ev in evidence_list:
            ev_title = (ev.get("title") or "").strip().lower()
            ev_url = (ev.get("url") or "").strip().lower()
            
            matched_did = doc_map_by_key.get(ev_title) or doc_map_by_key.get(ev_url)
            if matched_did:
                gold_dids.add(matched_did)
            else:
                # Substring match fallback
                for k, did in doc_map_by_key.items():
                    if ev_title and ev_title in k:
                        gold_dids.add(did)
                        break

        # Discard empty null queries or queries with missing docs
        if not gold_dids and qtype != "null_query":
            continue

        sorted_gold_dids = sorted(list(gold_dids))
        first_did = sorted_gold_dids[0] if sorted_gold_dids else "mhop_doc_0000"

        q_item = {
            "query_id": f"q_mhop_{idx:05d}",
            "query_group": "MultiHop-RAG News",
            "query_type": qtype,
            "question": query_text,
            "raw_question": query_text,
            "golden_answer": str(answer),
            "doc_id_source": first_did,
            "expected_doc_ids": sorted_gold_dids,
            "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in sorted_gold_dids],
            "hop_count": len(sorted_gold_dids)
        }
        formatted_queries.append(q_item)
        categorized_queries.setdefault(qtype, []).append(q_item)

    # Save full benchmark
    with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)
    with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)

    # Save balanced capped benchmark across question types
    capped_queries = []
    cap_per_type = max_capped_queries // max(1, len(categorized_queries))
    for qtype, items in categorized_queries.items():
        sample_k = min(len(items), cap_per_type)
        capped_queries.extend(random.sample(items, sample_k))
    capped_queries.sort(key=lambda x: x["query_id"])

    with open(os.path.join(out_dir, "final_benchmark_capped.json"), "w", encoding="utf-8") as f:
        json.dump(capped_queries, f, indent=2)

    print(f"  [OK] Successfully converted MultiHop-RAG: {len(doc_map)} docs, {len(formatted_queries)} queries (Capped: {len(capped_queries)}).")
    return True


# ============================================================================
# 3. FinanceBench Converter (Page-Level Document Units)
# ============================================================================

# Distractor source: full SEC 10-K filings already on disk (Markdown). Used to
# give FinanceBench a non-gold distractor pool (see _build_financebench_distractors).
FINANCEBENCH_DISTRACTOR_DIR = os.path.join(RAW_DIR, "10-K_2026", "10-K_2026")
FINANCEBENCH_DISTRACTOR_CHUNK_WORDS = 350  # approx. FinanceBench gold-page length


# FinanceBench company names that never appear in their SEC registrant name
# (FinanceBench "company" -> SEC registrant spelling). For these, match against
# the registrant spelling instead of the short abbreviation, which would also
# match unrelated names (e.g. "amd" is a substring of "Camden Property Trust").
_FINANCEBENCH_NAME_ALIASES = {
    "AMD": "advanced micro devices",
}


def _norm_company(s: str) -> str:
    """Lowercase and keep only alphanumerics for robust name matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


_FILING_BOILERPLATE = re.compile(
    r"SECURITIES\s+AND\s+EXCHANGE|FORM\s+10-?K|REGISTRANT|COMMISSION\s+FILE"
    r"|TRANSITION\s+REPORT|ANNUAL\s+REPORT|STATE\s+OF\s+INCORP"
    r"|I\.?R\.?S|EMPLOYER\s+IDENT|TELEPHONE",
    re.IGNORECASE,
)


def _strip_filing_markup(text: str) -> str:
    """Remove HTML tags and non-breaking spaces from a filing's Markdown text.

    The tag regex only matches genuine '<tag ...>' / '</tag>' constructs, so it
    leaves legitimate '<' / '>' comparison operators (e.g. "revenue < $1B")
    untouched.
    """
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    return text.replace("\xa0", " ")


def _extract_registrant_name(md_text: str) -> str:
    """Best-effort SEC registrant name extraction from a 10-K Markdown header.

    Two header layouts are common across the filings on disk:
      * the name immediately precedes '(Exact name of registrant ...)'; and
      * the name follows 'Commission file number <digits>' on the next heading
        (some filings omit the 'Exact name of registrant' marker entirely).
    The function tries the first layout, then falls back to the second, and
    returns '' when the match looks like boilerplate rather than a real name.
    """
    text = _strip_filing_markup(md_text)

    # Layout 1: name precedes the 'Exact name of registrant' marker.
    idx = text.lower().find("exact name of registrant")
    if idx != -1:
        before = text[max(0, idx - 700):idx]
        m = re.search(r"commission\s+file\s+(?:number|no\.?)\s*:?\s*[\d\-]+", before, re.IGNORECASE)
        if m:
            before = before[m.end():]
        before = re.sub(r"#{1,6}\s*", "", before)
        lines = [ln.strip(" \t\n\r|*_–—()-") for ln in before.splitlines()]
        lines = [ln for ln in lines if ln]
        if lines and not _FILING_BOILERPLATE.search(lines[-1]):
            return lines[-1]

    # Layout 2: name follows 'Commission file number <digits>' (no marker).
    m = re.search(r"commission\s+file\s+(?:number|no\.?)\s*:?", text, re.IGNORECASE)
    if not m:
        return ""
    after = re.sub(r"#{1,6}\s*", "", text[m.end():m.end() + 600])
    lines = [ln.strip(" \t\n\r|*_–—()-") for ln in after.splitlines()]
    lines = [ln for ln in lines if ln]
    while lines and re.fullmatch(r"[\d\-]+", lines[0]):
        lines.pop(0)
    if lines and not _FILING_BOILERPLATE.search(lines[0]):
        return lines[0]
    return ""


def _build_financebench_distractors(out_dir, docs_dir, financebench_companies, max_distractors=2000):
    """Add non-gold SEC filing pages so FinanceBench is not a 100%-gold corpus.

    convert_financebench() emits only the evidence (gold) pages, which leaves
    every document in the corpus as the answer to *some* query (zero true
    negatives). This helper builds a distractor pool from the full 10-K filings
    on disk, excluding filings whose registrant matches a FinanceBench company
    to avoid same-company answer leakage.

    Returns the number of distractor documents written (0 if disabled/absent).
    """
    if max_distractors <= 0:
        return 0
    if not os.path.isdir(FINANCEBENCH_DISTRACTOR_DIR):
        print(f"  [WARN] Distractor source not found at {FINANCEBENCH_DISTRACTOR_DIR}; "
              f"FinanceBench will contain only gold pages (0 true negatives).")
        return 0

    company_norms = set()
    for c in financebench_companies:
        if not c:
            continue
        alias = _FINANCEBENCH_NAME_ALIASES.get(c)
        # When the FinanceBench name is an abbreviation that never appears in
        # the SEC registrant name, match the registrant spelling instead.
        company_norms.add(_norm_company(alias if alias else c))

    filing_paths = sorted(glob.glob(os.path.join(FINANCEBENCH_DISTRACTOR_DIR, "*", "10-K.md")))
    distractor_count = 0
    chunks_per_filing = 50  # spread distractors across many filings for diversity

    for md_path in filing_paths:
        if distractor_count >= max_distractors:
            break
        try:
            with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        text = _strip_filing_markup(text)
        name = _extract_registrant_name(text[:8000])
        name_norm = _norm_company(name)
        if name_norm and any(cn in name_norm for cn in company_norms):
            continue  # same registrant as a FinanceBench gold filing -> skip

        # Directory names are '<date>_<accession>'; the accession is
        # '<CIK>-<year>-<seq>', so the CIK is the leading 10-digit part.
        accession = os.path.basename(os.path.dirname(md_path)).split("_")[1]
        cik = accession.split("-")[0]
        words = text.split()
        n_chunks = max(1, len(words) // FINANCEBENCH_DISTRACTOR_CHUNK_WORDS)
        take = min(chunks_per_filing, n_chunks, max_distractors - distractor_count)
        if take <= 0:
            continue

        # Evenly sample chunks across the filing to cover diverse sections
        # (business, MD&A, financial statements) rather than just the cover page.
        if take > 1:
            idxs = [int(round(i * (n_chunks - 1) / (take - 1))) for i in range(take)]
        else:
            idxs = [0]

        for ci in idxs:
            start = ci * FINANCEBENCH_DISTRACTOR_CHUNK_WORDS
            chunk = " ".join(words[start:start + FINANCEBENCH_DISTRACTOR_CHUNK_WORDS])
            if len(chunk.split()) < 30:
                continue
            did = f"{cik}_10K_distractor_{distractor_count:05d}"
            doc_record = {
                "doc_id": did,
                "title": f"SEC 10-K ({name or cik}) - distractor chunk",
                "text": chunk,
                "company": name or cik,
                "cik": cik,
                "source_type": "sec_filing_distractor",
                "is_distractor": True,
                "doc_length": len(chunk.split()),
            }
            with open(os.path.join(docs_dir, f"{did}.json"), "w", encoding="utf-8") as df:
                json.dump(doc_record, df, indent=2)
            distractor_count += 1

    print(f"  -> Added {distractor_count} distractor documents to {docs_dir}")
    return distractor_count


def convert_financebench(max_distractors: int = 2000):
    random.seed(42)
    print("\n=== Converting FinanceBench (Document-Level: Page Units) ===")
    
    src_file = os.path.join(RAW_DIR, "financebench", "financebench_train.jsonl")
    if not os.path.exists(src_file):
        print(f"[ERROR] FinanceBench file not found at {src_file}")
        return False

    out_dir = os.path.join(BENCHMARKS_DIR, "financebench_doc_level")
    docs_dir = os.path.join(out_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    records = []
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} FinanceBench QA records.")

    doc_map: Dict[str, Dict[str, Any]] = {}
    formatted_queries = []
    financebench_companies: Set[str] = set()

    for idx, rec in enumerate(records):
        doc_name = rec.get("doc_name", f"sec_doc_{idx}")
        company = rec.get("company", "")
        financebench_companies.add(company)
        doc_period = rec.get("doc_period", "")
        doc_type = rec.get("doc_type", "10k")
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        justification = rec.get("justification", "")
        qtype = rec.get("question_type", "Financial QA")
        qreason = rec.get("question_reasoning", "")
        qid = str(rec.get("financebench_id", f"fb_{idx:04d}"))

        evidence_list = rec.get("evidence", [])
        gold_dids = []

        for ev in evidence_list:
            page_num = ev.get("evidence_page_num", 0)
            full_page_text = ev.get("evidence_text_full_page", ev.get("evidence_text", "")).strip()
            
            if not full_page_text:
                continue

            page_did = f"{doc_name}_page_{page_num:03d}"
            gold_dids.append(page_did)

            if page_did not in doc_map:
                page_title = f"{company} {doc_type.upper()} ({doc_period}) - Page {page_num}"
                doc_record = {
                    "doc_id": page_did,
                    "title": page_title,
                    "text": full_page_text,
                    "company": company,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "source_type": "sec_filing_page",
                    "doc_length": len(full_page_text.split())
                }
                doc_map[page_did] = doc_record
                with open(os.path.join(docs_dir, f"{page_did}.json"), "w", encoding="utf-8") as df:
                    json.dump(doc_record, df, indent=2)

        if gold_dids:
            sorted_gold_dids = sorted(list(set(gold_dids)))
            formatted_queries.append({
                "query_id": qid,
                "query_group": "FinanceBench SEC Filings",
                "query_type": qtype,
                "question_reasoning": qreason,
                "question": question,
                "raw_question": question,
                "golden_answer": answer,
                "justification": justification,
                "doc_id_source": sorted_gold_dids[0],
                "expected_doc_ids": sorted_gold_dids,
                "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in sorted_gold_dids]
            })

    # Add non-gold SEC distractor pages so the corpus is not 100 % gold
    # (otherwise there are zero true negatives for retrieval evaluation).
    n_distractors = _build_financebench_distractors(
        out_dir, docs_dir, financebench_companies, max_distractors=max_distractors
    )

    # Save benchmark JSONs
    with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)
    with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
        json.dump(formatted_queries, f, indent=2)

    print(f"  [OK] Successfully converted FinanceBench: {len(doc_map)} gold pages + {n_distractors} distractors, {len(formatted_queries)} queries.")
    return True


# ============================================================================
# 4. BRIGHT Reasoning Subsets Converter
# ============================================================================

def sanitize_doc_id(raw_id: str) -> str:
    return str(raw_id).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def convert_bright(subdomains: List[str] = None):
    random.seed(42)
    if subdomains is None:
        subdomains = ["economics", "stackoverflow", "leetcode", "robotics", "biology"]

    print(f"\n=== Converting BRIGHT Subdomains: {subdomains} (Document-Level) ===")
    src_dir = os.path.join(RAW_DIR, "bright")
    if not os.path.exists(src_dir):
        print(f"[ERROR] BRIGHT directory not found at {src_dir}")
        return False

    success_domains = 0
    for domain in subdomains:
        print(f"\n  --- Converting BRIGHT Domain: {domain} ---")
        doc_parquet = os.path.join(src_dir, "documents", f"{domain}-00000-of-00001.parquet")
        example_parquet = os.path.join(src_dir, "examples", f"{domain}-00000-of-00001.parquet")

        if not os.path.exists(doc_parquet) or not os.path.exists(example_parquet):
            print(f"  [WARN] Missing files for domain '{domain}', skipping.")
            continue

        out_dir = os.path.join(BENCHMARKS_DIR, f"bright_{domain}_doc_level")
        docs_dir = os.path.join(out_dir, "documents")
        os.makedirs(docs_dir, exist_ok=True)

        # 1. Ingest Documents
        doc_df = pd.read_parquet(doc_parquet)
        doc_records = doc_df.to_dict(orient="records")
        doc_map = {}

        for d in doc_records:
            raw_did = str(d.get("id", ""))
            safe_did = sanitize_doc_id(raw_did)
            content = str(d.get("content", d.get("text", ""))).strip()
            if safe_did and content:
                doc_record = {
                    "doc_id": safe_did,
                    "raw_doc_id": raw_did,
                    "title": f"BRIGHT {domain.capitalize()} #{safe_did}",
                    "text": content,
                    "source_type": f"bright_{domain}",
                    "doc_length": len(content.split())
                }
                doc_map[safe_did] = doc_record
                with open(os.path.join(docs_dir, f"{safe_did}.json"), "w", encoding="utf-8") as df:
                    json.dump(doc_record, df, indent=2)

        print(f"  -> Saved {len(doc_map)} document JSONs to {docs_dir}")

        # 2. Ingest Queries & Qrels
        ex_df = pd.read_parquet(example_parquet)
        ex_records = ex_df.to_dict(orient="records")
        formatted_queries = []

        for q in ex_records:
            qid = str(q.get("id", ""))
            query_text = q.get("query", "")
            gold_ids = q.get("gold_ids", [])
            if isinstance(gold_ids, np.ndarray) or hasattr(gold_ids, "__iter__"):
                gold_ids = [str(g) for g in gold_ids]
            elif isinstance(gold_ids, str):
                gold_ids = [gold_ids]
            else:
                gold_ids = []

            safe_gold_ids = [sanitize_doc_id(g) for g in gold_ids]
            valid_gold_ids = [did for did in safe_gold_ids if did in doc_map]

            if valid_gold_ids and query_text:
                formatted_queries.append({
                    "query_id": f"q_bright_{domain}_{qid}",
                    "query_group": f"BRIGHT {domain.capitalize()}",
                    "query_type": "Reasoning Retrieval",
                    "question": query_text,
                    "raw_question": query_text,
                    "golden_answer": str(q.get("gold_answer", "")),
                    "doc_id_source": valid_gold_ids[0],
                    "expected_doc_ids": sorted(valid_gold_ids),
                    "ground_truth_child_chunks": [{"chunk_id": did, "text": ""} for did in sorted(valid_gold_ids)]
                })

        with open(os.path.join(out_dir, "final_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(formatted_queries, f, indent=2)
        with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
            json.dump(formatted_queries, f, indent=2)

        print(f"  [OK] Saved BRIGHT {domain}: {len(doc_map)} docs, {len(formatted_queries)} queries.")
        success_domains += 1

    return success_domains > 0


# ============================================================================
# 5. Global Validation & Integrity Audit
# ============================================================================

def validate_all_benchmarks():
    print("\n" + "=" * 60)
    print("=== GLOBAL DOCUMENT-LEVEL BENCHMARK INTEGRITY AUDIT ===")
    print("=" * 60)

    target_dirs = sorted([d for d in os.listdir(BENCHMARKS_DIR) if d.endswith("_doc_level")])
    
    audit_results = []
    for b_dir in target_dirs:
        full_b_path = os.path.join(BENCHMARKS_DIR, b_dir)
        docs_dir = os.path.join(full_b_path, "documents")
        q_file = os.path.join(full_b_path, "final_benchmark.json")

        if not os.path.exists(docs_dir) or not os.path.exists(q_file):
            audit_results.append((b_dir, 0, 0, 0, "MISSING_FILES"))
            continue

        doc_files = set(f.replace(".json", "") for f in os.listdir(docs_dir) if f.endswith(".json"))
        with open(q_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        missing_gold_count = 0
        total_gold_links = 0

        for q in queries:
            expected = q.get("expected_doc_ids", [])
            total_gold_links += len(expected)
            for did in expected:
                if did not in doc_files:
                    missing_gold_count += 1

        status = "PASSED (100% Gold Match)" if missing_gold_count == 0 and len(doc_files) > 0 and len(queries) > 0 else f"FAILED ({missing_gold_count} missing)"
        audit_results.append((b_dir, len(doc_files), len(queries), total_gold_links, status))

    print(f"{'Benchmark Name':<32} | {'Documents':<9} | {'Queries':<7} | {'Gold Links':<10} | {'Status'}")
    print("-" * 80)
    for b_name, n_docs, n_q, n_gold, status in audit_results:
        print(f"{b_name:<32} | {n_docs:<9} | {n_q:<7} | {n_gold:<10} | {status}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert and standardize raw retriever benchmarks into document-level format.")
    BEIR_ALL = [
        "scifact", "nfcorpus", "fiqa", "arguana", "climate-fever",
        "dbpedia-entity", "fever", "hotpotqa", "nq", "quora",
        "scidocs", "trec-covid", "webis-touche2020"
    ]

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "fiqa", "multihop_rag", "financebench", "bright"],
        help="Datasets to convert: scifact, nfcorpus, fiqa, arguana, climate-fever, dbpedia-entity, fever, hotpotqa, nq, quora, scidocs, trec-covid, webis-touche2020, multihop_rag, financebench, bright, beir_all, all",
    )
    args = parser.parse_args()
    
    requested = set(args.datasets)
    if "all" in requested:
        requested = set(BEIR_ALL) | {"multihop_rag", "financebench", "bright"}
    elif "beir_all" in requested:
        requested = requested | set(BEIR_ALL)

    # BEIR datasets
    for beir_name in BEIR_ALL:
        if beir_name in requested or beir_name.replace("-", "_") in requested:
            convert_beir(beir_name)

    # MultiHop-RAG
    if "multihop_rag" in requested:
        convert_multihop_rag()

    # FinanceBench
    if "financebench" in requested:
        convert_financebench()

    # BRIGHT
    if "bright" in requested:
        convert_bright(["economics", "stackoverflow", "leetcode", "robotics", "biology"])

    # Global Validation
    validate_all_benchmarks()


if __name__ == "__main__":
    main()
