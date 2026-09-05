"""
src/evaluation/benchmark_loader.py

Unified, Standardized Benchmark Loader for Edge-RAG.
Streams directly from raw academic corpora without lossy conversion artifacts:
1. BEIR (SciFact, NFCorpus, FiQA) - standard text: f"{title} {text}".strip() if title else text, graded qrels.
2. BRIGHT (Economics, StackOverflow, Robotics) - raw parquet documents and reasoning queries.
3. MultiHop-RAG - full 2,556 un-capped news retrieval queries.
4. FinanceBench - official SEC filing pages (no synthetic distractors).
5. EnterpriseRAG - canonical 50,000 document collection and 500 test queries.
6. LiveRAG - 895 streaming multi-session evaluation queries.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional, Set, Generator
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BENCHMARKS_DIR = os.path.join(BASE_DIR, "data", "benchmarks")


def sanitize_bright_doc_id(raw_id: str) -> str:
    """Sanitize doc IDs with forward/backward slashes, colons, or spaces."""
    return str(raw_id).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


class BenchmarkLoader:
    """
    Standardized loader providing uniform interface across all benchmarks:
    Returns:
        corpus_texts: List[str]
        corpus_docs: List[Dict[str, str]] (keys: 'doc_id', 'text')
        queries: List[Dict[str, Any]] (keys: 'query_id', 'question', 'gold_doc_ids', 'qrels')
        stats: Dict[str, Any]
    """

    @classmethod
    def _find_beir_dir(cls, subset: str) -> str:
        candidates = [
            os.path.join(RAW_DIR, "beir", subset),
            os.path.join(RAW_DIR, "beir", subset.replace("_", "-")),
            os.path.join(RAW_DIR, "beir", subset.replace("-", "_")),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"BEIR subset '{subset}' not found in {os.path.join(RAW_DIR, 'beir')}")

    @classmethod
    def load(cls, dataset_name: str) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
        
        if norm.startswith("beir_"):
            subset = norm[5:]
            return cls._load_beir(subset)
        elif norm in ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora", "hotpotqa", "fever", "nq", "climate_fever", "dbpedia_entity", "trec_covid", "webis_touche2020"):
            return cls._load_beir(norm)
        elif norm in ("bright_economics", "economics"):
            return cls._load_bright("economics")
        elif norm in ("bright_stackoverflow", "stackoverflow"):
            return cls._load_bright("stackoverflow")
        elif norm in ("bright_robotics", "robotics"):
            return cls._load_bright("robotics")
        elif norm in ("multihop_rag", "multihop"):
            return cls._load_multihop_rag()
        elif norm in ("financebench", "finance_bench"):
            return cls._load_financebench()
        elif norm in ("enterpriserag", "enterprise_rag"):
            return cls._load_enterpriserag()
        elif norm in ("liverag", "live_rag"):
            return cls._load_liverag()
        else:
            # Fallback: try loading as BEIR
            try:
                return cls._load_beir(norm)
            except FileNotFoundError:
                raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

    @classmethod
    def stream_corpus(cls, dataset_name: str) -> Generator[Tuple[str, str], None, None]:
        """
        Streams (doc_id, text) one-by-one from the raw source file.
        Does NOT materialize the full corpus in RAM.
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")

        if norm.startswith("beir_") or norm in (
            "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora",
            "hotpotqa", "fever", "nq", "climate_fever", "dbpedia_entity",
            "trec_covid", "webis_touche2020"
        ):
            subset = norm[5:] if norm.startswith("beir_") else norm
            src_dir = cls._find_beir_dir(subset)
            corpus_file = os.path.join(src_dir, "corpus.jsonl")
            if not os.path.exists(corpus_file):
                corpus_file = os.path.join(src_dir, "corpus_corpus.jsonl")

            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    did = str(rec.get("_id", ""))
                    title = (rec.get("title") or "").strip()
                    text = (rec.get("text") or "").strip()
                    full_text = f"{title} {text}".strip() if title else text
                    if did and full_text:
                        yield (did, full_text)
        else:
            # For in-memory supported datasets, load and yield
            corpus_texts, corpus_docs, _, _ = cls.load(dataset_name)
            for d in corpus_docs:
                yield (d["doc_id"], d["text"])

    @classmethod
    def load_queries(cls, dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads queries and qrels without materializing the corpus in RAM.
        """
        norm = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
        if norm.startswith("beir_") or norm in (
            "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "quora",
            "hotpotqa", "fever", "nq", "climate_fever", "dbpedia_entity",
            "trec_covid", "webis_touche2020"
        ):
            subset = norm[5:] if norm.startswith("beir_") else norm
            src_dir = cls._find_beir_dir(subset)

            queries_file = os.path.join(src_dir, "queries.jsonl")
            if not os.path.exists(queries_file):
                queries_file = os.path.join(src_dir, "queries_queries.jsonl")
            qrels_file = os.path.join(src_dir, "qrels", "test.tsv")

            queries_map = {}
            with open(queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    qid = str(rec.get("_id", ""))
                    qtext = (rec.get("text") or "").strip()
                    if qid and qtext:
                        queries_map[qid] = qtext

            qrels_by_q: Dict[str, Dict[str, float]] = {}
            total_qrel_entries = 0
            with open(qrels_file, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, score_val = str(parts[0]), str(parts[1]), float(parts[2])
                        total_qrel_entries += 1
                        qrels_by_q.setdefault(qid, {})[did] = score_val

            formatted_queries = []
            for qid in sorted(queries_map.keys()):
                if qid in qrels_by_q:
                    qrels = qrels_by_q[qid]
                    gold_dids = [did for did, s in qrels.items() if s > 0]
                    if gold_dids:
                        formatted_queries.append({
                            "query_id": f"q_beir_{subset}_{qid}",
                            "question": queries_map[qid],
                            "gold_doc_ids": sorted(gold_dids),
                            "qrels": qrels
                        })

            stats = {
                "dataset": f"beir_{subset}",
                "queries_loaded": len(formatted_queries),
                "total_qrel_entries": total_qrel_entries,
            }
            return formatted_queries, stats
        else:
            _, _, queries, stats = cls.load(dataset_name)
            return queries, stats

    @classmethod
    def _load_beir(cls, subset: str) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        src_dir = cls._find_beir_dir(subset)

        corpus_file = os.path.join(src_dir, "corpus.jsonl")
        if not os.path.exists(corpus_file):
            corpus_file = os.path.join(src_dir, "corpus_corpus.jsonl")

        queries_file = os.path.join(src_dir, "queries.jsonl")
        if not os.path.exists(queries_file):
            queries_file = os.path.join(src_dir, "queries_queries.jsonl")

        qrels_file = os.path.join(src_dir, "qrels", "test.tsv")

        # 1. Corpus
        corpus_texts = []
        corpus_docs = []
        doc_ids_set = set()

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                did = str(rec.get("_id", ""))
                title = (rec.get("title") or "").strip()
                text = (rec.get("text") or "").strip()
                
                # Official BEIR convention: f"{title} {text}".strip() if title else text
                full_text = f"{title} {text}".strip() if title else text
                if did and full_text:
                    corpus_texts.append(full_text)
                    corpus_docs.append({"doc_id": did, "text": full_text})
                    doc_ids_set.add(did)

        # 2. Queries text
        queries_map = {}
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = str(rec.get("_id", ""))
                qtext = (rec.get("text") or "").strip()
                if qid and qtext:
                    queries_map[qid] = qtext

        # 3. Qrels
        qrels_by_q: Dict[str, Dict[str, float]] = {}
        missing_qrel_docs_count = 0
        total_qrel_entries = 0

        with open(qrels_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score_val = str(parts[0]), str(parts[1]), float(parts[2])
                    total_qrel_entries += 1
                    if did in doc_ids_set:
                        qrels_by_q.setdefault(qid, {})[did] = score_val
                    else:
                        missing_qrel_docs_count += 1

        formatted_queries = []
        queries_with_gold = 0

        # Deterministic sorting by query id
        for qid in sorted(queries_map.keys()):
            if qid in qrels_by_q:
                qrels = qrels_by_q[qid]
                gold_dids = [did for did, s in qrels.items() if s > 0]
                if gold_dids:
                    queries_with_gold += 1
                    formatted_queries.append({
                        "query_id": f"q_beir_{subset}_{qid}",
                        "question": queries_map[qid],
                        "gold_doc_ids": sorted(gold_dids),
                        "qrels": qrels
                    })

        stats = {
            "dataset": f"beir_{subset}",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": queries_with_gold,
            "total_qrel_entries": total_qrel_entries,
            "missing_qrel_docs_count": missing_qrel_docs_count
        }
        return corpus_texts, corpus_docs, formatted_queries, stats

    @staticmethod
    def _load_bright(domain: str) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        src_dir = os.path.join(RAW_DIR, "bright")
        doc_parquet = os.path.join(src_dir, "documents", f"{domain}-00000-of-00001.parquet")
        example_parquet = os.path.join(src_dir, "examples", f"{domain}-00000-of-00001.parquet")

        if not os.path.exists(doc_parquet) or not os.path.exists(example_parquet):
            raise FileNotFoundError(f"BRIGHT domain {domain} parquets not found in {src_dir}")

        # 1. Documents
        doc_df = pd.read_parquet(doc_parquet)
        corpus_texts = []
        corpus_docs = []
        doc_ids_set = set()

        for d in doc_df.to_dict(orient="records"):
            raw_id = str(d.get("id", ""))
            safe_id = sanitize_bright_doc_id(raw_id)
            content = str(d.get("content", d.get("text", ""))).strip()
            if safe_id and content:
                corpus_texts.append(content)
                corpus_docs.append({"doc_id": safe_id, "text": content})
                doc_ids_set.add(safe_id)

        # 2. Examples & Queries
        ex_df = pd.read_parquet(example_parquet)
        formatted_queries = []
        missing_gold_docs = 0

        for q in ex_df.to_dict(orient="records"):
            qid = str(q.get("id", ""))
            qtext = str(q.get("query", "")).strip()
            raw_gold = q.get("gold_ids", [])
            if isinstance(raw_gold, (list, np.ndarray)) or hasattr(raw_gold, "__iter__"):
                gold_list = [str(g) for g in raw_gold]
            elif isinstance(raw_gold, str):
                gold_list = [raw_gold]
            else:
                gold_list = []

            safe_gold = [sanitize_bright_doc_id(g) for g in gold_list]
            valid_gold = [did for did in safe_gold if did in doc_ids_set]
            missing_gold_docs += (len(safe_gold) - len(valid_gold))

            if valid_gold and qtext:
                qrels = {did: 1.0 for did in valid_gold}
                formatted_queries.append({
                    "query_id": f"q_bright_{domain}_{qid}",
                    "question": qtext,
                    "gold_doc_ids": sorted(valid_gold),
                    "qrels": qrels
                })

        formatted_queries.sort(key=lambda x: x["query_id"])
        stats = {
            "dataset": f"bright_{domain}",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": len(formatted_queries),
            "missing_qrel_docs_count": missing_gold_docs
        }
        return corpus_texts, corpus_docs, formatted_queries, stats

    @staticmethod
    def _load_multihop_rag() -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        src_dir = os.path.join(RAW_DIR, "multihop_rag")
        corpus_file = os.path.join(src_dir, "corpus.json")
        queries_file = os.path.join(src_dir, "MultiHopRAG.json")

        if not os.path.exists(corpus_file) or not os.path.exists(queries_file):
            raise FileNotFoundError(f"MultiHop-RAG files not found in {src_dir}")

        # 1. Corpus
        with open(corpus_file, "r", encoding="utf-8") as f:
            articles = json.load(f)

        corpus_texts = []
        corpus_docs = []
        doc_map_by_key: Dict[str, str] = {}

        for idx, art in enumerate(articles):
            did = f"mhop_doc_{idx:04d}"
            title = (art.get("title") or "").strip()
            body = (art.get("body") or "").strip()
            url = (art.get("url") or "").strip()
            full_text = f"{title}\n\n{body}".strip() if title and title != body else body

            corpus_texts.append(full_text)
            corpus_docs.append({"doc_id": did, "text": full_text})

            if title:
                doc_map_by_key[title.lower()] = did
            if url:
                doc_map_by_key[url.lower()] = did

        # 2. Queries
        with open(queries_file, "r", encoding="utf-8") as f:
            raw_queries = json.load(f)

        formatted_queries = []
        missing_evidence = 0

        for idx, qrec in enumerate(raw_queries):
            qtext = str(qrec.get("query", "")).strip()
            qtype = qrec.get("question_type", "inference_query")
            evidence_list = qrec.get("evidence_list", [])

            gold_dids = set()
            for ev in evidence_list:
                ev_title = (ev.get("title") or "").strip().lower()
                ev_url = (ev.get("url") or "").strip().lower()

                matched = doc_map_by_key.get(ev_title) or doc_map_by_key.get(ev_url)
                if matched:
                    gold_dids.add(matched)
                else:
                    # Substring match fallback
                    found = False
                    for k, did in doc_map_by_key.items():
                        if ev_title and ev_title in k:
                            gold_dids.add(did)
                            found = True
                            break
                    if not found:
                        missing_evidence += 1

            if gold_dids and qtext:
                qrels = {did: 1.0 for did in gold_dids}
                formatted_queries.append({
                    "query_id": f"q_mhop_{idx:05d}",
                    "question": qtext,
                    "gold_doc_ids": sorted(list(gold_dids)),
                    "qrels": qrels
                })

        stats = {
            "dataset": "multihop_rag",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": len(formatted_queries),
            "missing_qrel_docs_count": missing_evidence
        }
        return corpus_texts, corpus_docs, formatted_queries, stats

    @staticmethod
    def _load_financebench() -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads official FinanceBench evidence pages directly from financebench_train.jsonl.
        Omits synthetic distractors for academic peer comparability.
        """
        src_file = os.path.join(RAW_DIR, "financebench", "financebench_train.jsonl")
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"FinanceBench raw file not found at {src_file}")

        records = []
        with open(src_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        doc_map = {}
        formatted_queries = []

        for idx, rec in enumerate(records):
            qid = str(rec.get("financebench_id", f"fb_{idx:04d}"))
            qtext = str(rec.get("question", "")).strip()
            evidence_list = rec.get("evidence", [])

            gold_page_ids = []
            for ev in evidence_list:
                doc_name = ev.get("doc_name", "")
                page_num = ev.get("evidence_page_num", 0)
                full_page_text = ev.get("evidence_text_full_page") or ev.get("evidence_text", "")

                if not doc_name or not full_page_text:
                    continue

                page_id = f"fb_{doc_name}_p{page_num}"
                gold_page_ids.append(page_id)

                if page_id not in doc_map:
                    doc_map[page_id] = {
                        "doc_id": page_id,
                        "text": full_page_text.strip()
                    }

            if gold_page_ids and qtext:
                qrels = {did: 1.0 for did in sorted(list(set(gold_page_ids)))}
                formatted_queries.append({
                    "query_id": f"q_financebench_{qid}",
                    "question": qtext,
                    "gold_doc_ids": sorted(list(set(gold_page_ids))),
                    "qrels": qrels
                })

        formatted_queries.sort(key=lambda x: x["query_id"])
        corpus_docs = [doc_map[did] for did in sorted(doc_map.keys())]
        corpus_texts = [d["text"] for d in corpus_docs]

        stats = {
            "dataset": "financebench",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": len(formatted_queries),
            "missing_qrel_docs_count": 0
        }
        return corpus_texts, corpus_docs, formatted_queries, stats

    @staticmethod
    def _load_enterpriserag() -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads the 50,000 document EnterpriseRAG benchmark.
        Reads from data/benchmarks/enterpriserag_doc_level to ensure the reproducible seed-42 50k subset.
        """
        b_dir = os.path.join(BENCHMARKS_DIR, "enterpriserag_doc_level")
        docs_dir = os.path.join(b_dir, "documents")
        q_file = os.path.join(b_dir, "final_benchmark.json")

        if not os.path.exists(docs_dir) or not os.path.exists(q_file):
            raise FileNotFoundError(f"EnterpriseRAG doc level directory not found at {b_dir}")

        doc_files = sorted([f for f in os.listdir(docs_dir) if f.endswith(".json")])
        corpus_texts = []
        corpus_docs = []

        for f in doc_files:
            fpath = os.path.join(docs_dir, f)
            with open(fpath, "r", encoding="utf-8") as df:
                doc = json.load(df)
            did = str(doc["doc_id"])
            text = doc["text"]
            corpus_texts.append(text)
            corpus_docs.append({"doc_id": did, "text": text})

        with open(q_file, "r", encoding="utf-8") as qf:
            q_data = json.load(qf)

        formatted_queries = []
        for q in q_data:
            qid = str(q.get("query_id", ""))
            qtext = str(q.get("question", "")).strip()
            expected = [str(e) for e in q.get("expected_doc_ids", [])]
            if expected and qtext:
                qrels = {did: 1.0 for did in expected}
                formatted_queries.append({
                    "query_id": qid,
                    "question": qtext,
                    "gold_doc_ids": expected,
                    "qrels": qrels
                })

        formatted_queries.sort(key=lambda x: x["query_id"])
        stats = {
            "dataset": "enterpriserag",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": len(formatted_queries),
            "missing_qrel_docs_count": 0
        }
        return corpus_texts, corpus_docs, formatted_queries, stats

    @staticmethod
    def _load_liverag() -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads 895 LiveRAG streaming benchmark queries and unchunked supporting documents.
        """
        raw_file = os.path.join(RAW_DIR, "liverag_bench", "LiveRAG_banchmark_20250910.parquet")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"LiveRAG raw parquet not found at {raw_file}")

        df = pd.read_parquet(raw_file)
        records = df.to_dict(orient="records")

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
                except Exception:
                    return []
            if isinstance(raw_val, dict):
                return [raw_val]
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
                        except Exception:
                            pass
                return res
            return []

        doc_map = {}
        for rec in records:
            supp_docs = _parse_supp_docs(rec.get("Supporting_Documents"))
            for d in supp_docs:
                did = str(d.get("doc_id", ""))
                text = d.get("content", d.get("text", ""))
                if did and text and did not in doc_map:
                    doc_map[did] = {
                        "doc_id": did,
                        "text": str(text).strip()
                    }

        corpus_docs = [doc_map[did] for did in sorted(doc_map.keys())]
        corpus_texts = [d["text"] for d in corpus_docs]

        formatted_queries = []
        for idx, rec in enumerate(records):
            qtext = str(rec.get("Question", rec.get("question", ""))).strip()
            supp_docs = _parse_supp_docs(rec.get("Supporting_Documents"))
            gold_dids = [str(d.get("doc_id", "")) for d in supp_docs if str(d.get("doc_id", "")) in doc_map]

            if gold_dids and qtext:
                qrels = {did: 1.0 for did in sorted(list(set(gold_dids)))}
                formatted_queries.append({
                    "query_id": f"q_live_{rec.get('Index', idx)}",
                    "question": qtext,
                    "gold_doc_ids": sorted(list(set(gold_dids))),
                    "qrels": qrels
                })

        formatted_queries.sort(key=lambda x: x["query_id"])
        stats = {
            "dataset": "liverag",
            "total_docs": len(corpus_docs),
            "queries_loaded": len(formatted_queries),
            "queries_with_gold": len(formatted_queries),
            "missing_qrel_docs_count": 0
        }
        return corpus_texts, corpus_docs, formatted_queries, stats
