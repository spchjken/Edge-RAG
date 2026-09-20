import os
import sys
import glob
import json
import random
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.benchmark_creation.step1_chunking import process_document, get_tokenizer

try:
    import pyarrow.ipc as ipc
except ImportError:
    ipc = None


def read_arrow_file(file_path: str) -> List[Dict[str, Any]]:
    records = []
    if ipc is not None:
        try:
            with open(file_path, 'rb') as f:
                reader = ipc.open_stream(f)
                table = reader.read_all()
                dict_batch = table.to_pydict()
                keys = list(dict_batch.keys())
                if keys:
                    num_rows = len(dict_batch[keys[0]])
                    for r in range(num_rows):
                        records.append({k: dict_batch[k][r] for k in keys})
                    return records
        except Exception:
            pass

        try:
            import pyarrow.feather as feather
            table = feather.read_table(file_path)
            dict_batch = table.to_pydict()
            keys = list(dict_batch.keys())
            if keys:
                num_rows = len(dict_batch[keys[0]])
                for r in range(num_rows):
                    records.append({k: dict_batch[k][r] for k in keys})
                return records
        except Exception:
            pass

        try:
            reader = ipc.RecordBatchFileReader(file_path)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                dict_batch = batch.to_pydict()
                keys = list(dict_batch.keys())
                if keys:
                    num_rows = len(dict_batch[keys[0]])
                    for r in range(num_rows):
                        records.append({k: dict_batch[k][r] for k in keys})
            if records:
                return records
        except Exception:
            pass

    try:
        from datasets import Dataset
        ds = Dataset.from_file(file_path)
        for r in ds:
            records.append(dict(r))
        return records
    except Exception as e:
        print(f"Error reading Arrow file {file_path}: {e}")

    return records


def load_parquet_or_arrow(file_path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
        df = pd.read_parquet(file_path)
        return df.to_dict(orient="records")
    except Exception:
        pass

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        dict_batch = table.to_pydict()
        keys = list(dict_batch.keys())
        records = []
        if keys:
            num_rows = len(dict_batch[keys[0]])
            for r in range(num_rows):
                records.append({k: dict_batch[k][r] for k in keys})
            return records
    except Exception:
        pass

    return read_arrow_file(file_path)


def convert_enterpriserag(doc_counts=[100, 250, 500, 1000]):
    random.seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    possible_raw_dirs = [
        os.path.join(base_dir, "data", "raw", "enterpriserag_bench", "data"),
        os.path.join(base_dir, "data", "raw", "enterpriserag-bench", "onyx-dot-app___enterprise_rag-bench"),
        os.path.join(base_dir, "data", "raw", "enterpriserag_bench"),
    ]

    doc_files = []
    quest_files = []
    for rd in possible_raw_dirs:
        df = glob.glob(os.path.join(rd, "**", "documents", "*.parquet"), recursive=True) + \
             glob.glob(os.path.join(rd, "**", "documents", "*.arrow"), recursive=True)
        qf = glob.glob(os.path.join(rd, "**", "questions", "*.parquet"), recursive=True) + \
             glob.glob(os.path.join(rd, "**", "questions", "*.arrow"), recursive=True)
        if df and qf:
            doc_files, quest_files = df, qf
            break

    if not doc_files or not quest_files:
        print("[WARN] EnterpriseRAG raw parquet/arrow files not found!")
        return

    doc_records = []
    for df in doc_files:
        doc_records.extend(load_parquet_or_arrow(df))

    quest_records = []
    for qf in quest_files:
        quest_records.extend(load_parquet_or_arrow(qf))

    print(f"Loaded EnterpriseRAG: {len(doc_records)} document records and {len(quest_records)} question records.")

    # Stratified Capped Query Sampling (215 Queries Total)
    categorized_q = {}
    for q in quest_records:
        qtype = str(q.get("question_type", "Basic")).strip()
        categorized_q.setdefault(qtype, []).append(q)

    print(f"EnterpriseRAG question categories found: { {k: len(v) for k, v in categorized_q.items()} }")

    capped_q_records = []
    # Case-insensitive lookup for stratified sampling targets
    target_sampling_raw = {
        "Basic": 30,
        "Semantic": 25,
        "Intra-Document Reasoning": 20,
        "Project Related": 20,
    }
    target_sampling_lower = {k.lower().replace("-", " ").replace("_", " "): v for k, v in target_sampling_raw.items()}

    def _normalize_cat(cat_name):
        return cat_name.lower().replace("-", " ").replace("_", " ").strip()

    for cat, items in categorized_q.items():
        cap = target_sampling_lower.get(_normalize_cat(cat))
        if cap is not None:
            sample_n = min(cap, len(items))
            capped_q_records.extend(random.sample(items, sample_n))
        else:
            # Full for all other categories
            capped_q_records.extend(items)

    print(f"EnterpriseRAG Capped Query Set: {len(capped_q_records)} stratified queries (Full set: {len(quest_records)} queries).")

    doc_map_by_id = {}
    for d in doc_records:
        did = str(d.get("doc_id", ""))
        if did:
            doc_map_by_id[did] = d

    tokenizer = get_tokenizer()

    # Collect ALL referenced doc IDs from ALL queries, sorted for determinism (Bug 1 fix)
    all_referenced_doc_ids = set()
    for q in quest_records:
        expected = q.get("expected_doc_ids", [])
        # Coerce numpy arrays / other iterables to Python list
        if isinstance(expected, str):
            all_referenced_doc_ids.add(expected)
        else:
            try:
                for e in expected:
                    all_referenced_doc_ids.add(str(e))
            except TypeError:
                pass

    # Remove empty strings
    all_referenced_doc_ids.discard("")
    gold_doc_ids = sorted([did for did in all_referenced_doc_ids if did in doc_map_by_id])
    noise_pool = sorted([did for did in doc_map_by_id.keys() if did not in all_referenced_doc_ids])
    print(f"EnterpriseRAG: {len(gold_doc_ids)} gold docs, {len(noise_pool)} available noise docs.")

    for target_noise_count in doc_counts:
        tier_tag = f"corpus_stress_{target_noise_count}"
        out_dir = os.path.join(base_dir, "data", "benchmarks", "enterpriserag", tier_tag)
        out_chunks_dir = os.path.join(out_dir, "step1_chunks")
        os.makedirs(out_chunks_dir, exist_ok=True)

        # doc_count = number of noise/distractor docs; gold docs are ALWAYS included (Bug 2 fix)
        distractor_ids = noise_pool[:target_noise_count]
        selected_doc_ids = set(gold_doc_ids + distractor_ids)

        print(f"\n--- Generating EnterpriseRAG Tier: {tier_tag} ({len(gold_doc_ids)} gold + {len(distractor_ids)} noise = {len(selected_doc_ids)} total docs) ---")

        processed_doc_map = {}
        for did in selected_doc_ids:
            d_rec = doc_map_by_id[did]
            text = d_rec.get("content", d_rec.get("text", ""))
            if not text:
                continue
            doc_data = process_document(text, doc_id=did, tokenizer=tokenizer)
            with open(os.path.join(out_chunks_dir, f"{did}_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2)
            processed_doc_map[did] = [c["chunk_id"] for c in doc_data.get("child_chunks", [])]

        print(f"Processed {len(processed_doc_map)} chunked documents for EnterpriseRAG {tier_tag}.")

        def format_queries(q_list, q_prefix="q_ent"):
            formatted = []
            for idx, q_rec in enumerate(q_list):
                query_id = str(q_rec.get("question_id", f"{q_prefix}_{idx}"))
                question = q_rec.get("question", "")
                golden_answer = q_rec.get("gold_answer", q_rec.get("answer", ""))
                expected_docs = q_rec.get("expected_doc_ids", [])
                # Coerce numpy arrays / other iterables to Python list
                if isinstance(expected_docs, str):
                    expected_docs = [expected_docs]
                else:
                    try:
                        expected_docs = [str(e) for e in expected_docs]
                    except TypeError:
                        expected_docs = []

                gt_chunks = []
                for ed_id in expected_docs:
                    ed_id_str = str(ed_id)
                    if ed_id_str in processed_doc_map:
                        for cid in processed_doc_map[ed_id_str]:
                            gt_chunks.append({"chunk_id": cid, "text": ""})

                first_doc = str(expected_docs[0]) if len(expected_docs) > 0 else "doc_0"

                formatted.append({
                    "query_id": query_id,
                    "query_group": "Enterprise RAG",
                    "query_type": q_rec.get("question_type", "Factual"),
                    "raw_question": question,
                    "paraphrased_question": question,
                    "golden_answer": golden_answer,
                    "doc_id_source": first_doc,
                    "ground_truth_child_chunks": gt_chunks if gt_chunks else [{"chunk_id": f"{first_doc}_block0_chunk0", "text": ""}],
                    "extended_child_chunks": []
                })
            return formatted

        capped_formatted = format_queries(capped_q_records)
        with open(os.path.join(out_dir, "final_benchmark_capped.json"), "w", encoding="utf-8") as f:
            json.dump(capped_formatted, f, indent=2)

        full_formatted = format_queries(quest_records)
        with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
            json.dump(full_formatted, f, indent=2)

        with open(os.path.join(out_dir, "final_benchmark_enterpriserag.json"), "w", encoding="utf-8") as f:
            json.dump(capped_formatted, f, indent=2)

        print(f"Saved EnterpriseRAG {tier_tag}: final_benchmark_capped.json ({len(capped_formatted)} queries), final_benchmark_full.json ({len(full_formatted)} queries).")


def convert_liverag(doc_counts=[100, 250]):
    random.seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    possible_raw_dirs = [
        os.path.join(base_dir, "data", "raw", "liverag_bench"),
        os.path.join(base_dir, "data", "raw", "liverag", "LiveRAG___benchmark"),
        os.path.join(base_dir, "data", "raw", "liverag"),
    ]

    arrow_files = []
    for rd in possible_raw_dirs:
        af = glob.glob(os.path.join(rd, "**", "*.parquet"), recursive=True) + \
             glob.glob(os.path.join(rd, "**", "*.arrow"), recursive=True)
        if af:
            arrow_files = af
            break

    if not arrow_files:
        print("[WARN] LiveRAG raw parquet/arrow files not found!")
        return

    records = []
    for af in arrow_files:
        records.extend(load_parquet_or_arrow(af))

    print(f"Loaded LiveRAG: {len(records)} total QA records.")

    # Dual-Session Deduplicated Sampling
    s1_records = [r for r in records if str(r.get("Session", "")).lower() in ("first", "both")]
    s2_records = [r for r in records if str(r.get("Session", "")).lower() in ("second", "both")]

    s1_sample = random.sample(s1_records, min(100, len(s1_records)))
    s2_sample = random.sample(s2_records, min(100, len(s2_records)))

    seen_q = set()
    capped_records = []
    for r in s1_sample + s2_sample:
        q_text = str(r.get("Question", r.get("question", ""))).strip()
        idx_val = r.get("Index", q_text)
        if idx_val not in seen_q:
            seen_q.add(idx_val)
            capped_records.append(r)

    print(f"LiveRAG Capped Query Set: {len(capped_records)} unique deduplicated queries (Full set: {len(records)} queries).")

    def _parse_supp_docs(raw_val):
        """Parse Supporting_Documents which may be a JSON string, list of JSON strings, or native list of dicts."""
        if raw_val is None:
            return []
        if isinstance(raw_val, str):
            try:
                parsed = json.loads(raw_val)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except (json.JSONDecodeError, TypeError):
                return []
        if isinstance(raw_val, dict):
            return [raw_val]
        if isinstance(raw_val, list):
            result = []
            for item in raw_val:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            result.append(parsed)
                        elif isinstance(parsed, list):
                            result.extend([x for x in parsed if isinstance(x, dict)])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return result
        # numpy array or other iterable
        try:
            return _parse_supp_docs(list(raw_val))
        except TypeError:
            return []

    doc_map_by_id = {}
    for rec in records:
        supp_docs = _parse_supp_docs(rec.get("Supporting_Documents", rec.get("supporting_documents")))
        for doc_item in supp_docs:
            did = str(doc_item.get("doc_id", ""))
            text = doc_item.get("content", doc_item.get("text", ""))
            if did and text:
                doc_map_by_id[did] = text

    print(f"Extracted {len(doc_map_by_id)} unique LiveRAG supporting documents.")

    # Collect gold doc IDs from capped queries — must always be in corpus (Bug 3 fix)
    capped_gold_ids_set = set()
    for rec in capped_records:
        supp_docs = _parse_supp_docs(rec.get("Supporting_Documents", rec.get("supporting_documents")))
        for doc_item in supp_docs:
            did = str(doc_item.get("doc_id", ""))
            if did and did in doc_map_by_id:
                capped_gold_ids_set.add(did)

    capped_gold_ids = sorted(capped_gold_ids_set)
    noise_pool = sorted([did for did in doc_map_by_id.keys() if did not in capped_gold_ids_set])
    print(f"LiveRAG: {len(capped_gold_ids)} capped gold docs, {len(noise_pool)} available noise docs.")

    tokenizer = get_tokenizer()
    # Use doc_counts param for tiers (Gap 4 fix); doc_count = number of noise docs
    tiers = [(f"corpus_stress_{n}", n) for n in doc_counts]
    tiers.append(("corpus_stress_full", len(noise_pool)))

    for tier_tag, target_noise_count in tiers:
        out_dir = os.path.join(base_dir, "data", "benchmarks", "liverag", tier_tag)
        out_chunks_dir = os.path.join(out_dir, "step1_chunks")
        os.makedirs(out_chunks_dir, exist_ok=True)

        # Gold docs always included; doc_count = noise added on top
        distractor_ids = noise_pool[:target_noise_count]
        selected_dids = sorted(set(capped_gold_ids + distractor_ids))

        print(f"\n--- Generating LiveRAG Tier: {tier_tag} ({len(capped_gold_ids)} gold + {len(distractor_ids)} noise = {len(selected_dids)} total docs) ---")

        processed_doc_map = {}
        for did in selected_dids:
            text = doc_map_by_id[did]
            doc_data = process_document(text, doc_id=did, tokenizer=tokenizer)
            with open(os.path.join(out_chunks_dir, f"{did}_chunks.json"), "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2)
            processed_doc_map[did] = [c["chunk_id"] for c in doc_data.get("child_chunks", [])]

        def format_liverag_queries(rec_list):
            formatted = []
            for idx, rec in enumerate(rec_list):
                question = rec.get("Question", rec.get("question", ""))
                answer = rec.get("Answer", rec.get("answer", ""))
                supp_docs = _parse_supp_docs(rec.get("Supporting_Documents", rec.get("supporting_documents")))

                gt_chunks = []
                first_doc_id = f"liverag_doc_{idx}"
                for s_idx, doc_item in enumerate(supp_docs):
                    did = str(doc_item.get("doc_id", ""))
                    if s_idx == 0 and did:
                        first_doc_id = did
                    if did in processed_doc_map:
                        for cid in processed_doc_map[did]:
                            gt_chunks.append({"chunk_id": cid, "text": ""})

                formatted.append({
                    "query_id": f"q_live_{rec.get('Index', idx)}",
                    "query_group": "LiveRAG Streaming",
                    "query_type": "Factual",
                    "raw_question": question,
                    "paraphrased_question": question,
                    "golden_answer": answer,
                    "doc_id_source": first_doc_id,
                    "ground_truth_child_chunks": gt_chunks if gt_chunks else [{"chunk_id": f"{first_doc_id}_block0_chunk0", "text": ""}],
                    "extended_child_chunks": []
                })
            return formatted

        capped_formatted = format_liverag_queries(capped_records)
        with open(os.path.join(out_dir, "final_benchmark_capped.json"), "w", encoding="utf-8") as f:
            json.dump(capped_formatted, f, indent=2)

        full_formatted = format_liverag_queries(records)
        with open(os.path.join(out_dir, "final_benchmark_full.json"), "w", encoding="utf-8") as f:
            json.dump(full_formatted, f, indent=2)

        with open(os.path.join(out_dir, "final_benchmark_liverag.json"), "w", encoding="utf-8") as f:
            json.dump(capped_formatted, f, indent=2)

        print(f"Saved LiveRAG {tier_tag}: final_benchmark_capped.json ({len(capped_formatted)} queries), final_benchmark_full.json ({len(full_formatted)} queries).")


def main():
    convert_enterpriserag()
    convert_liverag()


if __name__ == "__main__":
    main()
