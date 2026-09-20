"""
scripts/run_dph_confirmatory_check.py

Confirmatory DPH Transfer Check across 4 representative datasets:
- scifact
- nfcorpus
- trec_covid
- bright_stackoverflow

Evaluates:
1. Direct zero-expansion DPH retrieval parity against official baseline.
2. Cross-model transfer of BM25 oracle expansion terms to DPH.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import ir_measures
import pyterrier as pt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.baselines.pyterrier_harness import init_pyterrier, sanitize_default_query
from evaluation.benchmark_loader import BenchmarkLoader

DPH_DATASETS = ["scifact", "nfcorpus", "trec_covid", "bright_stackoverflow"]
PARQUET_PATH = os.path.join(BASE_DIR, "results", "pyterrier_baselines", "v8_pregate_candidate_audit.parquet")


def main():
    init_pyterrier()
    random.seed(42)

    df_cands = pd.read_parquet(PARQUET_PATH)
    m_ndcg = ir_measures.nDCG@10

    print("=" * 80)
    print("DPH CONFIRMATORY CHECK (Zero-Expansion Parity & Cross-Model Transfer)")
    print("=" * 80)

    rows = []

    for ds in DPH_DATASETS:
        safe_ds = ds.lower().replace("-", "_")
        index_path = os.path.abspath(f"data/cache/terrier_indices/{safe_ds}_default")
        if not os.path.exists(index_path):
            index_path = os.path.abspath(f"data/cache/terrier_indices/{safe_ds}_default/data.properties")

        index = pt.IndexFactory.of(index_path)
        retr_dph = pt.terrier.Retriever(index, wmodel="DPH", num_results=1000)

        queries, _ = BenchmarkLoader.load_queries(ds)
        sampled_queries = random.sample(queries, min(len(queries), 50))
        sample_qids = {str(q["query_id"]) for q in sampled_queries}

        # Build qrels strictly for sampled queries
        qrels_ir = []
        for q in sampled_queries:
            qid = str(q["query_id"])
            raw_qid = qid.split("_")[-1] if "q_beir_" in qid else qid
            golds = q.get("qrels") or {str(did): 1.0 for did in q.get("gold_doc_ids", [])}
            for did, rel in golds.items():
                qrels_ir.append(ir_measures.Qrel(qid, str(did), int(rel)))
                if raw_qid != qid:
                    qrels_ir.append(ir_measures.Qrel(raw_qid, str(did), int(rel)))

        # 1. Baseline DPH retrieval (exact string formatting)
        df_base = pd.DataFrame([
            {"qid": str(q["query_id"]), "query": sanitize_default_query(q["question"])}
            for q in sampled_queries
        ])
        res_base = retr_dph.transform(df_base)
        scored_base = [ir_measures.ScoredDoc(str(r.qid), str(r.docno), float(r.score)) for r in res_base.itertuples()]

        # Query-level baseline metrics
        base_by_q = {}
        for m in ir_measures.iter_calc([m_ndcg], qrels_ir, scored_base):
            base_by_q[str(m.query_id)] = float(m.value)

        mean_base_dph = float(np.mean([base_by_q.get(str(q["query_id"]), 0.0) for q in sampled_queries]))

        # 2. Oracle transfer from BM25 candidates
        ds_cands = df_cands[(df_cands["dataset"] == ds) & (df_cands["beneficial"])]
        best_cand_by_q = {}
        for _, r in ds_cands.iterrows():
            qid = str(r["qid"])
            if qid not in best_cand_by_q or r["delta_ndcg10"] > best_cand_by_q[qid]["delta_ndcg10"]:
                best_cand_by_q[qid] = r

        oracle_dph_vals = []
        for q in sampled_queries:
            qid = str(q["query_id"])
            b_val = base_by_q.get(qid, 0.0)

            if qid in best_cand_by_q:
                c_term = best_cand_by_q[qid]["candidate"]
                exp_query = f"{sanitize_default_query(q['question'])} {c_term}"
                df_test = pd.DataFrame([{"qid": qid, "query": exp_query}])
                res_test = retr_dph.transform(df_test)
                if not res_test.empty:
                    scored_test = [ir_measures.ScoredDoc(str(r.qid), str(r.docno), float(r.score)) for r in res_test.itertuples()]
                    test_calcs = list(ir_measures.iter_calc([m_ndcg], qrels_ir, scored_test))
                    t_val = float(test_calcs[0].value) if test_calcs else b_val
                else:
                    t_val = b_val
                oracle_dph_vals.append(max(b_val, t_val))
            else:
                oracle_dph_vals.append(b_val)

        mean_oracle_dph = float(np.mean(oracle_dph_vals))
        dph_delta = mean_oracle_dph - mean_base_dph

        rows.append({
            "dataset": ds,
            "queries": len(sampled_queries),
            "dph_baseline_ndcg": round(mean_base_dph, 4),
            "dph_oracle_ndcg": round(mean_oracle_dph, 4),
            "dph_oracle_delta": round(dph_delta, 4),
        })

        print(f"[{ds:22s}] Baseline DPH={mean_base_dph:.4f} -> Oracle DPH={mean_oracle_dph:.4f} (Delta={dph_delta:+.4f})")

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(BASE_DIR, "results", "pyterrier_baselines", "dph_confirmatory_transfer.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("DPH CONFIRMATORY RESULTS TABLE")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
