"""
CRVE/scripts/generate_nfcorpus_manifest.py

Generates the authoritative NFCorpus Eligibility Manifest directly from the
physical PyTerrier inverted index, reporting mutually exclusive exclusion counts
summing to the total lexicon size.
"""

import os
import sys
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import pyterrier as pt
from evaluation.baselines.pyterrier_harness import init_pyterrier

def main():
    init_pyterrier()
    index_path = os.path.abspath("data/cache/terrier_indices/nfcorpus_default/data.properties")
    index = pt.IndexFactory.of(index_path)
    lex = index.getLexicon()
    meta = index.getCollectionStatistics()
    num_docs = meta.getNumberOfDocuments()
    total_lex_size = meta.getNumberOfUniqueTerms()

    counts = {
        "len_lt_2": 0,
        "pure_digits": 0,
        "df_lt_2": 0,
        "cf_lt_3_given_df_ge_2": 0,
        "df_n_gt_0_15": 0,
        "eligible": 0,
    }

    eligible_terms = []

    for entry in lex:
        term = entry.getKey()
        df = entry.getValue().getDocumentFrequency()
        cf = entry.getValue().getFrequency()
        
        if len(term) < 2:
            counts["len_lt_2"] += 1
        elif term.isdigit():
            counts["pure_digits"] += 1
        elif df < 2:
            counts["df_lt_2"] += 1
        elif cf < 3:
            counts["cf_lt_3_given_df_ge_2"] += 1
        elif (df / max(num_docs, 1)) > 0.15:
            counts["df_n_gt_0_15"] += 1
        else:
            counts["eligible"] += 1
            eligible_terms.append(term)

    total_accounted = sum(counts.values())
    assert total_accounted == total_lex_size, f"Mismatch: {total_accounted} vs {total_lex_size}"
    assert counts["eligible"] == 7783, f"Eligible count mismatch: {counts['eligible']} vs 7783"

    # Index fingerprint
    prop_file = index_path
    with open(prop_file, "rb") as f:
        prop_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "dataset": "nfcorpus",
        "index_path": index_path,
        "index_fingerprint_sha256": prop_sha,
        "analyzer_version": "v1_krovetz_suppletion",
        "num_docs": num_docs,
        "total_lexicon_size": total_lex_size,
        "mutually_exclusive_counts": counts,
        "sum_check_passed": bool(total_accounted == total_lex_size),
        "eligible_count": counts["eligible"],
        "frozen_pool_size": 7783,
        "pool_policy_formula": "min(10000, |V_eligible|)",
        "computed_pool_size": min(10000, counts["eligible"]),
    }

    os.makedirs("for_review/pool_phase", exist_ok=True)
    manifest_path = "for_review/pool_phase/nfcorpus_eligibility_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
