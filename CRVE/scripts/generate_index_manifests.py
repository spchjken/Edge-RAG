"""
CRVE/scripts/generate_index_manifests.py

Generates explicit index provenance manifests for development corpora:
scifact, bright_aops, nfcorpus, trec_covid.
Records analyzer version, collection statistics, and cryptographic file hashes.
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

DATASETS = ["scifact", "bright_aops", "nfcorpus", "trec_covid"]

def file_sha256(fpath: str) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    init_pyterrier()
    manifests = {}

    for ds in DATASETS:
        safe_ds = ds.lower().replace("-", "_")
        idx_dir = os.path.abspath(f"data/cache/terrier_indices/{safe_ds}_default")
        prop_file = os.path.join(idx_dir, "data.properties")
        if not os.path.exists(prop_file):
            print(f"Index not found for {ds}: {prop_file}")
            continue

        index = pt.IndexFactory.of(prop_file)
        meta = index.getCollectionStatistics()

        # Hash all key index files in directory
        index_files = {}
        for fname in sorted(os.listdir(idx_dir)):
            if fname.endswith(".properties") or fname.endswith(".meta") or fname.endswith(".log"):
                fpath = os.path.join(idx_dir, fname)
                index_files[fname] = file_sha256(fpath)

        manifest = {
            "dataset": ds,
            "index_dir": idx_dir,
            "analyzer_version": "v1_krovetz_suppletion",
            "num_docs": meta.getNumberOfDocuments(),
            "num_unique_terms": meta.getNumberOfUniqueTerms(),
            "num_postings": meta.getNumberOfPointers(),
            "data_properties_sha256": file_sha256(prop_file),
            "index_files": index_files,
        }

        # Derive manifest content hash
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

        out_path = os.path.join(idx_dir, "index_manifest.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        manifests[ds] = manifest
        print(f"Generated index manifest for {ds} -> {out_path} (hash: {manifest['manifest_sha256'][:12]})")

if __name__ == "__main__":
    main()
