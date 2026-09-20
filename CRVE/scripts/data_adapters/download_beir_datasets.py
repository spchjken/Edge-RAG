#!/usr/bin/env python3
"""
scripts/data_adapters/download_beir_datasets.py

Downloads and extracts the 10 missing BEIR datasets from Table 2 of arXiv:2403.06789 (SPLADE-v3 paper):
1. arguana
2. climate-fever
3. dbpedia-entity
4. fever
5. hotpotqa
6. nq
7. quora
8. scidocs
9. trec-covid
10. webis-touche2020
"""

import os
import sys
import zipfile
import urllib.request
import time
import argparse
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_BEIR_DIR = os.path.join(BASE_DIR, "data", "raw", "beir")

TARGET_DATASETS = [
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "hotpotqa",
    "nq",
    "quora",
    "scidocs",
    "trec-covid",
    "webis-touche2020"
]

BEIR_URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


def download_and_extract(dataset_name: str, target_dir: str = RAW_BEIR_DIR):
    os.makedirs(target_dir, exist_ok=True)
    out_dataset_dir = os.path.join(target_dir, dataset_name)

    # Check if already extracted and contains required files
    corpus_file = os.path.join(out_dataset_dir, "corpus.jsonl")
    queries_file = os.path.join(out_dataset_dir, "queries.jsonl")
    qrels_file = os.path.join(out_dataset_dir, "qrels", "test.tsv")

    if os.path.exists(corpus_file) and os.path.exists(queries_file) and os.path.exists(qrels_file):
        print(f"[{dataset_name}] Already downloaded and extracted. Skipping.", flush=True)
        return True

    url = BEIR_URL_TEMPLATE.format(dataset=dataset_name)
    zip_path = os.path.join(target_dir, f"{dataset_name}.zip")

    print(f"\n[{dataset_name}] Downloading from {url}...", flush=True)
    t0 = time.time()
    
    # Progress reporter for urllib
    def report_hook(count, block_size, total_size):
        if total_size > 0 and count % 2000 == 0:
            percent = count * block_size * 100 / total_size
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  -> Downloading {dataset_name}: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=report_hook)
        print(f"\n  ✅ Download complete ({os.path.getsize(zip_path)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"\n  ❌ Failed to download {dataset_name} from {url}: {e}", flush=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    print(f"  [Extracting] Unzipping {zip_path} to {target_dir}...", flush=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"  ✅ Extracted successfully.", flush=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return True
    except Exception as e:
        print(f"  ❌ Failed to unzip {zip_path}: {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to download")
    args = parser.parse_args()

    datasets_to_download = [args.dataset] if args.dataset else TARGET_DATASETS

    print("=" * 80)
    print(f"📥 BEIR Dataset Downloader (arXiv:2403.06789 Table 2 Missing Corpora)")
    print(f"Target count: {len(datasets_to_download)} datasets")
    print("=" * 80, flush=True)

    success_count = 0
    for idx, ds in enumerate(datasets_to_download, 1):
        print(f"\n--- [{idx}/{len(datasets_to_download)}] Processing {ds} ---", flush=True)
        if download_and_extract(ds):
            success_count += 1

    print(f"\n🎉 Finished downloading {success_count}/{len(datasets_to_download)} datasets!", flush=True)


if __name__ == "__main__":
    main()
