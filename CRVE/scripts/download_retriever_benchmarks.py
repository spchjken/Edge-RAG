#!/usr/bin/env python3
"""
scripts/download_retriever_benchmarks.py

Downloads raw benchmark datasets for Retriever-Only evaluation from Hugging Face 
and established data hubs into `data/raw/<dataset_name>/`.

Supported Datasets:
1. BEIR Subsets (from 'BeIR/<dataset>' on Hugging Face / UKP Darmstadt Hub):
   - scifact: Scientific claim verification (5.1k docs, 300 test queries)
   - nfcorpus: Biomedical/nutrition fact search (3.6k docs, 323 test queries)
   - fiqa: Financial opinion QA (57.6k docs, 648 test queries)
   - trec-covid: Biomedical research (171.3k docs, 50 queries)
2. MultiHop-RAG (from 'yixuantt/MultiHop-RAG' on Hugging Face / GitHub Hub)
3. BRIGHT (from 'xlangai/BRIGHT' on Hugging Face)
4. FinanceBench (from 'patronus-ai/financebench' on Hugging Face)
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    snapshot_download = None

DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_beir_subset_zip(dataset_name: str, target_dir: str):
    """Download BEIR dataset directly from official UKP Darmstadt repository as zip."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    zip_path = os.path.join(target_dir, f"{dataset_name}.zip")
    extract_path = os.path.join(target_dir, dataset_name)
    
    print(f"[*] Downloading BEIR {dataset_name} from {url}...")
    os.makedirs(target_dir, exist_ok=True)
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())

    print(f"    -> Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"    [OK] Successfully extracted BEIR {dataset_name} to {extract_path}")


def download_beir_dataset(dataset_name: str):
    """Download BEIR dataset via UKP zip (complete with qrels)."""
    target_dir = os.path.join(DATA_RAW_DIR, "beir")
    os.makedirs(target_dir, exist_ok=True)
    dataset_dest = os.path.join(target_dir, dataset_name)
    
    # Check if complete (has corpus.jsonl and qrels)
    if os.path.exists(dataset_dest) and os.path.exists(os.path.join(dataset_dest, "corpus.jsonl")) and os.path.exists(os.path.join(dataset_dest, "qrels")):
        print(f"[SKIP] BEIR dataset '{dataset_name}' already exists and is complete at {dataset_dest}")
        return True

    try:
        download_beir_subset_zip(dataset_name, target_dir)
        return True
    except Exception as zip_err:
        print(f"[FAILED] Failed to download BEIR {dataset_name}: {zip_err}")
        return False


def download_multihop_rag():
    """Download MultiHop-RAG dataset from Hugging Face or GitHub raw files."""
    dest_dir = os.path.join(DATA_RAW_DIR, "multihop_rag")
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(dest_dir) and any(f.endswith(".json") for f in os.listdir(dest_dir)):
        print(f"[SKIP] MultiHop-RAG already exists at {dest_dir}")
        return True

    print("\n[*] Downloading MultiHop-RAG dataset from Hugging Face ('yixuantt/MultiHopRAG')...")
    
    # Try Hugging Face Hub snapshot
    try:
        if snapshot_download is not None:
            snapshot_download(repo_id="yixuantt/MultiHopRAG", repo_type="dataset", local_dir=dest_dir)
            print(f"[SUCCESS] Downloaded MultiHop-RAG to {dest_dir}")
            return True
    except Exception as e:
        print(f"    [!] HuggingFace snapshot failed ({e}), trying GitHub direct raw download...")

    # Direct raw files download from official GitHub repository
    raw_files = {
        "corpus.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/corpus.json",
        "MultiHopRAG.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json"
    }
    
    try:
        for fname, url in raw_files.items():
            out_file = os.path.join(dest_dir, fname)
            print(f"    -> Downloading {fname} from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(out_file, 'wb') as out_f:
                out_f.write(response.read())
        print(f"[SUCCESS] Downloaded MultiHop-RAG raw files to {dest_dir}")
        return True
    except Exception as err:
        print(f"[FAILED] Error downloading MultiHop-RAG: {err}")
        return False


def download_financebench():
    """Download FinanceBench dataset from Hugging Face ('PatronusAI/financebench')."""
    dest_dir = os.path.join(DATA_RAW_DIR, "financebench")
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(dest_dir) and any(f.endswith(".json") or f.endswith(".jsonl") for f in os.listdir(dest_dir)):
        print(f"[SKIP] FinanceBench already exists at {dest_dir}")
        return True

    print("\n[*] Downloading FinanceBench dataset from Hugging Face ('PatronusAI/financebench')...")
    try:
        if load_dataset is not None:
            ds = load_dataset("PatronusAI/financebench")
            for split in ds.keys():
                out_path = os.path.join(dest_dir, f"financebench_{split}.jsonl")
                ds[split].to_json(out_path)
            print(f"[SUCCESS] Downloaded FinanceBench to {dest_dir}")
            return True
        elif snapshot_download is not None:
            snapshot_download(repo_id="PatronusAI/financebench", repo_type="dataset", local_dir=dest_dir)
            print(f"[SUCCESS] Downloaded FinanceBench snapshot to {dest_dir}")
            return True
    except Exception as err:
        print(f"[FAILED] Error downloading FinanceBench: {err}")
        return False


def download_bright():
    """Download BRIGHT dataset from Hugging Face ('xlangai/BRIGHT')."""
    dest_dir = os.path.join(DATA_RAW_DIR, "bright")
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"[SKIP] BRIGHT dataset already exists at {dest_dir}")
        return True

    print("\n[*] Downloading BRIGHT dataset from Hugging Face ('xlangai/BRIGHT')...")
    try:
        if snapshot_download is not None:
            snapshot_download(repo_id="xlangai/BRIGHT", repo_type="dataset", local_dir=dest_dir)
            print(f"[SUCCESS] Downloaded BRIGHT snapshot to {dest_dir}")
            return True
        elif load_dataset is not None:
            for split_name in ["documents", "queries", "qrels"]:
                try:
                    ds = load_dataset("xlangai/BRIGHT", split_name)
                    out_path = os.path.join(dest_dir, f"{split_name}.jsonl")
                    for s in ds.keys():
                        ds[s].to_json(os.path.join(dest_dir, f"{split_name}_{s}.jsonl"))
                except Exception as sub_e:
                    print(f"    [!] Sub-dataset '{split_name}' error: {sub_e}")
            print(f"[SUCCESS] Downloaded BRIGHT to {dest_dir}")
            return True
    except Exception as err:
        print(f"[FAILED] Error downloading BRIGHT: {err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for Retriever-Only benchmark evaluation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "fiqa", "multihop_rag", "financebench"],
        help="Datasets to download: scifact, nfcorpus, fiqa, trec-covid, multihop_rag, financebench, bright, all",
    )
    args = parser.parse_args()
    
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    requested = set(args.datasets)
    if "all" in requested:
        requested = {"scifact", "nfcorpus", "fiqa", "trec-covid", "multihop_rag", "financebench", "bright"}
    
    print(f"=== Downloading Raw Retriever Benchmarks to: {DATA_RAW_DIR} ===")
    print(f"Target datasets: {sorted(list(requested))}\n")
    
    results = {}
    
    # BEIR datasets
    for beir_name in ["scifact", "nfcorpus", "fiqa", "trec-covid"]:
        if beir_name in requested:
            res = download_beir_dataset(beir_name)
            results[f"beir_{beir_name}"] = res
            
    # MultiHop-RAG
    if "multihop_rag" in requested:
        res = download_multihop_rag()
        results["multihop_rag"] = res
        
    # FinanceBench
    if "financebench" in requested:
        res = download_financebench()
        results["financebench"] = res
        
    # BRIGHT
    if "bright" in requested:
        res = download_bright()
        results["bright"] = res

    print("\n" + "=" * 50)
    print("=== Raw Dataset Download Summary ===")
    print("=" * 50)
    for name, status in results.items():
        status_str = "[OK] Downloaded / Present" if status else "[FAILED]"
        print(f"  - {name:<20}: {status_str}")
    print("=" * 50)


if __name__ == "__main__":
    main()
