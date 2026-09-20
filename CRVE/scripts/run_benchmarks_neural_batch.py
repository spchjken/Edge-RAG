"""
scripts/run_benchmarks_neural_batch.py

Helper orchestrator to evaluate standard PyTerrier neural baselines
(BGE_Small_Dense + SPLADE_v3_PISA) across curated batches of BEIR and BRIGHT benchmarks.
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.run_pyterrier_baselines import generate_summary_markdown, PIPELINE_SUITES

TIERS = {
    "compact": [
        "scifact",                    # 5k docs (already done)
        "nfcorpus",                   # 3.6k docs (already done)
        "arguana",                    # 8.6k docs (already done)
        "bright_pony",                # 7.8k docs (already done)
        "bright_theoremqa_theorems",  # 23.8k docs
        "scidocs",                    # 25.6k docs
    ],
    "medium": [
        "bright_economics",           # 50.2k docs
        "bright_psychology",          # 52.8k docs
        "bright_biology",             # 57.3k docs
        "fiqa",                       # 57.6k docs
        "bright_sustainable_living",  # 60.7k docs
        "bright_robotics",            # 61.9k docs
    ],
    "large": [
        "bright_stackoverflow",       # 107k docs
        "bright_earth_science",       # 121k docs
        "trec_covid",                 # 171k docs
        "bright_aops",                # 188k docs
        "bright_theoremqa_questions", # 188k docs
        "webis_touche2020",           # 382k docs
        "bright_leetcode",            # 413k docs
        "quora",                      # 522k docs
    ],
    "massive": [
        "nq",                         # 2.68M docs
        "dbpedia_entity",             # 4.63M docs
        "hotpotqa",                   # 5.23M docs
        "climate_fever",              # 5.41M docs
        "fever",                      # 5.41M docs
    ],
}


def is_dataset_done(csv_path: str, dataset_name: str, needed_pipelines: list) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if "pipeline" not in df.columns or "dataset" not in df.columns:
            return False
        norm_d = dataset_name.lower().replace("-", "_").replace("_doc_level", "")
        d_rows = df[df["dataset"].astype(str).str.lower().str.replace("-", "_").str.replace("_doc_level", "") == norm_d]
        done = set(d_rows["pipeline"].dropna().unique())
        return set(needed_pipelines).issubset(done)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Neural Baselines")
    parser.add_argument("--tier", type=str, choices=list(TIERS.keys()) + ["all", "all_submillion"], default="compact")
    parser.add_argument("--datasets", nargs="+", default=None, help="Explicit list of dataset names to run")
    parser.add_argument("--pipelines", type=str, default="neural", choices=["neural", "all8"])
    parser.add_argument("--overwrite", action="store_true", help="Force re-evaluation of completed datasets")
    args = parser.parse_args()

    if args.datasets:
        target_datasets = args.datasets
    elif args.tier == "all":
        target_datasets = TIERS["compact"] + TIERS["medium"] + TIERS["large"] + TIERS["massive"]
    elif args.tier == "all_submillion":
        target_datasets = TIERS["compact"] + TIERS["medium"] + TIERS["large"]
    else:
        target_datasets = TIERS[args.tier]

    csv_path = os.path.join(BASE_DIR, "results", "pyterrier_baselines", "pyterrier_baselines_results.csv")
    output_dir = os.path.join(BASE_DIR, "results", "pyterrier_baselines")
    needed_pipelines = PIPELINE_SUITES[args.pipelines]

    print(f"=======================================================")
    print(f"[Batch Runner] Target datasets ({len(target_datasets)}): {target_datasets}")
    print(f"[Batch Runner] Pipelines: {needed_pipelines}")
    print(f"[Batch Runner] Target CSV: {csv_path}")
    print(f"=======================================================")

    for idx, d_name in enumerate(target_datasets, 1):
        if not args.overwrite and is_dataset_done(csv_path, d_name, needed_pipelines):
            print(f"[{idx}/{len(target_datasets)}] [SKIP] '{d_name}' already evaluated for {needed_pipelines}.")
            continue

        print(f"\n[{idx}/{len(target_datasets)}] >>> Launching worker for: {d_name}...")
        cmd = [
            sys.executable, "-u", os.path.join(BASE_DIR, "scripts", "run_pyterrier_baselines.py"),
            "--dataset", d_name,
            "--pipelines", args.pipelines,
            "--no-isolate",
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd)
        dur = round(time.perf_counter() - t0, 1)
        if res.returncode == 0:
            print(f"[{idx}/{len(target_datasets)}] [DONE] '{d_name}' completed in {dur}s.")
        else:
            print(f"[{idx}/{len(target_datasets)}] [ERROR] '{d_name}' exited with code {res.returncode} after {dur}s.")

    generate_summary_markdown(csv_path, output_dir)
    print(f"\n[Batch Runner] Finished all {len(target_datasets)} target datasets.")


if __name__ == "__main__":
    main()
