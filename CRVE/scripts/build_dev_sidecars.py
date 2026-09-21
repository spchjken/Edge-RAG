#!/usr/bin/env python3
"""
CRVE/scripts/build_dev_sidecars.py

Authoritative pre-builder for Phase 2.1a Gate 1 Sidecars across development corpora:
- scifact
- bright_aops
- nfcorpus
- trec_covid

Enforces fail-closed configuration validation and authentic provenance generation.
"""

import os
import sys
import time
import json
import argparse
import yaml
import torch
import pyterrier as pt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from crve.selection.gate1_sidecars import (
    Gate1SidecarManager,
    validate_gate1_config,
    compute_pool_sha256,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-build Gate 1 Sidecars with Authentic Provenance")
    parser.add_argument(
        "--datasets",
        type=str,
        default="scifact,bright_aops,nfcorpus,trec_covid",
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--frozen-config-path",
        type=str,
        required=True,
        help="Path to frozen gate1_phase2_1a.yaml",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of all sidecar types bypassing existing caches",
    )
    args = parser.parse_args()

    if not os.path.exists(args.frozen_config_path):
        raise FileNotFoundError(f"FATAL: Frozen config not found at {args.frozen_config_path}")

    with open(args.frozen_config_path, "r", encoding="utf-8") as f:
        frozen_config = yaml.safe_load(f)

    # Validate config fail-closed
    validate_gate1_config(frozen_config)
    print(f"[Config] Validated frozen config schema successfully: {args.frozen_config_path}")

    if not pt.started():
        heap_gib = int(frozen_config.get("memory_watchdog", {}).get("jvm_heap_gib", 4))
        pt.java.set_memory_limit(heap_gib * 1024)
        pt.init()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    mgr = Gate1SidecarManager(config=frozen_config)

    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Encoder] Loading BGE encoder on {device}...")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    expected_sizes = frozen_config.get("pool_spec", {}).get("sizes", {})
    expected_hashes = frozen_config.get("pool_spec", {}).get("hashes", {})

    for ds in datasets:
        safe_ds = ds.lower().replace("-", "_")
        print(f"\n=======================================================")
        print(f"Building/Validating Sidecars for {ds} (force_rebuild={args.force_rebuild})")
        print(f"=======================================================")

        pool_path = f"data/cache/canonical_pools/{safe_ds}_canonical_pool.json"
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"FATAL: Canonical pool not found for {ds} at {pool_path}")

        with open(pool_path, "r", encoding="utf-8") as f:
            pool_data = json.load(f)
        pool_terms = pool_data["terms"]

        # Validate pool integrity against frozen config
        if ds in expected_sizes:
            assert len(pool_terms) == expected_sizes[ds], (
                f"FATAL: Pool size mismatch for {ds}: {len(pool_terms)} != {expected_sizes[ds]}"
            )
        pool_sha = compute_pool_sha256(pool_terms)
        if ds in expected_hashes:
            assert pool_sha == expected_hashes[ds], (
                f"FATAL: Pool SHA-256 mismatch for {ds}: {pool_sha} != {expected_hashes[ds]}"
            )

        idx_path = f"data/cache/terrier_indices/{safe_ds}_default/data.properties"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"FATAL: PyTerrier index not found for {ds} at {idx_path}")
        index = pt.IndexFactory.of(os.path.abspath(idx_path))
        num_docs = index.getCollectionStatistics().getNumberOfDocuments()

        # 1. BGE Sidecar
        t0 = time.perf_counter()
        bge = mgr.build_or_load_bge_sidecar(
            ds,
            pool_terms,
            num_docs=num_docs,
            encoder=encoder,
            device=device,
            force_rebuild=args.force_rebuild,
        )
        print(f"  [BGE] Ready in {time.perf_counter() - t0:.2f}s (Shape: {bge['pool_embeddings'].shape})")

        # 2. Bounded PPMI Sidecar
        t0 = time.perf_counter()
        ppmi_top_m = int(frozen_config.get("sidecars", {}).get("ppmi", {}).get("top_m", 600))
        ppmi = mgr.build_or_load_bounded_ppmi_sidecar(
            ds,
            index,
            pool_terms,
            num_docs=num_docs,
            top_m=ppmi_top_m,
            force_rebuild=args.force_rebuild,
        )
        print(f"  [PPMI] Ready in {time.perf_counter() - t0:.2f}s ({len(ppmi['anchor_ppmi']):,} anchors)")

        # 3. Acronym Rescue Sidecar
        t0 = time.perf_counter()
        acronym = mgr.build_or_load_acronym_rescue_sidecar(
            ds,
            pool_terms,
            num_docs=num_docs,
            force_rebuild=args.force_rebuild,
        )
        print(f"  [Acronym] Ready in {time.perf_counter() - t0:.2f}s ({len(acronym['acronym_to_pool']):,} triggers)")

        # 4. Sparse Lexical Sidecar
        t0 = time.perf_counter()
        lex = mgr.build_or_load_sparse_lexical_sidecar(
            ds,
            pool_terms,
            num_docs=num_docs,
            force_rebuild=args.force_rebuild,
        )
        print(f"  [SparseLex] Ready in {time.perf_counter() - t0:.2f}s -> {lex['index_path']}")

    print("\nAll dev sidecars successfully built/validated with authentic provenance!")


if __name__ == "__main__":
    main()
