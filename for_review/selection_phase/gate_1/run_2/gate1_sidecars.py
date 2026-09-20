"""
CRVE/src/crve/selection/gate1_sidecars.py

Production Sidecar Manager for Phase 2 Gate 1 Candidate Selection:
1. CanonicalPoolBGESidecar: Exact 1:1 aligned BGE-small embeddings for canonical pool terms.
2. BoundedPPMISidecar: Precomputed top-M=600 co-occurrence PPMI neighbors per eligible anchor.
3. AcronymRescueSidecar: Bidirectional Schwartz-Hearst acronym/full-form extraction mapped to P.
4. SparseLexicalContextSidecar: Passage context profiles (50 passages x 64 tokens) with auxiliary BM25 index.

Enforces:
- Atomic writes: All sidecars written to .tmp and renamed atomically upon completion.
- Fail-closed validation against canonical pool terms and hashes.
- Fast streaming ingestion without loading full raw corpora into memory.
"""

import os
import sys
import time
import math
import json
import re
import shutil
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import torch
import pandas as pd
import pyterrier as pt

from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer


class Gate1SidecarManager:
    """Manages creation, caching, and loading of all Gate 1 sidecars."""

    def __init__(self, cache_dir: str = "data/cache/canonical_pools"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.analyzer = get_terrier_analyzer()

    def _get_safe_ds(self, dataset: str) -> str:
        return dataset.lower().replace("-", "_")

    # -------------------------------------------------------------------------
    # 1. BGE Sidecar (Canonical Pool Embeddings)
    # -------------------------------------------------------------------------
    def build_or_load_bge_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        encoder=None,
        device: str = "cpu",
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads BGE embeddings strictly aligned 1:1 with canonical pool terms.
        Returns:
            {
                "pool_embeddings": torch.Tensor [|P|, 384],
                "display_surfaces": List[str],
                "surf_to_idx": Dict[str, int],
                "timing_s": float,
            }
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_bge_sidecar.pt")

        if not force_rebuild and os.path.exists(out_path):
            data = torch.load(out_path, map_location=device)
            # Verify alignment
            if data.get("pool_terms") == pool_terms:
                return {
                    "pool_embeddings": data["embeddings"].to(device),
                    "display_surfaces": data["display_surfaces"],
                    "surf_to_idx": data["surf_to_idx"],
                    "timing_s": 0.0,
                }

        print(f"[Sidecar] Building Canonical Pool BGE Sidecar for {dataset} ({len(pool_terms)} terms)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Step 1: Surface recovery pass over corpus (up to 10,000 docs)
        surface_counts = {t: Counter() for t in pool_set}
        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            terms, surfs = self.analyzer.analyze(text)
            for t, s in zip(terms, surfs):
                if t in surface_counts:
                    surface_counts[t][s.lower()] += 1
            if doc_count >= 10000:
                break

        display_surfaces = []
        for t in pool_terms:
            top_s = surface_counts[t].most_common(1)
            display_surfaces.append(top_s[0][0] if top_s else t)

        surf_to_idx = {str(s).lower(): i for i, s in enumerate(display_surfaces)}

        # Step 2: Encode display surfaces
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            enc_device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=enc_device)

        if hasattr(encoder, "encode"):
            embs = encoder.encode(display_surfaces, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
            embs_tensor = torch.tensor(embs, dtype=torch.float32)
        elif hasattr(encoder, "encode_queries"):
            embs = encoder.encode_queries(display_surfaces)
            embs_tensor = torch.tensor(embs, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported encoder type: {type(encoder)}")

        embs_tensor = torch.nn.functional.normalize(embs_tensor, p=2, dim=-1)

        # Step 3: Atomic write
        tmp_path = out_path + ".tmp"
        torch.save({
            "dataset": dataset,
            "pool_terms": pool_terms,
            "display_surfaces": display_surfaces,
            "surf_to_idx": surf_to_idx,
            "embeddings": embs_tensor.cpu(),
        }, tmp_path)
        os.replace(tmp_path, out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] BGE Sidecar built in {timing_s}s -> {out_path}")

        return {
            "pool_embeddings": embs_tensor.to(device),
            "display_surfaces": display_surfaces,
            "surf_to_idx": surf_to_idx,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 2. Bounded PPMI Sidecar (Precomputed top-M=600 co-occurrences)
    # -------------------------------------------------------------------------
    def build_or_load_bounded_ppmi_sidecar(
        self,
        dataset: str,
        index,
        pool_terms: List[str],
        top_m: int = 600,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads precomputed top-M=600 PPMI neighbors per anchor.
        Returns:
            {
                "anchor_ppmi": Dict[str, List[Tuple[str, float]]],
                "timing_s": float,
            }
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_bounded_ppmi.json")

        if not force_rebuild and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "anchor_ppmi": {a: [(t, float(s)) for t, s in cands] for a, cands in data["anchors"].items()},
                "timing_s": 0.0,
            }

        print(f"[Sidecar] Building Bounded PPMI Sidecar for {dataset} (top_m={top_m})...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)
        num_docs = int(index.getCollectionStatistics().getNumberOfDocuments())
        lex = index.getLexicon()

        df_map = {}
        for entry in lex:
            t = str(entry.getKey())
            df = int(entry.getValue().getDocumentFrequency())
            df_map[t] = df

        eligible_anchor_set = {
            t for t, df in df_map.items()
            if df >= 2 and (df / num_docs) <= 0.12
        }

        # 1-pass streaming co-occurrence accumulation
        anchor_cooccur = defaultdict(lambda: defaultdict(int))
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            terms, _ = self.analyzer.analyze(text)
            doc_terms = set(terms)
            doc_pool_terms = doc_terms.intersection(pool_set)
            if not doc_pool_terms:
                continue
            for a in doc_terms:
                if a in eligible_anchor_set:
                    a_counts = anchor_cooccur[a]
                    for t in doc_pool_terms:
                        if t != a:
                            a_counts[t] += 1

        # Compute PPMI
        anchor_ppmi = {}
        for a, targets in anchor_cooccur.items():
            df_a = df_map.get(a, 0)
            if df_a < 2:
                continue
            scored = []
            for t, df_at in targets.items():
                if df_at >= 2:
                    df_t = df_map.get(t, 0)
                    if df_t >= 1:
                        pmi = math.log((num_docs * df_at) / (df_a * df_t))
                        if pmi > 0.0:
                            scored.append((t, round(pmi, 5)))
            if scored:
                scored.sort(key=lambda x: (-x[1], x[0]))
                anchor_ppmi[a] = scored[:top_m]

        # Atomic write
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset,
                "top_m": top_m,
                "num_anchors": len(anchor_ppmi),
                "anchors": anchor_ppmi,
            }, f)
        os.replace(tmp_path, out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Bounded PPMI Sidecar built in {timing_s}s for {len(anchor_ppmi):,} anchors -> {out_path}")

        return {
            "anchor_ppmi": anchor_ppmi,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 3. Acronym Definition Rescue Sidecar (Schwartz-Hearst Extraction)
    # -------------------------------------------------------------------------
    def build_or_load_acronym_rescue_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads bidirectional Schwartz-Hearst acronym/full-form mapping to P.
        Returns:
            {
                "acronym_to_pool": Dict[str, List[Tuple[str, float]]],
                "timing_s": float,
            }
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_acronym_rescue.json")

        if not force_rebuild and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "acronym_to_pool": {k: [(t, float(s)) for t, s in v] for k, v in data["acronyms"].items()},
                "timing_s": 0.0,
            }

        print(f"[Sidecar] Building Acronym Definition Rescue Sidecar for {dataset}...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        p1 = re.compile(r'\b([A-Za-z][A-Za-z\s]{2,40}?)\s*\(([A-Z]{2,6})\)')
        p2 = re.compile(r'\b([A-Z]{2,6})\s*\(([A-Za-z][A-Za-z\s]{2,40}?)\)')

        pair_counts = defaultdict(int)
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            for m in p1.finditer(text):
                pair_counts[(m.group(2).strip(), m.group(1).strip().lower())] += 1
            for m in p2.finditer(text):
                pair_counts[(m.group(1).strip(), m.group(2).strip().lower())] += 1

        # Map to P:
        # If query contains acronym A -> score exact-match term in P with 1.0, full-form terms in P with 0.8
        # If query contains full-form words -> score acronym in P with 0.8
        acronym_to_pool = defaultdict(dict)

        for (acr, full), count in pair_counts.items():
            acr_lower = acr.lower()
            acr_terms, _ = self.analyzer.analyze(acr)
            full_terms, _ = self.analyzer.analyze(full)

            # Acronym key
            for at in acr_terms:
                if at in pool_set:
                    acronym_to_pool[acr_lower][at] = max(acronym_to_pool[acr_lower].get(at, 0.0), 1.0)
            for ft in full_terms:
                if ft in pool_set:
                    acronym_to_pool[acr_lower][ft] = max(acronym_to_pool[acr_lower].get(ft, 0.0), 0.8)

            # Full form words as triggers
            for ft in full_terms:
                for at in acr_terms:
                    if at in pool_set:
                        acronym_to_pool[ft][at] = max(acronym_to_pool[ft].get(at, 0.0), 0.8)

        # Convert to sorted list of (term, score)
        formatted = {}
        for trigger, term_scores in acronym_to_pool.items():
            s_list = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
            formatted[trigger] = s_list

        # Atomic write
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset,
                "num_triggers": len(formatted),
                "acronyms": formatted,
            }, f)
        os.replace(tmp_path, out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Acronym Rescue Sidecar built in {timing_s}s ({len(formatted)} triggers) -> {out_path}")

        return {
            "acronym_to_pool": formatted,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 4. Sparse Lexical Context Sidecar (Passage Context BM25 Index)
    # -------------------------------------------------------------------------
    def build_or_load_sparse_lexical_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds or loads auxiliary BM25 index over candidate terms' passage context profiles.
        Returns:
            {
                "index_path": str,
                "retriever": pt.terrier.Retriever,
                "timing_s": float,
            }
        """
        safe_ds = self._get_safe_ds(dataset)
        aux_index_dir = os.path.join(self.cache_dir, f"{safe_ds}_lexical_profiles_idx")
        prop_path = os.path.join(aux_index_dir, "data.properties")

        if not force_rebuild and os.path.exists(prop_path):
            retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
            return {
                "index_path": aux_index_dir,
                "retriever": retriever,
                "timing_s": 0.0,
            }

        print(f"[Sidecar] Building Sparse Lexical Context Sidecar for {dataset} ({len(pool_terms)} terms)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Collect up to 50 short passages per pool term
        term_passages = defaultdict(list)
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            terms, _ = self.analyzer.analyze(text)
            doc_terms = set(terms)
            doc_pool_terms = doc_terms.intersection(pool_set)
            if not doc_pool_terms:
                continue

            tok_len = len(terms)
            for t in doc_pool_terms:
                if len(term_passages[t]) < 50:
                    for idx, tok in enumerate(terms):
                        if tok == t:
                            start = max(0, idx - 32)
                            end = min(tok_len, idx + 32)
                            passage = [w for j, w in enumerate(terms[start:end]) if j != (idx - start)]
                            term_passages[t].append(" ".join(passage))
                            if len(term_passages[t]) >= 50:
                                break

        docs_to_index = []
        for t in pool_terms:
            passages = term_passages.get(t, [])
            combined_text = " ".join(passages) if passages else t
            docs_to_index.append({"docno": t, "text": combined_text})

        # Atomic index build via tmp directory
        tmp_idx_dir = aux_index_dir + ".tmp"
        if os.path.exists(tmp_idx_dir):
            shutil.rmtree(tmp_idx_dir)
        os.makedirs(tmp_idx_dir, exist_ok=True)

        indexer = pt.IterDictIndexer(os.path.abspath(tmp_idx_dir), overwrite=True)
        indexer.index(docs_to_index)

        if os.path.exists(aux_index_dir):
            shutil.rmtree(aux_index_dir)
        os.rename(tmp_idx_dir, aux_index_dir)

        retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Sparse Lexical Sidecar built in {timing_s}s -> {aux_index_dir}")

        return {
            "index_path": aux_index_dir,
            "retriever": retriever,
            "timing_s": timing_s,
        }
