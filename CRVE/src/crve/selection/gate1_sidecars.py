"""
CRVE/src/crve/selection/gate1_sidecars.py

Production Sidecar Manager for Phase 2 Gate 1 Candidate Selection:
1. CanonicalPoolBGESidecar: Exact 1:1 aligned BGE-small embeddings for canonical pool terms (FP16).
2. BoundedPPMISidecar: Precomputed top-M=600 co-occurrence PPMI neighbors per eligible anchor (exact float).
3. AcronymRescueSidecar: Bidirectional Schwartz-Hearst acronym/full-form extraction mapped to P with corpus frequency.
4. SparseLexicalContextSidecar: Passage context profiles (50 passages x 64 tokens, reservoir sampled) with auxiliary BM25 index.

Enforces:
- Full provenance metadata header (pool_sha256, num_docs, analyzer_version, top_m, created_at).
- Strict cache validation: Stale caches with mismatched pool hashes or document counts are rejected.
- Build-time resource guards: Time limits, memory watchdog (abort at 12 GiB), and disk footprint caps (<500 MB).
- Atomic writes: All sidecars written to .tmp and renamed atomically upon completion.
"""

import os
import sys
import time
import math
import json
import re
import shutil
import hashlib
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
from collections import defaultdict, Counter

import numpy as np
import torch
import psutil
import pyterrier as pt

from evaluation.benchmark_loader import BenchmarkLoader
from evaluation.baselines.pyterrier_qe import get_terrier_analyzer

ANALYZER_VERSION = "v1_krovetz_suppletion"
DEFAULT_BUILD_CEILINGS_SEC = {
    "bge": 300.0,
    "ppmi": 900.0,
    "acronym": 180.0,
    "lexical": 600.0,
}
MAX_DISK_MB = 500.0


class BuildResourceError(RuntimeError):
    """Raised when sidecar build exceeds resource limits."""
    pass


def get_process_rss_gib() -> float:
    """Returns the total RSS of the current process and all its children in GiB."""
    try:
        parent = psutil.Process()
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 ** 3)
    except Exception:
        return 0.0


def check_build_watchdog(abort_rss_gib: float = 12.0):
    """Checks process tree RSS and aborts if it exceeds the hard cap."""
    rss = get_process_rss_gib()
    if rss >= abort_rss_gib:
        raise BuildResourceError(
            f"FATAL: Process RSS ({rss:.2f} GiB) exceeded hard cap of {abort_rss_gib} GiB during sidecar build!"
        )


def compute_pool_sha256(terms: List[str]) -> str:
    """Computes exact SHA-256 over newline-delimited terms without trailing newline."""
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


class Gate1SidecarManager:
    """Manages creation, caching, and loading of all Gate 1 sidecars with strict validation."""

    def __init__(self, cache_dir: str = "data/cache/canonical_pools"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.analyzer = get_terrier_analyzer()

    def _get_safe_ds(self, dataset: str) -> str:
        return dataset.lower().replace("-", "_")

    def _check_disk_footprint(self, path: str):
        """Asserts written file or directory is under MAX_DISK_MB."""
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
        elif os.path.isdir(path):
            size_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path)
                for f in files
            ) / (1024 ** 2)
        else:
            return
        if size_mb > MAX_DISK_MB:
            raise BuildResourceError(
                f"Disk footprint for {path} ({size_mb:.2f} MB) exceeded cap of {MAX_DISK_MB} MB!"
            )

    # -------------------------------------------------------------------------
    # 1. BGE Sidecar (Canonical Pool Embeddings, FP16)
    # -------------------------------------------------------------------------
    def build_or_load_bge_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        num_docs: int,
        encoder=None,
        device: str = "cpu",
        force_rebuild: bool = False,
        timeout_sec: float = DEFAULT_BUILD_CEILINGS_SEC["bge"],
    ) -> Dict[str, Any]:
        """
        Builds or loads BGE embeddings strictly aligned 1:1 with canonical pool terms in FP16.
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_bge_sidecar.pt")
        expected_sha = compute_pool_sha256(pool_terms)

        if not force_rebuild and os.path.exists(out_path):
            try:
                data = torch.load(out_path, map_location=device)
                meta = data.get("metadata", {})
                if (
                    meta.get("pool_sha256") == expected_sha
                    and meta.get("num_docs") == num_docs
                    and meta.get("analyzer_version") == ANALYZER_VERSION
                    and data.get("pool_terms") == pool_terms
                ):
                    return {
                        "pool_embeddings": data["embeddings"].to(device),
                        "display_surfaces": data["display_surfaces"],
                        "surf_to_idx": data["surf_to_idx"],
                        "metadata": meta,
                        "timing_s": 0.0,
                    }
                else:
                    print(f"[Sidecar] BGE cache invalid or outdated for {dataset}. Rebuilding...")
            except Exception as e:
                print(f"[Sidecar] Failed to load BGE cache for {dataset} ({e}). Rebuilding...")

        print(f"[Sidecar] Building Canonical Pool BGE Sidecar for {dataset} ({len(pool_terms)} terms, FP16)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Step 1: Surface recovery pass over corpus (up to 10,000 docs)
        surface_counts = {t: Counter() for t in pool_set}
        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 1000 == 0:
                check_build_watchdog()
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"BGE sidecar build exceeded time limit of {timeout_sec}s")
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

        # Step 2: Encode display surfaces in FP16
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            enc_device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=enc_device)

        if hasattr(encoder, "encode"):
            embs = encoder.encode(display_surfaces, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
            embs_tensor = torch.tensor(embs, dtype=torch.float16)
        elif hasattr(encoder, "encode_queries"):
            embs = encoder.encode_queries(display_surfaces)
            embs_tensor = torch.tensor(embs, dtype=torch.float16)
        else:
            raise ValueError(f"Unsupported encoder type: {type(encoder)}")

        embs_tensor = torch.nn.functional.normalize(embs_tensor.float(), p=2, dim=-1).half()

        # Step 3: Atomic write with provenance metadata
        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "dtype": "float16",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        tmp_path = out_path + ".tmp"
        torch.save({
            "metadata": metadata,
            "pool_terms": pool_terms,
            "display_surfaces": display_surfaces,
            "surf_to_idx": surf_to_idx,
            "embeddings": embs_tensor.cpu(),
        }, tmp_path)
        os.replace(tmp_path, out_path)
        self._check_disk_footprint(out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] BGE Sidecar built in {timing_s}s -> {out_path}")

        return {
            "pool_embeddings": embs_tensor.to(device),
            "display_surfaces": display_surfaces,
            "surf_to_idx": surf_to_idx,
            "metadata": metadata,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 2. Bounded PPMI Sidecar (Precomputed top-M=600, Exact Float)
    # -------------------------------------------------------------------------
    def build_or_load_bounded_ppmi_sidecar(
        self,
        dataset: str,
        index,
        pool_terms: List[str],
        num_docs: int,
        top_m: int = 600,
        force_rebuild: bool = False,
        timeout_sec: float = DEFAULT_BUILD_CEILINGS_SEC["ppmi"],
    ) -> Dict[str, Any]:
        """
        Builds or loads precomputed top-M=600 PPMI neighbors per anchor with exact floats.
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_bounded_ppmi.json")
        expected_sha = compute_pool_sha256(pool_terms)

        if not force_rebuild and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                if (
                    meta.get("pool_sha256") == expected_sha
                    and meta.get("num_docs") == num_docs
                    and meta.get("analyzer_version") == ANALYZER_VERSION
                    and meta.get("top_m") == top_m
                ):
                    return {
                        "anchor_ppmi": {a: [(t, float(s)) for t, s in cands] for a, cands in data["anchors"].items()},
                        "metadata": meta,
                        "timing_s": 0.0,
                    }
                else:
                    print(f"[Sidecar] PPMI cache invalid or outdated for {dataset}. Rebuilding...")
            except Exception as e:
                print(f"[Sidecar] Failed to load PPMI cache for {dataset} ({e}). Rebuilding...")

        print(f"[Sidecar] Building Bounded PPMI Sidecar for {dataset} (top_m={top_m}, exact float)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)
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

        # Streaming co-occurrence accumulation with bounded memory
        anchor_cooccur: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 2000 == 0:
                check_build_watchdog()
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"PPMI sidecar build exceeded time limit of {timeout_sec}s")

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

        # Compute exact PPMI (no rounding)
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
                            scored.append((t, float(pmi)))
            if scored:
                scored.sort(key=lambda x: (-x[1], x[0]))
                anchor_ppmi[a] = scored[:top_m]

        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "top_m": top_m,
            "num_anchors": len(anchor_ppmi),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Atomic write
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": metadata,
                "anchors": anchor_ppmi,
            }, f)
        os.replace(tmp_path, out_path)
        self._check_disk_footprint(out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Bounded PPMI Sidecar built in {timing_s}s for {len(anchor_ppmi):,} anchors -> {out_path}")

        return {
            "anchor_ppmi": anchor_ppmi,
            "metadata": metadata,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 3. Acronym Definition Rescue Sidecar (Schwartz-Hearst + Frequency)
    # -------------------------------------------------------------------------
    def build_or_load_acronym_rescue_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        num_docs: int,
        force_rebuild: bool = False,
        timeout_sec: float = DEFAULT_BUILD_CEILINGS_SEC["acronym"],
    ) -> Dict[str, Any]:
        """
        Builds or loads bidirectional Schwartz-Hearst acronym/full-form mapping to P.
        """
        safe_ds = self._get_safe_ds(dataset)
        out_path = os.path.join(self.cache_dir, f"{safe_ds}_acronym_rescue.json")
        expected_sha = compute_pool_sha256(pool_terms)

        if not force_rebuild and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                if (
                    meta.get("pool_sha256") == expected_sha
                    and meta.get("num_docs") == num_docs
                    and meta.get("analyzer_version") == ANALYZER_VERSION
                ):
                    return {
                        "acronym_to_pool": {k: [(t, float(s)) for t, s in v] for k, v in data["acronyms"].items()},
                        "metadata": meta,
                        "timing_s": 0.0,
                    }
                else:
                    print(f"[Sidecar] Acronym cache invalid or outdated for {dataset}. Rebuilding...")
            except Exception as e:
                print(f"[Sidecar] Failed to load Acronym cache for {dataset} ({e}). Rebuilding...")

        print(f"[Sidecar] Building Acronym Definition Rescue Sidecar for {dataset}...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Full Schwartz-Hearst pattern pairs
        p1 = re.compile(r'\b([A-Za-z][A-Za-z\s]{2,40}?)\s*\(([A-Z0-9]{2,8})\)')
        p2 = re.compile(r'\b([A-Z0-9]{2,8})\s*\(([A-Za-z][A-Za-z\s]{2,40}?)\)')

        def is_valid_acronym_pair(short_form: str, long_form: str) -> bool:
            """Verifies that letters in short_form match leading characters in long_form words."""
            words = [w for w in long_form.split() if w.lower() not in {"of", "the", "and", "in", "for"}]
            if not words:
                return False
            initials = "".join(w[0].upper() for w in words if w)
            return short_form.upper() in initials or initials.startswith(short_form.upper())

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 5000 == 0:
                check_build_watchdog()
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"Acronym sidecar build exceeded time limit of {timeout_sec}s")

            for m in p1.finditer(text):
                short_f = m.group(2).strip()
                long_f = m.group(1).strip().lower()
                if is_valid_acronym_pair(short_f, long_f):
                    pair_counts[(short_f, long_f)] += 1

            for m in p2.finditer(text):
                short_f = m.group(1).strip()
                long_f = m.group(2).strip().lower()
                if is_valid_acronym_pair(short_f, long_f):
                    pair_counts[(short_f, long_f)] += 1

        acronym_to_pool = defaultdict(dict)
        for (acr, full), count in pair_counts.items():
            acr_lower = acr.lower()
            acr_terms, _ = self.analyzer.analyze(acr)
            full_terms, _ = self.analyzer.analyze(full)

            # Weight incorporates base confidence and corpus occurrence log-boost
            conf_boost = min(0.15, 0.03 * math.log(1.0 + count))

            # Trigger: acronym -> candidate in P
            for at in acr_terms:
                if at in pool_set:
                    acronym_to_pool[acr_lower][at] = max(acronym_to_pool[acr_lower].get(at, 0.0), 1.0 + conf_boost)
            for ft in full_terms:
                if ft in pool_set:
                    acronym_to_pool[acr_lower][ft] = max(acronym_to_pool[acr_lower].get(ft, 0.0), 0.8 + conf_boost)

            # Trigger: full form words -> acronym candidate in P
            for ft in full_terms:
                for at in acr_terms:
                    if at in pool_set:
                        acronym_to_pool[ft][at] = max(acronym_to_pool[ft].get(at, 0.0), 0.8 + conf_boost)

        formatted = {}
        for trigger, term_scores in acronym_to_pool.items():
            s_list = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
            formatted[trigger] = s_list

        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "num_triggers": len(formatted),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Atomic write
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": metadata,
                "acronyms": formatted,
            }, f)
        os.replace(tmp_path, out_path)
        self._check_disk_footprint(out_path)

        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Acronym Rescue Sidecar built in {timing_s}s ({len(formatted)} triggers) -> {out_path}")

        return {
            "acronym_to_pool": formatted,
            "metadata": metadata,
            "timing_s": timing_s,
        }

    # -------------------------------------------------------------------------
    # 4. Sparse Lexical Context Sidecar (Passage Context BM25 Index)
    # -------------------------------------------------------------------------
    def build_or_load_sparse_lexical_sidecar(
        self,
        dataset: str,
        pool_terms: List[str],
        num_docs: int,
        force_rebuild: bool = False,
        timeout_sec: float = DEFAULT_BUILD_CEILINGS_SEC["lexical"],
    ) -> Dict[str, Any]:
        """
        Builds or loads auxiliary BM25 index over candidate terms' passage context profiles
        using deterministic reservoir sampling (up to 50 passages per candidate).
        """
        safe_ds = self._get_safe_ds(dataset)
        aux_index_dir = os.path.join(self.cache_dir, f"{safe_ds}_lexical_profiles_idx")
        prop_path = os.path.join(aux_index_dir, "data.properties")
        meta_path = os.path.join(aux_index_dir, "sidecar_metadata.json")
        expected_sha = compute_pool_sha256(pool_terms)

        if not force_rebuild and os.path.exists(prop_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("pool_sha256") == expected_sha
                    and meta.get("num_docs") == num_docs
                    and meta.get("analyzer_version") == ANALYZER_VERSION
                ):
                    retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
                    return {
                        "index_path": aux_index_dir,
                        "retriever": retriever,
                        "metadata": meta,
                        "timing_s": 0.0,
                    }
                else:
                    print(f"[Sidecar] Lexical profiles cache invalid or outdated for {dataset}. Rebuilding...")
            except Exception as e:
                print(f"[Sidecar] Failed to load Lexical profiles cache for {dataset} ({e}). Rebuilding...")

        print(f"[Sidecar] Building Sparse Lexical Context Sidecar for {dataset} ({len(pool_terms)} terms, reservoir sampling)...")
        t0 = time.perf_counter()
        pool_set = set(pool_terms)

        # Reservoir sampling: reservoir size K=50 per pool term with fixed seed
        rng = random.Random(42)
        reservoirs: Dict[str, List[str]] = defaultdict(list)
        item_counts: Dict[str, int] = defaultdict(int)

        doc_count = 0
        for _, text in BenchmarkLoader.stream_corpus(dataset):
            doc_count += 1
            if doc_count % 2000 == 0:
                check_build_watchdog()
                if time.perf_counter() - t0 > timeout_sec:
                    raise TimeoutError(f"Lexical sidecar build exceeded time limit of {timeout_sec}s")

            terms, _ = self.analyzer.analyze(text)
            doc_terms = set(terms)
            doc_pool_terms = doc_terms.intersection(pool_set)
            if not doc_pool_terms:
                continue

            tok_len = len(terms)
            for t in doc_pool_terms:
                for idx, tok in enumerate(terms):
                    if tok == t:
                        item_counts[t] += 1
                        n = item_counts[t]
                        start = max(0, idx - 32)
                        end = min(tok_len, idx + 32)
                        passage_str = " ".join([w for j, w in enumerate(terms[start:end]) if j != (idx - start)])

                        if len(reservoirs[t]) < 50:
                            reservoirs[t].append(passage_str)
                        else:
                            # Standard Algorithm R reservoir replacement
                            j = rng.randint(0, n - 1)
                            if j < 50:
                                reservoirs[t][j] = passage_str

        # Stream documents to IterDictIndexer
        def doc_generator() -> Iterator[Dict[str, str]]:
            for t in pool_terms:
                passages = reservoirs.get(t, [])
                combined_text = " ".join(passages) if passages else t
                yield {"docno": t, "text": combined_text}

        tmp_idx_dir = aux_index_dir + ".tmp"
        if os.path.exists(tmp_idx_dir):
            shutil.rmtree(tmp_idx_dir)
        os.makedirs(tmp_idx_dir, exist_ok=True)

        indexer = pt.IterDictIndexer(os.path.abspath(tmp_idx_dir), overwrite=True)
        indexer.index(doc_generator())

        # Write metadata
        metadata = {
            "dataset": dataset,
            "pool_sha256": expected_sha,
            "pool_size": len(pool_terms),
            "num_docs": num_docs,
            "analyzer_version": ANALYZER_VERSION,
            "passages_per_candidate": 50,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(tmp_idx_dir, "sidecar_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        if os.path.exists(aux_index_dir):
            shutil.rmtree(aux_index_dir)
        os.rename(tmp_idx_dir, aux_index_dir)
        self._check_disk_footprint(aux_index_dir)

        retriever = pt.terrier.Retriever(os.path.abspath(aux_index_dir), wmodel="BM25", num_results=500)
        timing_s = round(time.perf_counter() - t0, 2)
        print(f"[Sidecar] Sparse Lexical Sidecar built in {timing_s}s -> {aux_index_dir}")

        return {
            "index_path": aux_index_dir,
            "retriever": retriever,
            "metadata": metadata,
            "timing_s": timing_s,
        }
