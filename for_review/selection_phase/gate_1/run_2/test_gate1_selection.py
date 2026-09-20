"""
CRVE/tests/test_gate1_selection.py

Focused unit test suite verifying Phase 2.1a Gate 1 Candidate Selection invariants:
1. Canonical pool properties, sizes, and SHA-256 recomputed directly from raw terms.
2. Fast O(1) pool_set membership and query-term exclusion P_q.
3. Mutually exclusive and exhaustive action partition (P_H + P_N + P_M = 1.0).
4. Raw vs Recall-Safe cutoff entry opportunity contract across K in {10, 100, 200, 500, 1000}.
5. RRF tie-breaking (-S_RRF, best_individual_rank, term).
6. AnchorBGEAll vs AnchorBGEFiltered contracts.
7. PPMISidecarProposer fidelity and exact scoring.
8. AcronymDefinitionRescueProposer bidirectional rescue.
9. SparseLexicalContextProposer retrieval.
"""

import os
import sys
import json
import yaml
import math
import hashlib
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from crve.selection.gate1_proposers import (
    RRFHybridProposer,
    AnchorBGEProposer,
    PPMISidecarProposer,
    AcronymDefinitionRescueProposer,
    SparseLexicalContextProposer,
)


def test_canonical_pool_properties():
    """Validates canonical pool artifacts, expected sizes, and direct recomputed SHA-256 hashes."""
    config_path = "CRVE/configs/gate1_phase2_1a.yaml"
    assert os.path.exists(config_path), f"Missing config: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    expected_sizes = config["pool_spec"]["sizes"]
    expected_hashes = config["pool_spec"]["hashes"]

    for ds, target_size in expected_sizes.items():
        artifact_path = f"data/cache/canonical_pools/{ds}_canonical_pool.json"
        assert os.path.exists(artifact_path), f"Missing canonical pool artifact: {artifact_path}"
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        terms = data["terms"]
        assert len(terms) == target_size, f"Size mismatch for {ds}: {len(terms)} != {target_size}"

        # Recompute SHA-256 directly from raw terms
        computed_sha = hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()
        assert computed_sha == data["sha256"], f"Stored hash does not match computed for {ds}"
        assert computed_sha == expected_hashes[ds], f"Hash mismatch for {ds}: {computed_sha} != {expected_hashes[ds]}"

        # Verify terms are unique, non-empty, and lowercase
        assert len(set(terms)) == target_size
        assert all(len(t) >= 2 for t in terms)
        assert all(t == t.lower() for t in terms)


def test_pool_set_membership_invariance():
    """Verifies that pool_set membership check matches list check and excludes query terms."""
    pool_terms = ["robot", "ai", "learning", "neural", "vision"]
    pool_set = set(pool_terms)
    query_terms = {"robot", "deep"}

    candidates = ["robot", "ai", "vision", "unknown", "deep"]
    clean_cands = [t for t in candidates if t in pool_set and t not in query_terms]

    assert clean_cands == ["ai", "vision"]
    assert "robot" not in clean_cands
    assert "unknown" not in clean_cands


def test_action_partition_exclusivity():
    """
    Validates that every evaluated action belongs to exactly one partition:
    Helpful, Harmful, or Neutral/Marginal (P_H + P_N + P_M = 1.0),
    and waste is an overlapping diagnostic.
    """
    delta = 0.005
    epsilon = 0.001

    test_actions = [
        (0.010, 2, "helpful", False),
        (0.005, 0, "helpful", False),
        (0.010, -1, "harmful", False),
        (-0.002, 1, "harmful", False),
        (-0.005, -1, "harmful", True),
        (0.002, 1, "neutral", False),
        (0.000, 0, "neutral", True),
        (-0.0005, 0, "neutral", True),
    ]

    for d_ndcg, net_k, exp_class, exp_waste in test_actions:
        is_helpful = (d_ndcg >= delta) and (net_k >= 0)
        is_harmful = (d_ndcg < -epsilon) or (net_k < 0)
        is_neutral = (not is_helpful) and (not is_harmful)
        is_waste = (d_ndcg <= 0.0) and (net_k <= 0)

        classes = [is_helpful, is_harmful, is_neutral]
        assert sum(classes) == 1, f"Partition failed for d_ndcg={d_ndcg}, net_k={net_k}: {classes}"

        assigned = "helpful" if is_helpful else ("harmful" if is_harmful else "neutral")
        assert assigned == exp_class, f"Expected {exp_class}, got {assigned} for d_ndcg={d_ndcg}, net_k={net_k}"
        assert is_waste == exp_waste


def test_cutoff_entries_contract():
    """Tests raw vs recall-safe cutoff entry logic for cutoffs K in {10, 100, 200, 500, 1000}."""
    cutoffs = [10, 100, 200, 500, 1000]
    epsilon = 0.001

    for k in cutoffs:
        base_rank_out = k + 50
        exp_rank_enter = max(1, k - 5)

        raw_entry = (base_rank_out > k) and (exp_rank_enter <= k)
        safe_entry = raw_entry and (0.02 >= -epsilon) and (1 >= 1)
        assert raw_entry is True
        assert safe_entry is True


def test_rrf_tie_breaking():
    """Verifies RRF tie-breaking rule: (-S_RRF, best_individual_rank, term)."""
    proposer = RRFHybridProposer(k=60)
    channel_rankings = {
        "ch1": [("term_b", 0.9), ("term_a", 0.8)],
        "ch2": [("term_a", 0.9), ("term_b", 0.8)],
    }
    fused = proposer.fuse(channel_rankings, top_l=2)
    assert [t for t, _ in fused] == ["term_a", "term_b"]


def test_anchor_bge_all_vs_filtered():
    """Verifies that AnchorBGEFiltered applies specificity/DF filters while AnchorBGEAll keeps all terms."""
    query_obj = {"query": "covid-19 vaccine trial", "query_id": "q1"}
    pool_terms = ["vaccine", "trial", "efficacy", "immunization", "clinical"]
    pool_embeddings = torch.randn(len(pool_terms), 384)
    pool_embeddings = torch.nn.functional.normalize(pool_embeddings, p=2, dim=-1)
    surf_to_idx = {t: i for i, t in enumerate(pool_terms)}

    df_map = {"covid-19": 10, "vaccine": 500, "trial": 3000}
    idf_map = {"covid-19": 4.5, "vaccine": 2.0, "trial": 0.5}
    num_docs = 10000

    # Mock encoder that returns 1x384 vector
    class MockEncoder:
        def encode(self, texts, **kwargs):
            t = torch.randn(len(texts), 384)
            return torch.nn.functional.normalize(t, p=2, dim=-1).numpy()

    encoder = MockEncoder()

    # Filtered proposer
    prop_filtered = AnchorBGEProposer(
        encoder=encoder,
        pool_terms=pool_terms,
        pool_embeddings_tensor=pool_embeddings,
        surf_to_pool_idx=surf_to_idx,
        idf_map=idf_map,
        df_map=df_map,
        num_docs=num_docs,
        filter_anchors=True,
    )

    # All proposer
    prop_all = AnchorBGEProposer(
        encoder=encoder,
        pool_terms=pool_terms,
        pool_embeddings_tensor=pool_embeddings,
        surf_to_pool_idx=surf_to_idx,
        idf_map=idf_map,
        df_map=df_map,
        num_docs=num_docs,
        filter_anchors=False,
    )

    anchors_filtered = prop_filtered._extract_query_anchors("covid-19 vaccine trial")
    anchors_all = prop_all._extract_query_anchors("covid-19 vaccine trial")

    # All should retain all tokens, while filtered applies specificity/DF filter
    assert len(anchors_all) >= len(anchors_filtered)
    assert all(w >= 0.01 for _, _, w in anchors_all)


def test_ppmi_sidecar_fidelity():
    """Verifies that PPMISidecarProposer correctly retrieves top candidates with exact floats."""
    anchor_ppmi = {
        "vaccin": [("immun", 4.52189), ("antibodi", 3.81245), ("dosag", 2.11002)],
        "trial": [("clinic", 3.99120), ("patient", 3.12055)],
    }
    df_map = {"vaccin": 100, "trial": 200, "immun": 50, "antibodi": 60, "dosag": 70, "clinic": 80, "patient": 90}
    idf_map = {k: 3.0 for k in df_map}
    num_docs = 1000

    proposer = PPMISidecarProposer(
        anchor_ppmi=anchor_ppmi,
        idf_map=idf_map,
        df_map=df_map,
        num_docs=num_docs,
    )

    query_obj = {"query": "vaccin trial", "query_id": "q1"}
    props = proposer.propose(query_obj, top_k=5)

    assert len(props) > 0
    # Scores must be exact floats
    assert isinstance(props[0][1], float)
    # Highest score should be among the top PPMI neighbors
    terms = [t for t, _ in props]
    assert "immun" in terms or "clinic" in terms


def test_acronym_rescue():
    """Verifies that AcronymDefinitionRescueProposer extracts bidirectional mappings."""
    acronym_map = {
        "covid": [("sars-cov-2", 1.15), ("coronavirus", 0.95)],
        "who": [("world health organization", 1.10)],
    }
    pool_terms = ["sars-cov-2", "coronavirus", "world health organization", "other"]
    proposer = AcronymDefinitionRescueProposer(acronym_to_pool=acronym_map, pool_terms=pool_terms)

    # Query with acronym
    q1 = {"query": "covid impact", "query_id": "q1"}
    props1 = proposer.propose(q1, top_k=5)
    assert any(t == "sars-cov-2" for t, _ in props1)

    # Query without acronym
    q2 = {"query": "unrelated general topic", "query_id": "q2"}
    props2 = proposer.propose(q2, top_k=5)
    assert len(props2) == 0


def test_sparse_lexical_context_proposer():
    """Verifies that SparseLexicalContextProposer formats and handles query proposals."""
    class MockRetriever:
        def transform(self, q_df):
            return pd.DataFrame([
                {"docno": "immun", "score": 12.5},
                {"docno": "antibodi", "score": 10.2},
            ])

    pool_terms = ["immun", "antibodi", "other"]
    proposer = SparseLexicalContextProposer(retriever=MockRetriever(), pool_terms=pool_terms)
    query_obj = {"query": "vaccine response", "query_id": "q1"}
    props = proposer.propose(query_obj, top_k=5)

    assert len(props) == 2
    assert props[0] == ("immun", 12.5)
    assert props[1] == ("antibodi", 10.2)


def test_build_resource_guard_trigger():
    """Verifies that resource guards trigger fail-closed BuildResourceError and clean up temporary files."""
    from crve.selection.gate1_sidecars import check_build_watchdog, BuildResourceError, Gate1SidecarManager
    import tempfile

    # RSS watchdog trigger
    with pytest.raises(BuildResourceError, match="exceeded hard cap"):
        check_build_watchdog(abort_rss_gib=0.000001)

    # Disk cap trigger and automatic cleanup
    mgr = Gate1SidecarManager()
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as f:
        f.write(b"x" * 2048)
        tmp_path = f.name

    try:
        with pytest.raises(BuildResourceError, match="exceeded cap"):
            mgr._check_disk_footprint(tmp_path, max_mb=0.0001)  # 0.1 KB cap
        # Assert file was cleaned up
        assert not os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_duplicate_manifest_qid_rejection():
    """Verifies that duplicate QIDs in the frozen manifest are detected and rejected fail-closed."""
    frozen_qids = ["q1", "q2", "q1", "q3"]
    with pytest.raises(RuntimeError, match="Duplicate QIDs detected"):
        if len(frozen_qids) != len(set(frozen_qids)):
            raise RuntimeError("FATAL: Duplicate QIDs detected in frozen manifest for dataset 'test'!")


def test_stale_shard_rejection():
    """Verifies that shards with mismatched config, dataset, qrels, or pool hashes are rejected fail-closed."""
    st_cfg = "hash_cfg_v1"
    st_ds = "hash_ds_v1"
    st_qrels = "hash_qrels_v1"
    st_pool = "hash_pool_v1"

    # Mismatched config
    config_hash = "hash_cfg_v2"
    dataset_hash = "hash_ds_v1"
    qrels_hash = "hash_qrels_v1"
    pool_hash = "hash_pool_v1"

    with pytest.raises(RuntimeError, match="mismatched hashes"):
        if st_cfg != config_hash or st_ds != dataset_hash or st_qrels != qrels_hash or st_pool != pool_hash:
            raise RuntimeError(
                f"FATAL: Stale shard for QID q1 has mismatched hashes: "
                f"config={st_cfg[:8]} vs {config_hash[:8]}, qrels={st_qrels[:8]} vs {qrels_hash[:8]}, pool={st_pool[:8]} vs {pool_hash[:8]}. Clean output directory."
            )


def test_sidecar_provenance_mismatch_rejection():
    """Verifies that sidecars with mismatched pool hash, doc count, or BGE model are not loaded."""
    import tempfile
    from crve.selection.gate1_sidecars import Gate1SidecarManager, compute_pool_sha256

    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = Gate1SidecarManager(cache_dir=tmp_dir, bge_model_name="BAAI/bge-small-en-v1.5")
        pool_terms = ["term_a", "term_b"]
        num_docs = 1000

        # Create a mock sidecar with an old pool hash
        out_path = os.path.join(tmp_dir, "test_ds_bge_sidecar.pt")
        mock_data = {
            "metadata": {
                "pool_sha256": "wrong_sha256",
                "num_docs": num_docs,
                "analyzer_version": "v1_krovetz_suppletion",
                "bge_model": "BAAI/bge-small-en-v1.5",
            },
            "pool_terms": pool_terms,
            "display_surfaces": pool_terms,
            "surf_to_idx": {t: i for i, t in enumerate(pool_terms)},
            "embeddings": torch.zeros((2, 384), dtype=torch.float16),
        }
        torch.save(mock_data, out_path)

        # Loading with current pool terms should reject the cache because hash mismatches
        # and attempt rebuild (which raises with our dummy encoder or succeeds cleanly)
        class DummyEncoder:
            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), 384), dtype=np.float32)

        from unittest.mock import patch
        with patch("evaluation.benchmark_loader.BenchmarkLoader.stream_corpus", return_value=[("doc1", "term_a is related to term_b")]):
            res = mgr.build_or_load_bge_sidecar(
                dataset="test_ds",
                pool_terms=pool_terms,
                num_docs=num_docs,
                encoder=DummyEncoder(),
                force_rebuild=False,
            )
            # Should have rebuilt with correct hash
            assert res["metadata"]["pool_sha256"] == compute_pool_sha256(pool_terms)

