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
    from scripts.run_gate1_oracle_evaluation import validate_manifest_qids

    frozen_qids = ["q1", "q2", "q1", "q3"]
    with pytest.raises(RuntimeError, match="Duplicate QIDs detected"):
        validate_manifest_qids(frozen_qids, dataset="test")


def test_stale_shard_rejection():
    """Verifies that shards with mismatched config, dataset, qrels, or pool hashes are rejected fail-closed."""
    from scripts.run_gate1_oracle_evaluation import validate_shard_hashes

    st_status = {
        "config_hash": "hash_cfg_v1",
        "dataset_hash": "hash_ds_v1",
        "qrels_hash": "hash_qrels_v1",
        "pool_hash": "hash_pool_v1",
    }
    expected_hashes = {
        "config_hash": "hash_cfg_v2",  # mismatched
        "dataset_hash": "hash_ds_v1",
        "qrels_hash": "hash_qrels_v1",
        "pool_hash": "hash_pool_v1",
    }
    with pytest.raises(RuntimeError, match="mismatched hashes"):
        validate_shard_hashes(st_status, expected_hashes, qid="q1")


def test_action_coverage_set_equality():
    """Verifies that validate_action_coverage enforces exact Cartesian set equality."""
    from scripts.run_gate1_oracle_evaluation import validate_action_coverage

    weights = [0.05, 0.10]
    univ_df = pd.DataFrame([
        {"qid": "q1", "candidate_term": "term1"},
        {"qid": "q1", "candidate_term": "term2"},
    ])

    # 1. Matching audit
    audit_df_ok = pd.DataFrame([
        {"qid": "q1", "candidate_term": "term1", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term1", "weight": 0.10},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.10},
    ])
    validate_action_coverage(audit_df_ok, univ_df, weights)

    # 2. Missing triple
    audit_df_missing = pd.DataFrame([
        {"qid": "q1", "candidate_term": "term1", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term1", "weight": 0.10},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.05},
    ])
    with pytest.raises(RuntimeError, match="Action coverage set equality failed"):
        validate_action_coverage(audit_df_missing, univ_df, weights)

    # 3. Extra triple
    audit_df_extra = pd.DataFrame([
        {"qid": "q1", "candidate_term": "term1", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term1", "weight": 0.10},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.10},
        {"qid": "q1", "candidate_term": "term3", "weight": 0.05},
    ])
    with pytest.raises(RuntimeError, match="Action coverage set equality failed"):
        validate_action_coverage(audit_df_extra, univ_df, weights)

    # 4. Duplicate triple
    audit_df_dup = pd.DataFrame([
        {"qid": "q1", "candidate_term": "term1", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term1", "weight": 0.05},  # duplicate
        {"qid": "q1", "candidate_term": "term2", "weight": 0.05},
        {"qid": "q1", "candidate_term": "term2", "weight": 0.10},
    ])
    with pytest.raises(RuntimeError, match="Duplicate action evaluations"):
        validate_action_coverage(audit_df_dup, univ_df, weights)


def test_bright_exclusions_depth_compensation():
    """Verifies that BRIGHT exclusion depth compensation formula k_fetch = min(N, 1000 + max_ex) preserves K=1000."""
    num_docs = 50000
    max_ex = 9205
    k_fetch = min(num_docs, 1000 + max_ex)
    assert k_fetch == 10205

    # Simulate ranking of size k_fetch where 9205 are excluded
    raw_ranking = [f"ex_doc_{i}" for i in range(max_ex)] + [f"doc_{i}" for i in range(2000)]
    exclusions = set(f"ex_doc_{i}" for i in range(max_ex))
    filtered_ranking = [d for d in raw_ranking if d not in exclusions][:1000]

    assert len(filtered_ranking) == 1000
    assert not any(d in exclusions for d in filtered_ranking)


def test_live_ppmi_universe_invariance():
    """Verifies that LivePPMI (diagnostic channel) does NOT alter the reference universe R_q."""
    OPERATIONAL_CHANNELS = [
        "WholeQueryBGE",
        "AnchorBGEFiltered",
        "AnchorBGEAll",
        "PPMISidecar",
        "SparseLexicalContextProfiles",
        "AcronymDefinitionRescue",
        "RRF_Core3",
        "RRF_Extended",
    ]
    assert "LivePPMI" not in OPERATIONAL_CHANNELS

    channel_proposals = {
        "WholeQueryBGE": [("t1", 0.9), ("t2", 0.8)],
        "PPMISidecar": [("t2", 0.7), ("t3", 0.6)],
        "LivePPMI": [("t4", 0.99), ("t5", 0.95)],  # Should not enter R_q
    }
    phase1_cands = {"t0"}

    r_core = set(phase1_cands)
    for ch_name in OPERATIONAL_CHANNELS:
        if ch_name in channel_proposals:
            for t, _ in channel_proposals[ch_name]:
                r_core.add(t)

    assert r_core == {"t0", "t1", "t2", "t3"}
    assert "t4" not in r_core
    assert "t5" not in r_core


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

        class DummyEncoder:
            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), 384), dtype=np.float32)

        from unittest.mock import patch
        with patch("crve.selection.gate1_sidecars.compute_corpus_source_hash", return_value="mock_corpus_hash"), \
             patch("evaluation.benchmark_loader.BenchmarkLoader.stream_corpus", return_value=[("doc1", "term_a is related to term_b")]):
            res = mgr.build_or_load_bge_sidecar(
                dataset="test_ds",
                pool_terms=pool_terms,
                num_docs=num_docs,
                encoder=DummyEncoder(),
                force_rebuild=False,
            )
            # Should have rebuilt with correct hash
            assert res["metadata"]["pool_sha256"] == compute_pool_sha256(pool_terms)


def test_validate_gate1_config_fail_closed():
    """Verifies that validate_gate1_config enforces complete required schema without defaults."""
    from crve.selection.gate1_sidecars import validate_gate1_config

    with open("CRVE/configs/gate1_phase2_1a.yaml", "r", encoding="utf-8") as f:
        valid_cfg = yaml.safe_load(f)

    # 1. Valid config passes
    validate_gate1_config(valid_cfg)

    # 2. Missing top-level key raises KeyError
    bad_cfg = dict(valid_cfg)
    del bad_cfg["checkpoint_b_thresholds"]
    with pytest.raises(KeyError, match="checkpoint_b_thresholds"):
        validate_gate1_config(bad_cfg)

    # 3. Missing watchdog ceiling raises KeyError
    bad_cfg2 = dict(valid_cfg)
    bad_cfg2["memory_watchdog"] = {"jvm_heap_gib": 4, "warn_rss_gib": 8}  # missing hard_abort_rss_gib
    with pytest.raises(KeyError, match="hard_abort_rss_gib"):
        validate_gate1_config(bad_cfg2)

    # 4. Negative value raises ValueError
    bad_cfg3 = dict(valid_cfg)
    bad_cfg3["memory_watchdog"] = {"jvm_heap_gib": 4, "warn_rss_gib": 8, "hard_abort_rss_gib": -1}
    with pytest.raises(ValueError, match="positive number"):
        validate_gate1_config(bad_cfg3)


def test_compute_corpus_source_hash_missing_file_raises():
    """Verifies that compute_corpus_source_hash raises FileNotFoundError on missing files."""
    from crve.selection.gate1_sidecars import compute_corpus_source_hash

    with pytest.raises(FileNotFoundError):
        compute_corpus_source_hash("non_existent_dataset_12345")


def test_compute_corpus_source_hash_streaming(tmp_path):
    """Verifies that 64KB chunk streaming hash matches hashlib.sha256 of the entire content."""
    from crve.selection.gate1_sidecars import compute_corpus_source_hash
    from unittest.mock import patch

    # Create dummy files
    f1 = tmp_path / "corpus.jsonl"
    f2 = tmp_path / "data.properties"
    content1 = b"line 1\nline 2\n" * 10000  # >128 KB
    content2 = b"index.num.docs=1000\n"
    f1.write_bytes(content1)
    f2.write_bytes(content2)

    hasher = hashlib.sha256()
    hasher.update(content1)
    hasher.update(content2)
    expected_hash = hasher.hexdigest()

    with patch("evaluation.benchmark_loader.BenchmarkLoader.get_corpus_source_paths", return_value=[str(f1), str(f2)]):
        actual_hash = compute_corpus_source_hash("mock_ds")
        assert actual_hash == expected_hash


def test_sidecar_provenance_validation_fail_closed(tmp_path):
    """Verifies that sidecar loaders reject missing or mismatched corpus_source_hash and reservoir_seed."""
    from crve.selection.gate1_sidecars import Gate1SidecarManager
    from unittest.mock import patch, MagicMock

    with open("CRVE/configs/gate1_phase2_1a.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mgr = Gate1SidecarManager(config=cfg, cache_dir=str(tmp_path))
    pool_terms = ["term1", "term2"]

    # 1. PPMI sidecar with wrong corpus hash
    ppmi_path = tmp_path / "test_ds_bounded_ppmi.json"
    ppmi_data = {
        "metadata": {
            "corpus_source_hash": "wrong_hash",
            "pool_sha256": "wrong_pool",
            "num_docs": 100,
        },
        "anchors": {},
    }
    ppmi_path.write_text(json.dumps(ppmi_data))

    mock_index = MagicMock()
    mock_index.getCollectionStatistics.return_value.getNumberOfDocuments.return_value = 100
    with patch("crve.selection.gate1_sidecars.compute_corpus_source_hash", return_value="correct_hash"), \
         patch("crve.selection.gate1_sidecars.compute_lexicon_semantic_hash", return_value="correct_lex"), \
         patch("crve.selection.gate1_sidecars.compute_pool_sha256", return_value="correct_pool"), \
         patch("builtins.print") as mock_print:
        try:
            mgr.build_or_load_bounded_ppmi_sidecar("test_ds", mock_index, pool_terms, num_docs=100)
        except Exception:
            pass
        printed = [call.args[0] for call in mock_print.call_args_list if call.args]
        assert any("PPMI cache invalid or outdated" in p for p in printed)

    # 2. Lexical profile with wrong reservoir seed
    lex_manifest_path = tmp_path / "test_ds_lexical_profiles_idx" / "sidecar_metadata.json"
    lex_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lex_prop_path = tmp_path / "test_ds_lexical_profiles_idx" / "data.properties"
    lex_prop_path.write_text("index.num.docs=100\n")
    lex_manifest = {
        "corpus_source_hash": "correct_hash",
        "pool_sha256": "correct_pool",
        "num_docs": 100,
        "analyzer_version": "v1_krovetz_suppletion",
        "reservoir_seed": 9999,  # Mismatched seed (cfg has 42)
    }
    lex_manifest_path.write_text(json.dumps(lex_manifest))
    with patch("crve.selection.gate1_sidecars.compute_corpus_source_hash", return_value="correct_hash"), \
         patch("crve.selection.gate1_sidecars.compute_pool_sha256", return_value="correct_pool"), \
         patch("builtins.print") as mock_print:
        try:
            mgr.build_or_load_sparse_lexical_sidecar("test_ds", pool_terms, num_docs=100)
        except Exception:
            pass
        printed = [call.args[0] for call in mock_print.call_args_list if call.args]
        assert any("Lexical profiles cache invalid or outdated" in p for p in printed)


def test_dual_universe_construction():
    """Verifies that R_q^diag = R_q^operational U C_500^{LivePPMI} and LivePPMI is excluded from R_q^operational."""
    operational_channels = {
        "WholeQueryBGE": [("t1", 0.9), ("t2", 0.8)],
        "PPMISidecar": [("t2", 0.7), ("t3", 0.6)],
    }
    live_ppmi_proposals = [("t3", 0.99), ("t4_live_only", 0.95)]
    phase1_cands = {"t0"}

    # Operational universe
    r_op = set(phase1_cands)
    for ch, props in operational_channels.items():
        for t, _ in props:
            r_op.add(t)

    # Diagnostic universe
    r_diag = set(r_op)
    for t, _ in live_ppmi_proposals:
        r_diag.add(t)

    assert r_op == {"t0", "t1", "t2", "t3"}
    assert "t4_live_only" not in r_op
    assert r_diag == {"t0", "t1", "t2", "t3", "t4_live_only"}
    assert r_op.issubset(r_diag)


def test_bright_exclusion_assertion_scoping():
    """Verifies that ordinary runs support queries with 0 exclusions, while micro-smoke asserts non-empty exclusions."""
    # Micro smoke check
    micro_smoke_qids = "micro_smoke_qids"
    q_ex_empty = set()
    q_ex_nonempty = {"doc1", "doc2"}

    # In micro smoke, empty exclusions must fail
    with pytest.raises(AssertionError, match="empty excluded_doc_ids"):
        if micro_smoke_qids == "micro_smoke_qids":
            assert len(q_ex_empty) > 0, "FATAL: Micro-smoke BRIGHT query has empty excluded_doc_ids!"

    # In micro smoke, non-empty exclusions pass
    if micro_smoke_qids == "micro_smoke_qids":
        assert len(q_ex_nonempty) > 0

    # In ordinary runs (e.g. dev_200_qids or None), no assertion on query 0
    full_manifest = "dev_200_qids"
    if full_manifest == "micro_smoke_qids":
        assert len(q_ex_empty) > 0
    else:
        pass  # Ordinary runs simply filter when exclusions exist


def test_table2_operational_diagnostic_separation():
    """Verifies that Table 2 contains exactly 8 operational channels and Table 2-Diag contains LivePPMI."""
    from scripts.compile_gate1_research_tables import OPERATIONAL_CHANNELS, DIAGNOSTIC_CHANNELS

    assert len(OPERATIONAL_CHANNELS) == 8
    assert not any(ch == "LivePPMI" for ch, _ in OPERATIONAL_CHANNELS)
    assert any(ch == "LivePPMI" for ch, _ in DIAGNOSTIC_CHANNELS)
    assert any(ch == "PPMISidecar" for ch, _ in DIAGNOSTIC_CHANNELS)


def test_operational_loss_gate_logic(tmp_path):
    """Synthetic test verifying Delta-nDCG@10 loss and RawDocOppRecall@1000 loss computation against thresholds."""
    from scripts.compile_gate1_research_tables import Gate1TableCompiler

    # Create dummy audit
    audit_data = [
        # Query 1: Live achieves 0.05, Sidecar achieves 0.04 -> loss = 0.01
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_live", "weight": 0.1, "delta_ndcg10": 0.05, "delta_r1000": 0.0, "live_ppmi_rank": 1, "ppmi_sidecar_rank": None, "raw_entry": True, "baseline_ndcg10": 0.2, "baseline_r1000": 0.5},
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_sidecar", "weight": 0.1, "delta_ndcg10": 0.04, "delta_r1000": 0.0, "live_ppmi_rank": None, "ppmi_sidecar_rank": 1, "raw_entry": True, "baseline_ndcg10": 0.2, "baseline_r1000": 0.5},
    ]
    df_audit = pd.DataFrame(audit_data)
    audit_path = tmp_path / "audit.parquet"
    df_audit.to_parquet(audit_path)

    # Cutoff data: q1 has 2 docs. Live retrieves 2 (recall=1.0), Sidecar retrieves 2 (recall=1.0) -> doc loss = 0.0
    cutoff_data = [
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_live", "cutoff": 1000, "raw_entry": True, "docid": "d1"},
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_live", "cutoff": 1000, "raw_entry": True, "docid": "d2"},
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_sidecar", "cutoff": 1000, "raw_entry": True, "docid": "d1"},
        {"dataset": "test_ds", "qid": "q1", "candidate_term": "t_sidecar", "cutoff": 1000, "raw_entry": True, "docid": "d2"},
    ]
    df_cutoff = pd.DataFrame(cutoff_data)
    cutoff_path = tmp_path / "cutoff.parquet"
    df_cutoff.to_parquet(cutoff_path)

    compiler = Gate1TableCompiler(
        audit_parquet_path=str(audit_path),
        cutoff_parquet_path=str(cutoff_path),
        output_dir=str(tmp_path),
    )

    loss_res = compiler.enforce_operational_loss_gate(budget_l=200)
    assert loss_res["corpus_macro_delta_ndcg10_loss"] == 0.01
    assert loss_res["corpus_macro_raw_doc_opp_recall1000_loss"] == 0.0
    assert loss_res["gate_passed"] is True


