import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from run_gate1_oracle_evaluation import (
    assemble_shards_to_master,
    verify_probe_shards_for_continuation,
    get_shard_paths,
    compute_input_hashes,
    load_canonical_pool_terms,
    compute_pool_sha256,
)
from compile_gate1_research_tables import Gate1TableCompiler


def test_gate1_table_compiler_constructor_signature(tmp_path):
    """Verifies that Gate1TableCompiler instantiates with the exact keyword arguments used in staged runner."""
    audit_file = str(tmp_path / "synthetic_audit.parquet")
    cutoff_file = str(tmp_path / "synthetic_cutoff.parquet")
    universe_file = str(tmp_path / "synthetic_universe.parquet")
    diag_universe_file = str(tmp_path / "synthetic_diag_universe.parquet")
    out_dir = str(tmp_path / "compiler_out")

    # Create dummy parquet files with minimal required schemas
    df_audit = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
        "weight": 0.1,
        "delta_ndcg10": 0.01,
        "live_ppmi_rank": 1.0,
        "ppmi_sidecar_rank": 1.0,
    }])
    df_audit.to_parquet(audit_file, index=False)

    df_cutoff = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
        "cutoff": 1000,
        "docid": "d1",
        "raw_entry": True,
        "recall_safe_entry": True,
    }])
    df_cutoff.to_parquet(cutoff_file, index=False)

    df_universe = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
    }])
    df_universe.to_parquet(universe_file, index=False)
    df_universe.to_parquet(diag_universe_file, index=False)

    # Must instantiate cleanly without TypeError
    compiler = Gate1TableCompiler(
        audit_parquet_path=audit_file,
        cutoff_parquet_path=cutoff_file,
        output_dir=out_dir,
        universe_parquet_path=universe_file,
        diag_universe_parquet_path=diag_universe_file,
        frozen_config_path=None,
    )
    assert compiler.audit_path == audit_file
    assert compiler.cutoff_path == cutoff_file
    assert compiler.universe_path == universe_file
    assert compiler.diag_universe_path == diag_universe_file


def test_mixed_shard_assembly_with_index_reset_and_cutoff_filtering(tmp_path):
    """
    Tests synthetic mixed-shard assembly:
    - Shard 1 (Probe query 'q1'): Has diagnostic channel 'live_ppmi_rank', extra candidate 'term_diag' (not in R_q^op),
      and non-contiguous index after filtering.
    - Shard 2 (Operational query 'q2'): Has only operational terms 'term_op1', 'term_op2', no live_ppmi_rank.
    - Cutoff Shard 1 (Probe query 'q1'): Has entries for 'term_op1' and 'term_diag'.
    - Cutoff Shard 2 (Operational query 'q2'): Has entries for 'term_op1'.
    - Reference Universe: Only contains (q1, term_op1), (q2, term_op1), (q2, term_op2).
    """
    shards_dir = str(tmp_path / "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # 1. Create Shard 1 (Probe query q1)
    df_p1 = pd.DataFrame([
        {
            "dataset": "scifact",
            "qid": "q1",
            "candidate_term": "term_op1",
            "weight": 0.1,
            "delta_ndcg10": 0.02,
            "live_ppmi_rank": 1.0,
            "ppmi_sidecar_rank": 2.0,
        },
        {
            "dataset": "scifact",
            "qid": "q1",
            "candidate_term": "term_diag",  # Extra term outside operational universe
            "weight": 0.1,
            "delta_ndcg10": 0.01,
            "live_ppmi_rank": 2.0,
            "ppmi_sidecar_rank": np.nan,
        },
    ])
    p1_path = os.path.join(shards_dir, "scifact_q1.parquet")
    df_p1.to_parquet(p1_path, index=False)

    # 2. Create Shard 2 (Operational query q2)
    df_p2 = pd.DataFrame([
        {
            "dataset": "scifact",
            "qid": "q2",
            "candidate_term": "term_op1",
            "weight": 0.1,
            "delta_ndcg10": 0.05,
            "live_ppmi_rank": np.nan,
            "ppmi_sidecar_rank": 1.0,
        },
        {
            "dataset": "scifact",
            "qid": "q2",
            "candidate_term": "term_op2",
            "weight": 0.1,
            "delta_ndcg10": -0.01,
            "live_ppmi_rank": np.nan,
            "ppmi_sidecar_rank": 3.0,
        },
    ])
    p2_path = os.path.join(shards_dir, "scifact_q2.parquet")
    df_p2.to_parquet(p2_path, index=False)

    # 3. Create Cutoff Shards
    df_c1 = pd.DataFrame([
        {"dataset": "scifact", "qid": "q1", "candidate_term": "term_op1", "cutoff": 1000, "docid": "d1", "raw_entry": True, "recall_safe_entry": True},
        {"dataset": "scifact", "qid": "q1", "candidate_term": "term_diag", "cutoff": 1000, "docid": "d2", "raw_entry": True, "recall_safe_entry": True},
    ])
    c1_path = os.path.join(shards_dir, "scifact_q1_cutoff_entries.parquet")
    df_c1.to_parquet(c1_path, index=False)

    df_c2 = pd.DataFrame([
        {"dataset": "scifact", "qid": "q2", "candidate_term": "term_op1", "cutoff": 1000, "docid": "d3", "raw_entry": True, "recall_safe_entry": True},
    ])
    c2_path = os.path.join(shards_dir, "scifact_q2_cutoff_entries.parquet")
    df_c2.to_parquet(c2_path, index=False)

    # 4. Create Universe Shards
    df_u1 = pd.DataFrame([{"dataset": "scifact", "qid": "q1", "candidate_term": "term_op1"}])
    df_u2 = pd.DataFrame([
        {"dataset": "scifact", "qid": "q2", "candidate_term": "term_op1"},
        {"dataset": "scifact", "qid": "q2", "candidate_term": "term_op2"},
    ])
    u1_path = os.path.join(shards_dir, "scifact_q1_universe.parquet")
    u2_path = os.path.join(shards_dir, "scifact_q2_universe.parquet")
    df_u1.to_parquet(u1_path, index=False)
    df_u2.to_parquet(u2_path, index=False)

    # Assemble Master Universe
    master_univ_path = str(tmp_path / "reference_universe.parquet")
    assemble_shards_to_master(shards_dir, "universe", master_univ_path, expected_qids={"q1", "q2"})
    df_master_univ = pd.read_parquet(master_univ_path)
    assert len(df_master_univ) == 3

    # Assemble Master Audit with filtering & clear_diagnostic_cols=True
    master_audit_path = str(tmp_path / "gate1_candidate_audit.parquet")
    assemble_shards_to_master(
        shards_dir,
        "audit",
        master_audit_path,
        expected_qids={"q1", "q2"},
        filter_to_universe_df=df_master_univ,
        clear_diagnostic_cols=True,
    )
    tbl_master_audit = pq.read_table(master_audit_path)
    df_master_audit = tbl_master_audit.to_pandas()

    # Check 1: No index column (e.g. __index_level_0__) in Arrow schema
    assert "__index_level_0__" not in tbl_master_audit.column_names
    assert "__index_level_0__" not in df_master_audit.columns

    # Check 2: Filtered out term_diag for q1 (only term_op1 kept for q1)
    assert len(df_master_audit) == 3
    q1_terms = df_master_audit[df_master_audit["qid"] == "q1"]["candidate_term"].tolist()
    assert q1_terms == ["term_op1"]

    # Check 3: live_ppmi_rank cleared to NaN for all rows
    assert df_master_audit["live_ppmi_rank"].isna().all()

    # Assemble Master Cutoff with filter_to_universe_df
    master_cutoff_path = str(tmp_path / "gate1_cutoff_entries.parquet")
    assemble_shards_to_master(
        shards_dir,
        "cutoff",
        master_cutoff_path,
        filter_to_universe_df=df_master_univ,
    )
    tbl_master_cutoff = pq.read_table(master_cutoff_path)
    df_master_cutoff = tbl_master_cutoff.to_pandas()

    # Check 4: No index column in cutoff
    assert "__index_level_0__" not in tbl_master_cutoff.column_names

    # Check 5: Cutoff filtered: term_diag is excluded from master cutoff
    assert len(df_master_cutoff) == 2
    assert "term_diag" not in df_master_cutoff["candidate_term"].values
    assert set(df_master_cutoff["candidate_term"].unique()) == {"term_op1"}


def test_continuation_preflight_missing_probe_shard_and_zero_row_cutoff(tmp_path):
    """
    Tests continuation preflight:
    1. Satisfies 10 QIDs per dataset requirement.
    2. Verifies zero-row cutoff shard support when status num_cutoff_entries == 0.
    3. Aborts with RuntimeError when a probe shard is missing.
    4. Aborts with RuntimeError when a shard row count mismatches status.
    """
    shards_dir = str(tmp_path / "shards")
    os.makedirs(shards_dir, exist_ok=True)
    dataset = "scifact"
    qids = [f"q{i}" for i in range(10)]
    q_manifest = {"probe_40_qids": {dataset: qids}}
    config_hash = "test_config_hash_12345"

    dataset_hash, qrels_hash, _ = compute_input_hashes(dataset)
    pool_terms = load_canonical_pool_terms(dataset)
    pool_hash = compute_pool_sha256(pool_terms)

    for qid in qids:
        p_path, c_path, s_path, u_path, d_path = get_shard_paths(shards_dir, dataset, qid)

        # Status: q0 has 0 cutoff entries (tests zero-row cutoff support), others have 1
        num_c = 0 if qid == "q0" else 1
        st_df = pd.DataFrame([{
            "dataset": dataset,
            "qid": qid,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "qrels_hash": qrels_hash,
            "pool_hash": pool_hash,
            "num_variants": 1,
            "num_cutoff_entries": num_c,
            "reference_universe_size": 1,
            "diagnostic_universe_size": 1,
        }])
        st_df.to_parquet(s_path, index=False)

        # Parent shard (1 row)
        p_df = pd.DataFrame([{"dataset": dataset, "qid": qid, "candidate_term": "t1", "weight": 0.1, "delta_ndcg10": 0.01}])
        p_df.to_parquet(p_path, index=False)

        # Cutoff shard (0 rows for q0, 1 row for others)
        if num_c == 0:
            c_df = pd.DataFrame(columns=["dataset", "qid", "candidate_term", "cutoff", "docid", "raw_entry", "recall_safe_entry"])
        else:
            c_df = pd.DataFrame([{"dataset": dataset, "qid": qid, "candidate_term": "t1", "cutoff": 1000, "docid": "d1", "raw_entry": True, "recall_safe_entry": True}])
        c_df.to_parquet(c_path, index=False)

        # Universe and diag_universe (1 row each)
        u_df = pd.DataFrame([{"dataset": dataset, "qid": qid, "candidate_term": "t1"}])
        u_df.to_parquet(u_path, index=False)
        u_df.to_parquet(d_path, index=False)

    # 1. All 10 shards present, q0 has 0-row cutoff -> Preflight MUST PASS
    verify_probe_shards_for_continuation(shards_dir, [dataset], q_manifest, config_hash)

    # 2. Omit cutoff shard for q9 -> Preflight MUST ABORT with RuntimeError
    _, c9_path, _, _, _ = get_shard_paths(shards_dir, dataset, "q9")
    os.remove(c9_path)
    with pytest.raises(RuntimeError, match="Missing cutoff shard for scifact QID q9"):
        verify_probe_shards_for_continuation(shards_dir, [dataset], q_manifest, config_hash)

    # Recreate c9_path to test row count mismatch
    c_df.to_parquet(c9_path, index=False)

    # 3. Row count mismatch: q0 cutoff has 1 row but status says 0 -> Preflight MUST ABORT
    _, c0_path, _, _, _ = get_shard_paths(shards_dir, dataset, "q0")
    c_df.to_parquet(c0_path, index=False)
    with pytest.raises(RuntimeError, match="Cutoff shard row count mismatch for scifact QID q0: 1 != 0"):
        verify_probe_shards_for_continuation(shards_dir, [dataset], q_manifest, config_hash)


def test_compiler_exploratory_report_with_empty_live_ppmi_and_failed_gate(tmp_path):
    """
    Tests Gate1TableCompiler exploratory report generation:
    1. Operational audit has live_ppmi_rank empty/all-null.
    2. Sources Checkpoint B from preserved artifact.
    3. Injects dynamic exploratory warning banner with exact losses.
    4. Enforces strict fail-closed requirements on manifest and artifact.
    """
    out_dir = str(tmp_path / "compiler_out")
    os.makedirs(out_dir, exist_ok=True)

    audit_file = str(tmp_path / "operational_audit.parquet")
    cutoff_file = str(tmp_path / "operational_cutoff.parquet")
    universe_file = str(tmp_path / "operational_universe.parquet")
    manifest_file = str(tmp_path / "run_manifest.json")
    checkpoint_b_file = str(tmp_path / "checkpoint_b_loss_gate.json")

    # Operational audit: live_ppmi_rank is null/empty
    df_audit = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
        "weight": 0.1,
        "delta_ndcg10": 0.01,
        "live_ppmi_rank": np.nan,
        "ppmi_sidecar_rank": 1.0,
        "WholeQueryBGE_rank": 1.0,
        "AnchorBGEFiltered_rank": 1.0,
        "AnchorBGEAll_rank": 1.0,
        "SparseLexicalContextProfiles_rank": 1.0,
        "AcronymDefinitionRescue_rank": 1.0,
        "RRF_Core3_rank": 1.0,
        "RRF_Extended_rank": 1.0,
    }])
    df_audit.to_parquet(audit_file, index=False)

    df_cutoff = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
        "cutoff": 1000,
        "docid": "d1",
        "raw_entry": True,
        "recall_safe_entry": True,
    }])
    df_cutoff.to_parquet(cutoff_file, index=False)

    df_universe = pd.DataFrame([{
        "dataset": "scifact",
        "qid": "q1",
        "candidate_term": "term1",
    }])
    df_universe.to_parquet(universe_file, index=False)

    # Preserved Checkpoint B loss gate artifact with gate_passed: False
    checkpoint_b_data = {
        "budget_l": 200,
        "corpus_macro_delta_ndcg10_loss": 0.0072,
        "corpus_macro_raw_doc_opp_recall1000_loss": 0.0186,
        "max_oracle_loss_corpus_macro_threshold": 0.02,
        "max_doc_opp_recall_loss_corpus_macro_threshold": 0.02,
        "per_corpus_results": {
            "trec_covid": {
                "queries": 10,
                "mean_delta_ndcg10_loss": 0.0270,
                "mean_raw_doc_opp_recall1000_loss": 0.0742,
                "ndcg_loss_passed": False,
                "doc_loss_passed": False,
            }
        },
        "gate_passed": False,
    }
    with open(checkpoint_b_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_b_data, f, indent=2)

    # Valid exploratory manifest
    manifest_data = {
        "git_commit": "abc1234",
        "git_dirty": False,
        "config_hash": "cfg_test_hash",
        "weights": [0.1],
        "gate_passed": False,
        "is_exploratory_run": True,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 1. Compile with allow_exploratory=True -> MUST SUCCEED and include exploratory banner
    compiler = Gate1TableCompiler(
        audit_parquet_path=audit_file,
        cutoff_parquet_path=cutoff_file,
        output_dir=out_dir,
        universe_parquet_path=universe_file,
        run_manifest_path=manifest_file,
        checkpoint_b_path=checkpoint_b_file,
        archived_checkpoint_path=checkpoint_b_file,
        allow_exploratory=True,
    )
    report_path = compiler.compile_all_tables_and_report()
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "EXPLORATORY PROTOCOL AMENDMENT" in report_text
    assert "**Checkpoint B Status:** **FAILED**" in report_text
    assert "0.027" in report_text
    assert "0.0072" in report_text

    # 2. Fail-closed: allow_exploratory=False -> MUST RAISE RuntimeError
    compiler_strict = Gate1TableCompiler(
        audit_parquet_path=audit_file,
        cutoff_parquet_path=cutoff_file,
        output_dir=out_dir,
        universe_parquet_path=universe_file,
        run_manifest_path=manifest_file,
        checkpoint_b_path=checkpoint_b_file,
        archived_checkpoint_path=checkpoint_b_file,
        allow_exploratory=False,
    )
    with pytest.raises(RuntimeError, match="Checkpoint B operational-loss gate failed"):
        compiler_strict.compile_all_tables_and_report()

    # 3. Fail-closed: manifest claims gate_passed=True while artifact says False -> MUST RAISE ValueError
    manifest_data["gate_passed"] = True
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    compiler_inconsistent = Gate1TableCompiler(
        audit_parquet_path=audit_file,
        cutoff_parquet_path=cutoff_file,
        output_dir=out_dir,
        universe_parquet_path=universe_file,
        run_manifest_path=manifest_file,
        checkpoint_b_path=checkpoint_b_file,
        archived_checkpoint_path=checkpoint_b_file,
        allow_exploratory=True,
    )
    with pytest.raises(ValueError, match="Exploratory compilation requires run_manifest 'gate_passed' to be False"):
        compiler_inconsistent.compile_all_tables_and_report()


def test_sidecar_disallow_rebuild_fail_closed(tmp_path):
    """Verifies that Gate1SidecarManager(disallow_rebuild=True) strictly fails closed on missing/invalid caches and implicit writes."""
    from crve.selection.gate1_sidecars import Gate1SidecarManager, ANALYZER_VERSION, compute_pool_sha256, compute_corpus_source_hash

    cache_dir = str(tmp_path / "sidecar_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mgr = Gate1SidecarManager(cache_dir=cache_dir, disallow_rebuild=True)

    pool_terms = ["cancer", "cell"]
    expected_sha = compute_pool_sha256(pool_terms)
    corpus_hash = compute_corpus_source_hash("scifact")

    # 1. BGE Sidecar missing cache -> RuntimeError
    with pytest.raises(RuntimeError, match="BGE sidecar missing .* Auto-rebuild is disallowed"):
        mgr.build_or_load_bge_sidecar("scifact", pool_terms, num_docs=10)

    # 2. PPMI Sidecar: legacy JSON exists but Parquet missing -> RuntimeError (blocks implicit conversion)
    class MockLexEntry:
        def getKey(self): return "cancer"
        def getValue(self):
            class Val:
                def getDocumentFrequency(self): return 5
                def getFrequency(self): return 10
            return Val()

    class MockIndex:
        def getLexicon(self): return [MockLexEntry()]

    legacy_json_path = os.path.join(cache_dir, "scifact_ppmi_sidecar.json")
    with open(legacy_json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "anchors": {}}, f)
    with pytest.raises(RuntimeError, match="Parquet PPMI sidecar missing .* Auto-rebuild and legacy JSON conversion are disallowed"):
        mgr.build_or_load_bounded_ppmi_sidecar("scifact", index=MockIndex(), pool_terms=pool_terms, num_docs=10)

    # 3. Lexical Sidecar: Parquet exists but auxiliary index missing -> RuntimeError (blocks implicit index build)
    lex_parquet_path = os.path.join(cache_dir, "scifact_lexical_profiles.parquet")
    meta = {
        "pool_sha256": expected_sha,
        "num_docs": 10,
        "analyzer_version": ANALYZER_VERSION,
        "corpus_source_hash": corpus_hash,
        "reservoir_seed": 42,
    }
    table = pa.Table.from_pydict({"docno": ["d1"], "text": ["cancer cell"]})
    custom_meta = {b"sidecar_metadata": json.dumps(meta).encode("utf-8")}
    table = table.replace_schema_metadata(custom_meta)
    pq.write_table(table, lex_parquet_path)

    with pytest.raises(RuntimeError, match="Lexical profiles auxiliary index missing .* Auto-rebuild is disallowed"):
        mgr.build_or_load_sparse_lexical_sidecar("scifact", pool_terms=pool_terms, num_docs=10)

    # 4. Acronym Sidecar missing cache -> RuntimeError
    with pytest.raises(RuntimeError, match="Acronym sidecar missing .* Auto-rebuild is disallowed"):
        mgr.build_or_load_acronym_rescue_sidecar("scifact", pool_terms=pool_terms, num_docs=10)


def test_pre_stage2_archive_comparison(tmp_path):
    """Verifies the pre-Stage 2 archive comparison logic under exploratory continuation mode."""
    archived_gate_path = str(tmp_path / "stage1_halt" / "checkpoint_b_loss_gate.json")
    fresh_gate_data = {
        "gate_passed": False,
        "corpus_macro_delta_ndcg10_loss": 0.027,
        "corpus_macro_raw_doc_opp_recall1000_loss": 0.0072,
    }

    def verify_pre_stage2(fresh_data, archive_path):
        if not os.path.exists(archive_path):
            raise FileNotFoundError(
                f"FATAL: Continuation mode requires preserved Checkpoint B artifact to verify against, "
                f"but '{archive_path}' was not found! Aborting before Stage 2."
            )
        with open(archive_path, "r", encoding="utf-8") as f:
            archived_data = json.load(f)
        if fresh_data != archived_data:
            raise ValueError(
                f"FATAL: Freshly computed Checkpoint B loss gate does not match archived Stage 1 halt artifact!\n"
                f"  Fresh:    {fresh_data}\n"
                f"  Archived: {archived_data}\n"
                f"Aborting before spending compute on Stage 2."
            )

    # 1. Missing archive -> FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Continuation mode requires preserved Checkpoint B artifact"):
        verify_pre_stage2(fresh_gate_data, archived_gate_path)

    # 2. Mismatched data -> ValueError
    os.makedirs(os.path.dirname(archived_gate_path), exist_ok=True)
    with open(archived_gate_path, "w", encoding="utf-8") as f:
        json.dump({**fresh_gate_data, "corpus_macro_delta_ndcg10_loss": 0.015}, f)
    with pytest.raises(ValueError, match="Freshly computed Checkpoint B loss gate does not match"):
        verify_pre_stage2(fresh_gate_data, archived_gate_path)

    # 3. Matching data -> Passes cleanly
    with open(archived_gate_path, "w", encoding="utf-8") as f:
        json.dump(fresh_gate_data, f)
    verify_pre_stage2(fresh_gate_data, archived_gate_path)


