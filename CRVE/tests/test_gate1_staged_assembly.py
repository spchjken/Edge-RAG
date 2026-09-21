import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.run_gate1_oracle_evaluation import assemble_shards_to_master
from scripts.compile_gate1_research_tables import Gate1TableCompiler


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
