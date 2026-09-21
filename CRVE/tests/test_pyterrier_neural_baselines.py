"""
tests/test_pyterrier_neural_baselines.py

Verifies standard PyTerrier ecosystem neural retrieval pipelines:
1. Dense Retrieval using `pyterrier_dr` (BAAI/bge-small-en-v1.5 + FlexIndex)
2. Neural Sparse Retrieval using `pyterrier_splade` + `pyterrier_pisa` (naver/splade-v3-distilbert + PisaIndex)
3. End-to-end evaluation with `ir_measures` linear nDCG@10.
"""

import os
import shutil
import tempfile
import pytest
import pandas as pd
import pyterrier as pt
import ir_measures
from ir_measures import nDCG, R

if not pt.java.started():
    pt.java.init()

import pyterrier_dr as pt_dr
import pyterrier_splade as pt_splade
from pyterrier_pisa import PisaIndex


@pytest.fixture(scope="module")
def sample_dataset():
    docs = [
        {"docno": "d1", "text": "The quick brown fox jumps over the lazy dog."},
        {"docno": "d2", "text": "Deep learning and artificial intelligence are transforming modern research."},
        {"docno": "d3", "text": "Edge-RAG reduces VRAM consumption using statistical query expansion."},
        {"docno": "d4", "text": "Financial market microstructure and algorithmic trading systems require low latency."},
    ]
    queries = pd.DataFrame([
        {"qid": "q1", "query": "Edge RAG sparse retrieval"},
        {"qid": "q2", "query": "deep learning artificial intelligence"},
    ])
    qrels = pd.DataFrame([
        {"qid": "q1", "docno": "d3", "iteration": "0", "relevance": 1},
        {"qid": "q2", "docno": "d2", "iteration": "0", "relevance": 1},
    ])
    return docs, queries, qrels


def test_pyterrier_dr_bge_small(sample_dataset):
    """Verifies pyterrier_dr HgfBiEncoder + FlexIndex end-to-end."""
    docs, queries, qrels = sample_dataset
    tmpdir = tempfile.mkdtemp()
    try:
        model = pt_dr.HgfBiEncoder.from_pretrained("BAAI/bge-small-en-v1.5", device="cuda", batch_size=4)
        idx_path = os.path.join(tmpdir, "flex_index")
        indexer = pt_dr.FlexIndex(idx_path)
        
        # Build index
        (model >> indexer).index(docs)
        assert os.path.exists(idx_path)

        # Retrieve
        pipeline = model.query_encoder() >> indexer.retriever(num_results=4)
        results = pipeline.transform(queries)
        assert len(results) > 0
        assert "docno" in results.columns and "score" in results.columns

        # Evaluate
        eval_res = pt.Experiment(
            [pipeline],
            queries,
            qrels,
            eval_metrics=[nDCG @ 10, R @ 10],
            names=["BGE_Small_Dense"]
        )
        assert len(eval_res) == 1
        assert eval_res.iloc[0]["nDCG@10"] > 0.5
    finally:
        shutil.rmtree(tmpdir)


def test_pyterrier_splade_pisa(sample_dataset):
    """Verifies pyterrier_splade + pyterrier_pisa end-to-end."""
    docs, queries, qrels = sample_dataset
    tmpdir = tempfile.mkdtemp()
    try:
        splade = pt_splade.Splade(model="naver/splade-v3-distilbert", device="cuda")
        idx_path = os.path.join(tmpdir, "pisa_index")
        pisa_index = PisaIndex(idx_path, stemmer="none")

        # Build index
        idx_pipeline = splade.doc_encoder() >> pisa_index.toks_indexer()
        idx_pipeline.index(docs)
        assert pisa_index.built()

        # Retrieve
        retr_pipeline = splade.query_encoder() >> pisa_index.quantized(num_results=4)
        results = retr_pipeline.transform(queries)
        assert len(results) > 0
        assert "docno" in results.columns and "score" in results.columns

        # Evaluate
        eval_res = pt.Experiment(
            [retr_pipeline],
            queries,
            qrels,
            eval_metrics=[nDCG @ 10, R @ 10],
            names=["SPLADE_v3_PISA"]
        )
        assert len(eval_res) == 1
        assert eval_res.iloc[0]["nDCG@10"] > 0.5
    finally:
        shutil.rmtree(tmpdir)
