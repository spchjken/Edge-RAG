import json
from typing import List, Dict, Any
from src.utils.llm_client import LLMClient

RERANK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "relevance_score": {"type": "number"}
    },
    "required": ["relevant", "relevance_score"]
}

RERANK_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a set of compressed document evidence samples, determine if the evidence directly contains factual answers, metrics, definitions, or named entities needed to satisfy the query.
You must treat these evidence samples as independent snippets, not continuous prose. Do not penalize text for missing narrative context.

User Query: {query}

Document Evidence Samples:
{evidence_json}

Evaluate the evidence. Return a JSON object with:
- "relevant": true if the evidence contains direct factual answers or key entity/metric matches, false otherwise.
- "relevance_score": a float between 0.0 and 1.0 indicating confidence.
DO NOT provide any explanations. Return ONLY the JSON object.
"""

class LLMReranker:
    """
    Implements the JSON-Structured Pointwise Reranking to avoid sequence-break hallucinations.
    See section 3.4 of ARCHITECTURE.md.
    """
    def __init__(self, model_name: str, config_path: str = "configs/models.yaml"):
        self.llm = LLMClient(model_name=model_name, config_path=config_path)

    def format_evidence(self, chunk: Dict[str, Any]) -> str:
        """
        Wraps the discontinuous intervals into the strict JSON schema.
        """
        samples = [m.get("text", "") for m in chunk.get("compressed_samples", [])]
        evidence_obj = {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "evidence_samples": samples
        }
        return json.dumps(evidence_obj, indent=2)

    def rerank(self, rerank_queue: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Reranks the chunks in the rerank_queue using the LLM.
        Returns a list of chunks that the LLM deemed relevant, sorted by relevance_score.
        """
        verified_chunks = []
        
        for chunk in rerank_queue:
            evidence_json = self.format_evidence(chunk)
            prompt = RERANK_PROMPT_TEMPLATE.format(query=query, evidence_json=evidence_json)
            
            try:
                response_text = self.llm.generate(
                    prompt=prompt,
                    json_schema=RERANK_OUTPUT_SCHEMA,
                    temperature=0.0
                )
                result = json.loads(response_text)
                
                if result.get("relevant", False):
                    chunk["relevance_score"] = result.get("relevance_score", 0.0)
                    verified_chunks.append(chunk)
                    
            except Exception as e:
                # In a strict pipeline, we might skip or fallback. Here we just log/raise.
                print(f"Warning: Reranker failed for chunk {chunk.get('chunk_id')}: {str(e)}")
                continue
                
        # Sort verified chunks by the LLM's confidence score
        verified_chunks.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return verified_chunks


BATCH_RERANK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "relevance_score": {"type": "number"}
                },
                "required": ["chunk_id", "relevant", "relevance_score"]
            }
        }
    },
    "required": ["evaluations"]
}

BATCH_RERANK_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a batch of compressed document evidence samples, evaluate each document independently to determine if its evidence directly contains factual answers, metrics, definitions, or named entities needed to satisfy the query.
You must treat evidence samples within each document as independent snippets, not continuous prose. Do not penalize text for missing narrative context.

User Query: {query}

Candidate Document Evidence Samples:
{documents_text}

Evaluate each document independently. Return a JSON object with a single key "evaluations" containing a list of objects, one for each document:
- "chunk_id": the ID of the document evaluated.
- "relevant": true if the evidence contains direct factual answers or key entity/metric matches, false otherwise.
- "relevance_score": a float between 0.0 and 1.0 indicating confidence.
DO NOT provide any explanations. Return ONLY the JSON object.
"""

def clean_json_response(text: str) -> str:
    """
    Cleans up any markdown wrapper blocks (e.g. ```json ... ```) or whitespace.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class BatchPointwiseLLMReranker:
    """
    Evaluates a batch of candidate chunks in a single prompt to save request latency.
    """
    def __init__(self, model_name: str, config_path: str = "configs/models.yaml", batch_size: int = 5):
        self.llm = LLMClient(model_name=model_name, config_path=config_path)
        self.batch_size = batch_size

    def rerank(self, rerank_queue: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        verified_chunks = []
        
        # Partition into batches
        for i in range(0, len(rerank_queue), self.batch_size):
            batch = rerank_queue[i : i + self.batch_size]
            
            alias_map = {}  # alias/original -> original chunk
            doc_lines = []
            for idx, chunk in enumerate(batch):
                alias = f"doc_{idx}"
                alias_map[alias] = chunk
                # Fallback mapping in case LLM outputs original chunk_id
                alias_map[chunk.get("chunk_id")] = chunk
                
                samples = [m.get("text", "") for m in chunk.get("compressed_samples", [])]
                doc_lines.append(f"Document ID: {alias}")
                for s_idx, sample in enumerate(samples):
                    doc_lines.append(f"Evidence Sample {s_idx + 1}:\n{sample.strip()}")
                doc_lines.append("-" * 40)
                
            documents_text = "\n".join(doc_lines)
            prompt = BATCH_RERANK_PROMPT_TEMPLATE.format(query=query, documents_text=documents_text)
            
            response_text = ""
            try:
                response_text = self.llm.generate(
                    prompt=prompt,
                    json_schema=BATCH_RERANK_OUTPUT_SCHEMA,
                    temperature=0.0
                )
                cleaned_text = clean_json_response(response_text)
                result = json.loads(cleaned_text)
                evals = result.get("evaluations", [])
                
                # Create a map for quick lookup
                eval_map = {ev.get("chunk_id"): ev for ev in evals if "chunk_id" in ev}
                
                for alias, chunk in alias_map.items():
                    # Process only alias keys (starts with doc_) to avoid double processing
                    if alias.startswith("doc_") and alias in eval_map:
                        ev = eval_map[alias]
                        if ev.get("relevant", False):
                            chunk["relevance_score"] = ev.get("relevance_score", 0.0)
                            if chunk not in verified_chunks:
                                verified_chunks.append(chunk)
            except Exception as e:
                print(f"Warning: BatchPointwiseReranker failed for batch slice: {str(e)}")
                print(f"Prompt sent:\n{prompt}\n")
                print(f"Raw response was: {repr(response_text)}")
                continue
                
        verified_chunks.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return verified_chunks


LISTWISE_RERANK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target_fact": {
            "type": "string"
        },
        "relevant_chunk_ids": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["target_fact", "relevant_chunk_ids"]
}

LISTWISE_RERANK_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a set of compressed document evidence samples, identify which evidence samples directly contain factual answers, metrics, definitions, or named entities needed to satisfy the query.
You must treat evidence samples within each document as independent snippets, not continuous prose. Do not penalize text for missing narrative context.

User Query: {query}

Candidate Document Evidence Samples:
{documents_text}

Evaluate all candidate evidence globally. Return a JSON object with:
- "target_fact": a brief 3-6 word summary of the key factual requirement needed to answer the query.
- "relevant_chunk_ids": a list of candidate chunk IDs (e.g. ["doc_0", "doc_2"]) that contain direct evidence, ordered by relevance (most relevant first).

DO NOT provide any other text or explanations. Return ONLY the JSON object.
"""

LISTWISE_NO_ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_chunk_ids": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["relevant_chunk_ids"]
}

LISTWISE_NO_ANCHOR_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a set of compressed document evidence samples, identify which evidence samples directly contain factual answers, metrics, definitions, or named entities needed to satisfy the query.
You must treat evidence samples within each document as independent snippets, not continuous prose. Do not penalize text for missing narrative context.

User Query: {query}

Candidate Document Evidence Samples:
{documents_text}

Return a JSON object with a single key "relevant_chunk_ids" containing a list of strings of the chunk IDs that are relevant, sorted by their relevance (most relevant first).
DO NOT provide any explanations. Return ONLY the JSON object.
"""

class ListwiseLLMReranker:
    """
    Submits a comparative list of candidate chunks in one prompt to globally rank relevance.
    Supports ablation configurations for RankCoT anchor and lexical pre-sorting.
    """
    def __init__(
        self,
        model_name: str,
        config_path: str = "configs/models.yaml",
        batch_size: int = 10,
        enable_anchor: bool = True,
        enable_presort: bool = True
    ):
        self.llm = LLMClient(model_name=model_name, config_path=config_path)
        self.batch_size = batch_size
        self.enable_anchor = enable_anchor
        self.enable_presort = enable_presort

    def rerank(self, rerank_queue: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not rerank_queue:
            return []
            
        verified_chunks = []
        
        # Pre-sort candidate queue by initial search score / sample length if enabled
        if self.enable_presort:
            working_queue = sorted(
                rerank_queue,
                key=lambda c: sum(s.get("length", 0) for s in c.get("compressed_samples", [])),
                reverse=True
            )
        else:
            working_queue = list(rerank_queue)
            
        schema = LISTWISE_RERANK_OUTPUT_SCHEMA if self.enable_anchor else LISTWISE_NO_ANCHOR_SCHEMA
        prompt_tmpl = LISTWISE_RERANK_PROMPT_TEMPLATE if self.enable_anchor else LISTWISE_NO_ANCHOR_PROMPT_TEMPLATE
        
        # Partition into batches
        for i in range(0, len(working_queue), self.batch_size):
            batch = working_queue[i : i + self.batch_size]
            
            alias_map = {}  # alias/original -> original chunk
            doc_lines = []
            for idx, chunk in enumerate(batch):
                alias = f"doc_{idx}"
                alias_map[alias] = chunk
                # Fallback mapping in case LLM outputs original chunk_id
                alias_map[chunk.get("chunk_id")] = chunk
                
                samples = [m.get("text", "") for m in chunk.get("compressed_samples", [])]
                doc_lines.append(f"Document ID: {alias}")
                for s_idx, sample in enumerate(samples):
                    doc_lines.append(f"Evidence Sample {s_idx + 1}:\n{sample.strip()}")
                doc_lines.append("-" * 40)
                
            documents_text = "\n".join(doc_lines)
            prompt = prompt_tmpl.format(query=query, documents_text=documents_text)
            
            response_text = ""
            try:
                response_text = self.llm.generate(
                    prompt=prompt,
                    json_schema=schema,
                    temperature=0.0
                )
                cleaned_text = clean_json_response(response_text)
                result = json.loads(cleaned_text)
                relevant_ids = result.get("relevant_chunk_ids", [])
                
                # To maintain order and filter
                for rank, cid in enumerate(relevant_ids):
                    if cid in alias_map:
                        chunk = alias_map[cid]
                        chunk["relevance_score"] = max(0.0, 1.0 - rank * 0.1)
                        if chunk not in verified_chunks:
                            verified_chunks.append(chunk)
            except Exception as e:
                print(f"Warning: ListwiseReranker failed for batch slice: {str(e)}")
                print(f"Prompt sent:\n{prompt}\n")
                print(f"Raw response was: {repr(response_text)}")
                continue
                
        return verified_chunks

