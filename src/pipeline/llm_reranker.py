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

RERANK_PROMPT_TEMPLATE = """You are an expert relevance evaluator.
Given a user query and a set of discontinuous text intervals extracted from a single document, determine if the document is highly relevant to answering the query.
You must treat these evidence samples as independent snippets, not continuous prose.

User Query: {query}

Document Evidence:
{evidence_json}

Evaluate the evidence. Return a JSON object with:
- "relevant": true if the evidence strongly supports answering the query, false otherwise.
- "relevance_score": a float between 0.0 and 1.0 indicating confidence.
DO NOT provide any explanations. Return ONLY the JSON object.
"""

class LLMReranker:
    """
    Implements the JSON-Structured Listwise Reranking to avoid sequence-break hallucinations.
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
