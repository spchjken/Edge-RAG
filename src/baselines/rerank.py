"""Independent listwise LLM reranker for baselines (full-passage variant).

Self-contained per the baseline isolation rule (must not import from
``src.pipeline``). This reranker operates on **full retrieved passages**, not
the compressed interval fragments the Edge-RAG pipeline's reranker consumes.

The pipeline's ``ListwiseLLMReranker`` prompt is specifically written for
*compressed, fragmented evidence* ("treat evidence samples as independent
snippets, do not penalize missing narrative context"). This baseline variant is
deliberately different: it feeds each candidate's complete chunk text as one
passage and uses a prompt written for whole passages. This models the standard
retriever + passage-level reranker RAG recipe.

Comparability note: in the fair retriever+reranker comparison, Edge-RAG's
reranker sees compressed snippets (less token noise, less context) while the
baselines' reranker sees full passages. That asymmetry is a property of
Edge-RAG's compression and is part of the system being compared, not a bug.

Uses the same RankCoT anchor + lexical pre-sorting + batch listwise method and
the same ``gemma4-e2b`` model as the pipeline reranker.
"""
import json
from typing import List, Dict, Any

from src.utils.llm_client import LLMClient

PASSAGE_RERANK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target_fact": {"type": "string"},
        "relevant_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target_fact", "relevant_chunk_ids"],
}

PASSAGE_RERANK_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a set of candidate document passages, identify which passages directly contain factual answers, metrics, definitions, or named entities needed to satisfy the query.
Each passage is a complete excerpt from a document; evaluate it as a whole.

User Query: {query}

Candidate Document Passages:
{documents_text}

Evaluate all candidate passages globally. Return a JSON object with:
- "target_fact": a brief 3-6 word summary of the key factual requirement needed to answer the query.
- "relevant_chunk_ids": a list of candidate passage IDs (e.g. ["doc_0", "doc_2"]) that contain direct evidence, ordered by relevance (most relevant first).

DO NOT provide any other text or explanations. Return ONLY the JSON object.
"""

PASSAGE_NO_ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relevant_chunk_ids"],
}

PASSAGE_NO_ANCHOR_PROMPT_TEMPLATE = """You are an expert factual relevance evaluator for Retrieval-Augmented Generation.
Given a user query and a set of candidate document passages, identify which passages directly contain factual answers, metrics, definitions, or named entities needed to satisfy the query.
Each passage is a complete excerpt from a document; evaluate it as a whole.

User Query: {query}

Candidate Document Passages:
{documents_text}

Return a JSON object with a single key "relevant_chunk_ids" containing a list of strings of the passage IDs that are relevant, sorted by their relevance (most relevant first).
DO NOT provide any explanations. Return ONLY the JSON object.
"""


def clean_json_response(text: str) -> str:
    """Strips markdown code fences or stray whitespace around a JSON response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class BaselineListwiseReranker:
    """
    Applies the listwise LLM rerank step to a baseline retriever's top-K.

    Candidates are plain chunks ``{"chunk_id", "text", "score"}``. Each chunk's
    full ``text`` is presented as one document passage (truncated to bound the
    prompt). The prompt is written for whole passages, not compressed snippets.
    """

    def __init__(
        self,
        model_name: str = "gemma4-e2b",
        config_path: str = "configs/models.yaml",
        batch_size: int = 10,
        enable_anchor: bool = True,
        enable_presort: bool = True,
        max_evidence_chars: int = 2000,
    ):
        self.llm = LLMClient(model_name=model_name, config_path=config_path)
        self.batch_size = batch_size
        self.enable_anchor = enable_anchor
        self.enable_presort = enable_presort
        self.max_evidence_chars = max_evidence_chars

    def rerank(self, candidates: List[Dict[str, Any]],
               query: str) -> List[Dict[str, Any]]:
        """
        Reranks the retrieved candidate passages and returns the accepted subset,
        sorted most-relevant first. Accepts candidates with real chunk IDs.
        """
        if not candidates:
            return []

        # Pre-sort by the retriever's initial score (most relevant first),
        # counteracting listwise position bias.
        working_queue = list(candidates)
        if self.enable_presort:
            working_queue.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        schema = (PASSAGE_RERANK_OUTPUT_SCHEMA if self.enable_anchor
                  else PASSAGE_NO_ANCHOR_SCHEMA)
        prompt_tmpl = (PASSAGE_RERANK_PROMPT_TEMPLATE if self.enable_anchor
                       else PASSAGE_NO_ANCHOR_PROMPT_TEMPLATE)

        verified_chunks = []
        for i in range(0, len(working_queue), self.batch_size):
            batch = working_queue[i:i + self.batch_size]

            alias_map: Dict[str, Dict[str, Any]] = {}
            doc_lines = []
            for idx, chunk in enumerate(batch):
                alias = f"doc_{idx}"
                alias_map[alias] = chunk
                alias_map[chunk.get("chunk_id")] = chunk  # fallback map
                text = chunk.get("text", "")
                if len(text) > self.max_evidence_chars:
                    text = text[:self.max_evidence_chars]
                doc_lines.append(f"Document ID: {alias}")
                doc_lines.append(f"Passage:\n{text.strip()}")
                doc_lines.append("-" * 40)

            documents_text = "\n".join(doc_lines)
            prompt = prompt_tmpl.format(query=query, documents_text=documents_text)

            response_text = ""
            try:
                response_text = self.llm.generate(
                    prompt=prompt,
                    json_schema=schema,
                    temperature=0.0,
                )
                result = json.loads(clean_json_response(response_text))
                relevant_ids = result.get("relevant_chunk_ids", [])

                for rank, cid in enumerate(relevant_ids):
                    if cid in alias_map:
                        chunk = alias_map[cid]
                        chunk["relevance_score"] = max(0.0, 1.0 - rank * 0.1)
                        if chunk not in verified_chunks:
                            verified_chunks.append(chunk)
            except Exception as e:
                print(f"Warning: BaselineListwiseReranker failed for batch slice: {e}")
                print(f"Raw response was: {repr(response_text)}")
                continue

        return verified_chunks
