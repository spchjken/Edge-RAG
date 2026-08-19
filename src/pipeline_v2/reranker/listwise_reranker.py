import re
from typing import List, Dict, Any, Optional
from src.utils.llm_client import LLMClient


class ListwiseLLMRerankerV2:
    """
    Single-pass Listwise LLM Reranker for Pipeline V2.
    Extracts IDF-filtered sentence snippets (~250 tokens per chunk) around anchor terms
    before passing candidates to the LLM client.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, top_k: int = 5):
        self.llm_client = llm_client if llm_client is not None else LLMClient(model_name="qwen3.5-2b")
        self.top_k = top_k

    def extract_idf_snippets(self, text: str, query_terms: List[str], max_tokens: int = 250) -> str:
        """Extracts ~250 token window around high-IDF query term anchors."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text[:1000]

        query_terms_lower = [t.lower() for t in query_terms]
        sentence_scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for q in query_terms_lower if q in sent_lower)
            sentence_scores.append(score)

        # Select top-scoring sentences
        ranked = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)
        selected = [s for s, sc in ranked if sc > 0]

        if not selected:
            selected = sentences[:3]

        snippet = " ".join(selected)
        words = snippet.split()
        if len(words) > max_tokens:
            snippet = " ".join(words[:max_tokens]) + "..."
        return snippet

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        query_terms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes single-pass Listwise reranking over candidate chunks.
        """
        if not candidates:
            return []

        q_terms = query_terms if query_terms is not None else query.lower().split()

        # Build snippet payloads
        candidate_snippets = []
        for i, cand in enumerate(candidates):
            snippet = self.extract_idf_snippets(cand["text"], q_terms)
            candidate_snippets.append({
                "id": i + 1,
                "chunk_id": cand["chunk_id"],
                "snippet": snippet,
                "full_text": cand["text"],
                "score": cand.get("score", 0.0)
            })

        # Format Listwise prompt
        formatted_candidates = "\n".join(
            [f"[{item['id']}] Chunk ID: {item['chunk_id']}\nSnippet: {item['snippet']}" for item in candidate_snippets]
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Rank the following candidate text snippets by relevance to the query. "
            f"Return a JSON array of chunk IDs in order of relevance, e.g. [\"chunk_id_1\", \"chunk_id_2\"].\n\n"
            f"{formatted_candidates}\n\n"
            f"Output JSON array:"
        )

        try:
            response = self.llm_client.generate(prompt=prompt)
            # Parse JSON array from response
            import json
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                ranked_ids = json.loads(match.group(0))
                id_to_cand = {c["chunk_id"]: c for c in candidates}
                reranked = [id_to_cand[cid] for cid in ranked_ids if cid in id_to_cand]
                # Append un-ranked fallback
                for cand in candidates:
                    if cand not in reranked:
                        reranked.append(cand)
                return reranked[:self.top_k]
        except Exception:
            pass

        # Fallback to sorting by raw score
        sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_cands[:self.top_k]
