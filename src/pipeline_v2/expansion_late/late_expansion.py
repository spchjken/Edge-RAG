from typing import List, Dict, Any, Optional
from src.utils.llm_client import LLMClient


class LateExpansionV2:
    """
    Late Expansion & VRAM Safety Module for Pipeline V2.
    Restores original uncompressed chunk text and enforces N_max <= 10 VRAM budget.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, N_max: int = 10):
        self.llm_client = llm_client if llm_client is not None else LLMClient(model_name="qwen3.5-2b")
        self.N_max = N_max

    def generate(
        self,
        query: str,
        target_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restores chunks, enforces N_max budget, and generates final answer.
        """
        # Enforce N_max budget cap
        final_chunks = target_chunks[:self.N_max]

        # Format context blocks
        context_blocks = []
        for i, chunk in enumerate(final_chunks):
            cid = chunk.get("chunk_id", f"c_{i}")
            text = chunk.get("text", "")
            context_blocks.append(f"--- DOCUMENT [ID: {cid}] ---\n{text}")

        context_str = "\n\n".join(context_blocks)
        sys_prompt = system_prompt or "You are a helpful AI assistant. Answer the query using ONLY the provided document context."

        user_prompt = (
            f"Context Documents:\n{context_str}\n\n"
            f"Query: {query}\n\n"
            f"Provide a concise, fact-grounded answer based strictly on the document context:"
        )

        answer = self.llm_client.generate(prompt=user_prompt)

        return {
            "answer": answer,
            "used_chunks": [c.get("chunk_id") for c in final_chunks],
            "num_chunks_used": len(final_chunks)
        }
