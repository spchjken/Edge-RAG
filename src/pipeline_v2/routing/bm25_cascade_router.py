import re
from typing import List, Dict, Any


class BM25CascadeRouter:
    """
    3-Way BM25-Driven Cascade Router (Bypass / Rerank / Discard).
    Computes routing score using Normalized BM25 score and Aspect Coverage alpha.
    """

    def __init__(self, tau_bypass: float = 0.75, tau_discard: float = 0.15):
        self.tau_bypass = tau_bypass
        self.tau_discard = tau_discard

    def route(
        self,
        candidates: List[Dict[str, Any]],
        aspect_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Triages candidate chunks into bypass_queue, rerank_queue, and discarded.
        """
        if not candidates:
            return {"bypass": [], "rerank": [], "discarded": []}

        max_bm25_score = max([c["score"] for c in candidates]) if candidates else 1.0
        aspects = aspect_payload.get("aspects", [])
        N_aspects = max(1, len(aspects))

        bypass_queue = []
        rerank_queue = []
        discarded_queue = []

        for cand in candidates:
            raw_score = cand["score"]
            norm_bm25 = raw_score / max_bm25_score if max_bm25_score > 0 else 0.0

            # Calculate Aspect Coverage alpha
            doc_text = cand["text"].lower()
            matched_aspects = 0
            for asp in aspects:
                asp_keywords = [kw["term"].lower() for kw in asp.get("keywords", [])]
                if any(kw in doc_text for kw in asp_keywords):
                    matched_aspects += 1

            alpha = matched_aspects / float(N_aspects)
            score_route = norm_bm25 * (0.5 + 0.5 * alpha)

            cand_entry = {
                "chunk_id": cand["chunk_id"],
                "text": cand["text"],
                "score": raw_score,
                "score_route": score_route,
                "alpha": alpha
            }

            if score_route >= self.tau_bypass:
                bypass_queue.append(cand_entry)
            elif score_route < self.tau_discard:
                discarded_queue.append(cand_entry)
            else:
                rerank_queue.append(cand_entry)

        return {
            "bypass": bypass_queue,
            "rerank": rerank_queue,
            "discarded": discarded_queue
        }
