import json
from typing import List, Dict, Any
from src.utils.llm_client import LLMClient

class LLMAspectExtractor:
    """
    Extracts a list of orthogonal aspects and their weights from a query using an LLM.
    Strictly follows the blueprint in pathway_llm_aspects.md (no keyword expansion).
    """
    def __init__(self, model_name: str = "qwen3.5-2b"):
        self.llm = LLMClient(model_name=model_name)
        
        # Enforced JSON Schema for extraction
        self.schema = {
            "type": "object",
            "properties": {
                "aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The extracted orthogonal aspect"
                            },
                            "weight": {
                                "type": "number",
                                "description": "Float from 0.0 to 1.0 representing constraint importance"
                            }
                        },
                        "required": ["name", "weight"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["aspects"],
            "additionalProperties": False
        }

    def _build_prompt(self, query: str) -> str:
        return f"""You are an expert query parser for a retrieval engine. 
Your goal is to break the user's query into a list of orthogonal (mutually exclusive) core aspects.

CORE RULES:
1. Extract EXACTLY what is in the query. Do not add related concepts. Do not expand the terms.
2. Identify the core entities, actions, and constraints.
3. Combine tightly coupled modifiers (e.g., 'Third-party AI models' is one aspect, not 'Third-party' and 'AI models').
4. Never drop short acronyms (e.g., 'AI', 'API') or negations.
5. Do not hallucinate product categories or generic terms not explicitly stated in the query.
6. CRITICAL: You MUST return exact, unmodified sub-strings from the query. DO NOT format words as snake_case. DO NOT replace spaces with underscores. Keep original spacing and punctuation.

WEIGHTING LOGIC (0.0 to 1.0):
- 1.0 (Critical): Core entities, hard constraints, acronyms (e.g., 'Digital Experience', 'AI').
- 0.5 - 0.9 (Secondary): Actions, soft modifiers, generic context (e.g., 'integrate', 'strategy').

USER QUERY:
"{query}"

Return a JSON object containing the list of extracted aspects and their weights.
"""

    def extract(self, query: str) -> List[Dict[str, Any]]:
        """
        Extracts aspects from the query.
        Returns a list of dicts: [{'name': 'aspect', 'weight': 1.0}, ...]
        """
        if not query or not query.strip():
            return []
            
        prompt = self._build_prompt(query)
        
        # Call the LLM with strict JSON schema enforcement and 0.0 temperature
        raw_response = self.llm.generate(
            prompt=prompt,
            json_schema=self.schema,
            temperature=0.0
        )
        
        try:
            parsed = json.loads(raw_response)
            return parsed.get("aspects", [])
        except json.JSONDecodeError:
            # Fallback if the LLM somehow breaks the schema enforcement
            return [{"name": query, "weight": 1.0}]
