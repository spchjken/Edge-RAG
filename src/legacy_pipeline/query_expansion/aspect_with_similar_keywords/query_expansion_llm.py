import json
from typing import Dict, Any
from src.utils.llm_client import LLMClient

# JSON Schema mapping to the structure defined in section 3.1 of ARCHITECTURE.md
QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aspect_weight": {"type": "number"},
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "weight": {"type": "number"}
                            },
                            "required": ["term", "weight"]
                        }
                    }
                },
                "required": ["name", "aspect_weight", "keywords"]
            }
        }
    },
    "required": ["aspects"]
}

PROMPT_TEMPLATE = """You are an expert query expansion agent. Given a user query, decompose it into its core relational aspects.
For each aspect, assign an aspect weight (0.0 to 1.0) based on its importance to the query.
Then, generate synonyms and keywords for that aspect, assigning a precision weight (0.0 to 1.0) to each keyword.
DO NOT provide any Chain-of-Thought or explanation. Return ONLY the JSON object.

Example Query: What are the side effects of Drug X?
Example JSON Output:
{{
  "aspects": [
    {{
      "name": "Drug X",
      "aspect_weight": 1.0,
      "keywords": [
        {{"term": "Drug X", "weight": 1.0}},
        {{"term": "chemical name Y", "weight": 0.8}}
      ]
    }},
    {{
      "name": "Side effects",
      "aspect_weight": 0.8,
      "keywords": [
        {{"term": "side effects", "weight": 1.0}},
        {{"term": "adverse reactions", "weight": 0.9}},
        {{"term": "toxicity", "weight": 0.7}}
      ]
    }}
  ]
}}

User Query: {query}
Output JSON:
<think>
</think>

"""

class QueryExpander:
    def __init__(self, model_name: str, config_path: str = "configs/models.yaml"):
        """
        Initializes the QueryExpander with the designated LLM configuration.
        """
        self.llm = LLMClient(model_name=model_name, config_path=config_path)

    def expand(self, query: str) -> Dict[str, Any]:
        """
        Expands a natural language query into a structured JSON of weighted lexical anchors.
        See section 3.1 of ARCHITECTURE.md for specifications.
        """
        prompt = PROMPT_TEMPLATE.format(query=query)
        
        # We strictly avoid CoT to preserve ultra-low TTFT and rely on constrained JSON decoding.
        response_text = self.llm.generate(
            prompt=prompt, 
            json_schema=QUERY_EXPANSION_SCHEMA,
            temperature=0.0
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode LLM response as JSON: {response_text}") from e
