import json
import yaml
import requests
from typing import Dict, Any, Optional

class LLMClient:
    """
    Generic wrapper for LLM backends (Ollama and llama-cpp).
    Auto-detects backend based on the model configuration.
    """
    def __init__(self, model_name: str, config_path: str = "configs/models.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if model_name not in config["models"]:
            raise ValueError(f"Model {model_name} not found in config.")
            
        self.model_config = config["models"][model_name]
        self.backend = self.model_config["backend"]
        self.endpoint = self.model_config["endpoint"]
        self.tag = self.model_config.get("tag", "")

    def generate(self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0) -> str:
        """
        Generates text using the configured backend.
        If json_schema is provided, enforces constrained decoding.
        """
        if self.backend == "ollama":
            payload = {
                "model": self.tag,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            if json_schema:
                # Ollama supports constrained decoding via the format parameter
                payload["format"] = json_schema
                
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
            
        elif self.backend == "llama-cpp":
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            if json_schema:
                # llama.cpp server supports json_schema parameter natively
                payload["json_schema"] = json_schema
                
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("content", "")
            
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented.")
