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
        self.context_window = self.model_config.get("context_window", 2048)

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
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.context_window,
                    "num_predict": 4096
                }
            }
            if json_schema:
                # Ollama structured outputs: pass schema directly to `format`
                # for constrained decoding via GBNF grammar (v0.5.0+).
                # This guarantees valid JSON matching the schema shape,
                # unlike `format: "json"` which is only a prompt-level hint.
                payload["format"] = json_schema
                
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            res_json = response.json()
            out_text = res_json.get("response", "")
            if not out_text and "thinking" in res_json:
                out_text = res_json.get("thinking", "")
            return out_text
            
        elif self.backend == "llama-cpp":
            # Format ChatML natively and use prompt injection to shutdown thinking mode
            if "<|im_start|>" not in prompt:
                prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n"

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
