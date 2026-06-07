try:
    import torch
    from llmlingua import PromptCompressor
except ImportError:
    torch = None
    PromptCompressor = None

from typing import List, Dict, Any

class LLMLinguaBaseline:
    """
    LLMLingua-2 Compression Baseline.
    Optimized for VRAM using the smaller XLM-RoBERTa model in fp16.
    """
    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
        if not PromptCompressor:
            raise ImportError("llmlingua library is required for LLMLinguaBaseline.")
            
        # Optimization: Use the token-classification distilled model (LLMLingua-2) in fp16.
        # This requires significantly less VRAM (<2GB) than loading a 7B LLM.
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map="cuda" if torch and torch.cuda.is_available() else "cpu"
        )
        
        if torch and torch.cuda.is_available():
            # Force fp16 if supported
            self.compressor.model.half()

    def log_vram_usage(self):
        if torch and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[LLMLingua Baseline VRAM] Peak memory allocated: {peak_vram:.2f} GB")

    def compress(self, query: str, context_chunks: List[str], target_ratio: float = 0.3) -> str:
        """
        Compresses the context chunks given the query string.
        """
        # Join context chunks
        context_str = "\n\n".join(context_chunks)
        
        # Perform compression
        results = self.compressor.compress_prompt(
            context=context_str,
            question=query,
            target_token=int(len(context_str.split()) * target_ratio),
            rate=target_ratio,
            force_tokens=['\n', '.', '!', '?', ','],
        )
        
        self.log_vram_usage()
        return results.get("compressed_prompt", "")
