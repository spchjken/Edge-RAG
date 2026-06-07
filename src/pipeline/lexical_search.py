import yaml
import ahocorasick
from typing import List, Dict, Any

class LexicalSearcher:
    """
    Implements the CPU-bound Aho-Corasick lexical search and 
    the 1D Continuous Interval Merging Algorithm.
    See section 3.2 of ARCHITECTURE.md.
    """
    def __init__(self, config_path: str = "configs/thresholds.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Approximate 1 token = 4 characters for raw string slicing
        self.window_l_chars = config.get("extraction", {}).get("window_L", 50) * 4

    def build_automaton(self, aspects: List[Dict[str, Any]]) -> ahocorasick.Automaton:
        """
        Builds the Aho-Corasick automaton from the JSON expansion output.
        """
        A = ahocorasick.Automaton()
        for aspect in aspects:
            aspect_name = aspect["name"]
            aspect_weight = aspect["aspect_weight"]
            for keyword_obj in aspect["keywords"]:
                term = keyword_obj["term"].lower()
                weight = keyword_obj["weight"]
                # Store metadata to retrieve upon hit
                A.add_word(term, (term, weight, aspect_name, aspect_weight))
        A.make_automaton()
        return A

    def search_and_merge(self, chunk_text: str, A: ahocorasick.Automaton) -> List[Dict[str, Any]]:
        """
        Scans chunk_text using pyahocorasick.
        Extracts symmetric windows around hits, tracks weights, and applies 
        the 1D Continuous Interval Merging Algorithm.
        """
        intervals = []
        chunk_lower = chunk_text.lower()
        chunk_len = len(chunk_text)
        
        # 1. Exact String Matching (O(N))
        for end_idx, (term, term_weight, aspect_name, aspect_weight) in A.iter(chunk_lower):
            start_idx = end_idx - len(term) + 1
            center_idx = (start_idx + end_idx) // 2
            
            # Extract symmetric window L
            win_start = max(0, center_idx - self.window_l_chars // 2)
            win_end = min(chunk_len, center_idx + self.window_l_chars // 2)
            
            intervals.append({
                "start": win_start,
                "end": win_end,
                "keyword_weights": [term_weight],
                "aspects": {aspect_name: aspect_weight}
            })
            
        # 2. 1D Continuous Interval Merging Algorithm
        if not intervals:
            return []
            
        intervals.sort(key=lambda x: x["start"])
        merged = [intervals[0]]
        
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr["start"] <= prev["end"]:
                # Collapse overlapping windows
                prev["end"] = max(prev["end"], curr["end"])
                prev["keyword_weights"].extend(curr["keyword_weights"])
                prev["aspects"].update(curr["aspects"])
            else:
                merged.append(curr)
                
        # 3. Finalize payloads
        compressed_samples = []
        for m in merged:
            extracted_text = chunk_text[m["start"]:m["end"]]
            compressed_samples.append({
                "text": extracted_text,
                "length": m["end"] - m["start"],
                "max_keyword_weight": max(m["keyword_weights"]),
                "aspects": m["aspects"]
            })
            
        return compressed_samples
