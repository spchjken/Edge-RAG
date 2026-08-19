import os
import re
import glob
import json
import time
import uuid
import argparse
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ensure you have DEEPSEEK_API_KEY in your .env
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """\
You are an expert RAG evaluation dataset creator. You will be provided with a single "Parent Block" (a contiguous section of text extracted from an AI/Fintech research paper, up to 3000 tokens long).

Your task is to generate EXACTLY 4 high-quality queries that can be answered *strictly and entirely* using the provided text. 

To ensure maximum diversity and adversarial difficulty, the query types are divided into 4 groups. You should aim to distribute queries across these groups, choosing the specific query type that best fits the provided text.

CRITICAL GROUP DISTRIBUTION RULE:
1. While you should strive to generate one query per group, not every text block naturally supports all query types.
2. Specifically, if the text block does not naturally support "Group 3: Analytical & Relational" queries (e.g., if there are no comparisons, causal mechanisms, multi-hop connections, or quantitative/statistical relations in the text), do NOT force it. Doing so leads to low-quality or hallucinated queries.
3. Instead, you may skip Group 3 (or any other group that cannot be naturally grounded) and generate an additional query from the other groups (Group 1, 2, or 4) that are better supported by the block's text.
4. Regardless of the group distribution, you MUST generate EXACTLY 4 high-quality, strictly grounded queries in total.

### Group 1: Basic Extraction (Choose 1)
- **Factual**: Direct retrieval of a specific fact, number, metric, or entity.
- **Definitional**: Asking for the definition of a specific term or acronym introduced in the text.

### Group 2: Comprehension & Explanation (Choose 1)
- **Conceptual**: Asking for an explanation of a methodology, underlying theory, or architectural choice.
- **Procedural**: Asking for the sequential steps of a process or experiment.
- **Summary**: Asking for the main takeaway, primary contribution, or core argument of a specific section.

### Group 3: Analytical & Relational (Choose 1)
- **Multi-hop**: Connecting two distinct ideas, variables, or paragraphs within the block.
- **Comparative**: Comparing two methods, entities, baselines, or results discussed in the text.
- **Causal**: Asking about cause-and-effect or the "why" behind a decision.
- **Quantitative**: Asking about mathematical relationships, formulas, or statistical significance.

### Group 4: Applied & Evaluative (Choose 1)
- **Scenario (Applied)**: Framing a technical or operational query as a specific use case scenario without conversational filler (e.g., "What parameter adjustments are recommended to prevent sequence collapse when using group-constrained rewards?").
- **Critique**: Asking to identify drawbacks, assumptions, or limitations mentioned.
- **Inferential**: A query where the answer is strongly implied by combining facts, but not explicitly spelled out in a single sentence.

CRITICAL RULES FOR QUERY PHRASING:
1. NO CONVERSATIONAL PREAMBLES: Never start queries with filler like "As a bank risk manager...", "If I am building X...", "If I want to...", "If you are asked to...", etc. Queries must start directly with search target keywords (e.g., "What", "How", "Why", "Compare", "Explain", "Which").
2. NO METADATA REFERENCES: Do not reference "the paper", "this study", "the authors", "the text", "in this section", or "in our framework". The query must target the underlying concepts directly as if searching a general web knowledge base.
3. NO ABUSE OF SYSTEM NAMES: Do not over-use specific framework names from the paper (like "AIR", "AI Economist Agent") to formulate meta-questions (e.g., do not ask "What does AIR achieve?"; instead, ask "What are the effects of a group-constrained reward function on reinforcement learning stability for interleaved reasoning?").
4. NO EXTERNAL KNOWLEDGE: The query and its golden answer must be fully answerable using ONLY the provided text block.
5. Provide a `golden_answer` that is a concise, 2-3 sentence reference answer.
6. Provide the exact string quote(s) from the text that proves the answer (`evidence_quotes`).

Format your output EXACTLY as a JSON array of objects. An example output where all 4 groups are used:
[
  {
    "query_group": "Group 1: Basic Extraction",
    "query_type": "Factual", 
    "question": "...",
    "golden_answer": "...",
    "evidence_quotes": ["..."]
  },
  {
    "query_group": "Group 2: Comprehension & Explanation",
    "query_type": "Conceptual", 
    "question": "...",
    "golden_answer": "...",
    "evidence_quotes": ["..."]
  },
  {
    "query_group": "Group 3: Analytical & Relational",
    "query_type": "Multi-hop", 
    "question": "...",
    "golden_answer": "...",
    "evidence_quotes": ["..."]
  },
  {
    "query_group": "Group 4: Applied & Evaluative",
    "query_type": "Scenario (Applied)", 
    "question": "...",
    "golden_answer": "...",
    "evidence_quotes": ["..."]
  }
]
Return ONLY the raw JSON array.
"""

def extract_json(text: str) -> list:
    if not text:
        raise ValueError("Empty response text")
        
    text = text.strip()
    
    # Strip <think>...</think> reasoning blocks from DeepSeek
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        pos = text.find("<think>")
        end_pos = text.find(">", pos)
        if end_pos != -1:
            text = text[end_pos+1:].strip()

    # 1. Try markdown code block extraction
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2. Try regex extraction of JSON array or object
    json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if json_match:
        candidate = json_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return json.loads(text)

# Regex patterns for detecting citation/reference lines
_CITATION_PATTERNS = re.compile(
    r'\[\d{1,3}\]'           # [1], [23], [100]
    r'|\b\d{4}[a-z]?\)'     # 2023a), 2024)
    r'|\bet\s+al\.?'         # et al.
    r'|arXiv:\d'             # arXiv:2301
    r'|\bdoi[:/]'            # doi:10. or doi/
    r'|\bpp\.\s*\d'          # pp. 123
    r'|\bvol\.\s*\d'         # vol. 5
    r'|\bISSN\b|\bISBN\b',
    re.IGNORECASE
)

def is_reference_block(text: str, threshold: float = 0.30) -> bool:
    """Detect if a block is predominantly references/citations.
    
    Returns True if more than `threshold` fraction of non-empty lines
    contain citation patterns (e.g., [1], et al., arXiv:, doi:).
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    citation_lines = sum(1 for l in lines if _CITATION_PATTERNS.search(l))
    ratio = citation_lines / len(lines)
    return ratio > threshold

def validate_queries(queries, block_text):
    if len(queries) != 4:
        raise ValueError(f"Expected exactly 4 queries, got {len(queries)}")
    required_keys = {"query_group", "query_type", "question", "golden_answer", "evidence_quotes"}
    for q in queries:
        missing = required_keys - set(q.keys())
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        for quote in q.get("evidence_quotes", []):
            if quote not in block_text:
                print(f"    [WARN] Evidence quote not found verbatim in text: '{quote[:30]}...'")

def call_deepseek_streaming(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Helper to call DeepSeek-v4-flash."""
    print(f"  -> Calling DeepSeek-v4-flash API (temp={temperature})...", flush=True)
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        reasoning_effort="high",
        extra_body={
            "thinking": {
                "type": "enabled"
            }
        },
        stream=False
    )
    
    print("\n  -> Generating JSON...", flush=True)
    return response.choices[0].message.content

def process_chunk_file(file_path: str, output_dir: str, limit_blocks: int = None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    doc_id = data.get("doc_id", "unknown_doc")
    parent_blocks = data.get("parent_blocks", [])
    
    if not parent_blocks:
        print(f"[SKIP] No parent blocks found in {file_path}")
        return
        
    out_path = os.path.join(output_dir, f"{doc_id}_queries.json")
    all_generated_queries = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                all_generated_queries = json.load(f)
            print(f"  [INFO] Resuming {doc_id}. Found {len(all_generated_queries)} existing queries.")
        except Exception:
            pass

    if limit_blocks and len(parent_blocks) > limit_blocks:
        import random
        random.seed(doc_id)
        parent_blocks = random.sample(parent_blocks, limit_blocks)
        random.seed() # reset to default state

    processed_parent_ids = {q["parent_id"] for q in all_generated_queries}

    print(f"\n{'='*50}")
    print(f"[INFO] Processing Document: {doc_id} ({len(parent_blocks)} parent blocks)")
    
    
    for idx, block in enumerate(parent_blocks):
        parent_id = block["parent_id"]
        text = block["text"]
        
        if parent_id in processed_parent_ids:
            print(f"  [{idx+1}/{len(parent_blocks)}] Skipping already processed block: {parent_id}")
            continue
        
        if is_reference_block(text):
            print(f"  [{idx+1}/{len(parent_blocks)}] Skipping reference/citation block: {parent_id}")
            continue
            
        print(f"  [{idx+1}/{len(parent_blocks)}] Generating queries for block: {parent_id}")
        user_prompt = f"Here is the Parent Block text:\n\n{text}"
        
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                resp_text = call_deepseek_streaming(SYSTEM_PROMPT, user_prompt, temperature=0.7)
                t1 = time.time()
                
                queries = extract_json(resp_text)
                
                if not isinstance(queries, list):
                    raise ValueError("LLM did not return a JSON array.")
                    
                validate_queries(queries, text)
                    
                for q in queries:
                    q["query_id"] = f"q_{uuid.uuid4().hex[:8]}"
                    q["doc_id"] = doc_id
                    q["parent_id"] = parent_id
                    all_generated_queries.append(q)
                    
                print(f"  -> Success in {t1-t0:.1f}s. Generated {len(queries)} queries.")
                
                # Incremental Save
                with open(out_path, "w", encoding='utf-8') as f:
                    json.dump(all_generated_queries, f, indent=2, ensure_ascii=False)
                    
                success = True
                break
                
            except Exception as e:
                delay = 2 ** (attempt + 1)
                print(f"  [ERROR] Attempt {attempt+1}/{max_retries} failed for {parent_id}: {e}", flush=True)
                if attempt < max_retries - 1:
                    print(f"    -> Retrying in {delay}s...")
                    time.sleep(delay)
                    
        if not success:
            print(f"  [FATAL] Skipping {parent_id} after {max_retries} failed attempts.")
            
    print(f"[SUCCESS] Saved total of {len(all_generated_queries)} queries for {doc_id} to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Seed Query Generation")
    parser.add_argument("--input-dir", type=str, default="../../data/processed/benchmark_chunks", 
                        help="Directory containing output JSON from Step 1")
    parser.add_argument("--output-dir", type=str, default="../../data/processed/benchmark_seeds", 
                        help="Directory to save the generated query JSON files")
    parser.add_argument("--limit-blocks", type=int, default=None,
                        help="Limit the number of parent blocks to process per document (for testing)")
    
    args = parser.parse_args()
    
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, "*_chunks.json"))
    if not json_files:
        print(f"No JSON chunk files found in {input_dir}")
        return
        
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY environment variable not found.")
        print("Please ensure it is set in your .env file or exported in your shell.")
        
    print(f"Found {len(json_files)} document chunk files. Starting seed generation...", flush=True)
    
    for file_path in json_files:
        process_chunk_file(file_path, output_dir, limit_blocks=args.limit_blocks)
        
    print("\n--- Seed Generation Complete ---", flush=True)

if __name__ == "__main__":
    main()
