import os
import json
import time
import random
import argparse
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """\
You are an expert RAG evaluation judge. Your task is to determine if a given text chunk contains sufficient evidence to answer a specific question.

QUESTION: {raw_question}
GOLDEN ANSWER: {golden_answer}

### INSTRUCTIONS
Below are {N} candidate text chunks, each identified by a chunk_id. 
For each chunk, evaluate whether it contains enough information to answer the QUESTION. It does not need to contain the exact wording of the GOLDEN ANSWER, but it must contain the factual evidence required to deduce it.

### FEW-SHOT EXAMPLES
**Example Question**: "What is the capital of France?"
**Example Golden Answer**: "Paris is the capital of France."
- **Chunk A**: "France is a country in Europe. It is known for its wine and cheese. The Eiffel Tower is located in Paris, which serves as the nation's capital." 
  *Reasoning*: Contains the exact fact. -> `true`
- **Chunk B**: "Paris is a major European city. Many tourists visit the Louvre museum there every year." 
  *Reasoning*: Tangential. Mentions Paris but does not state it is the capital of France. -> `false`
- **Chunk C**: "The capital of France is a bustling metropolis situated on the Seine river."
  *Reasoning*: Insufficient. Mentions it has a capital but doesn't name it. -> `false`

Format your output EXACTLY as a JSON dictionary mapping the chunk_id to a boolean (true if it contains sufficient evidence, false otherwise).

{{
  "chunk_1_id": false,
  "chunk_2_id": true
}}
Return ONLY the raw JSON dictionary.
"""

import re

def extract_json(text: str) -> Any:
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

def evaluate_chunk_batch(query_text: str, golden_answer: str, evidence_quotes: list, chunks: list) -> list:
    """Evaluate a batch of chunks against a single query."""
    user_prompt = f"Query: {query_text}\nGolden Answer: {golden_answer}\n"
    if evidence_quotes:
        user_prompt += f"Evidence Quotes from Ground Truth:\n"
        for idx, eq in enumerate(evidence_quotes):
            user_prompt += f"  {idx+1}. \"{eq}\"\n"
    user_prompt += "\n"
    
    for i, chunk in enumerate(chunks):
        user_prompt += f"--- Chunk {i} ---\n{chunk['text']}\n\n"
    
    user_prompt += "\nReturn a JSON list of booleans corresponding to the chunk indices."

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a judge. Output a JSON list of booleans [true, false, ...] for each chunk provided."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            stream=False,
            timeout=60.0
        )
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"    [ERROR] Batch evaluation failed: {e}")
        return [False] * len(chunks)

def process_batch(query_obj, chunks, batch_size=5):
    """
    Process chunks in batches, querying the Oracle LLM.
    Returns the list of chunks with an added 'is_relevant' boolean field.
    """
    shuffled_chunks = chunks[:]
    random.shuffle(shuffled_chunks)
    
    query_text = query_obj["paraphrased_question"]
    golden = query_obj["golden_answer"]
    evidence_quotes = query_obj.get("evidence_quotes", [])
    
    results_map = {}
    
    for i in range(0, len(shuffled_chunks), batch_size):
        batch = shuffled_chunks[i:i+batch_size]
        try:
            judgments = evaluate_chunk_batch(query_text, golden, evidence_quotes, batch)
            
            if not isinstance(judgments, list) or len(judgments) != len(batch):
                print(f"      [WARN] Oracle returned {len(judgments) if isinstance(judgments, list) else 'non-list'} results for {len(batch)} chunks. Defaulting to False.")
                judgments = [False] * len(batch)
                
            for j, chunk in enumerate(batch):
                is_relevant = judgments[j]
                
                # D5.1: Double Independent Judgment for TRUE
                if is_relevant is True:
                    second_pass = evaluate_chunk_batch(query_text, golden, evidence_quotes, [chunk])
                    if isinstance(second_pass, list) and len(second_pass) == 1:
                        if not second_pass[0]:
                            print(f"      [INFO] Second pass reversed True -> False for chunk {chunk['chunk_id']}")
                        is_relevant = is_relevant and second_pass[0]
                    else:
                        is_relevant = False
                        
                results_map[chunk["chunk_id"]] = is_relevant
                
        except Exception as e:
            print(f"      [ERROR] Batch evaluation failed: {e}")
            for chunk in batch:
                results_map[chunk["chunk_id"]] = False
                
    evaluated_chunks = []
    for c in chunks:
        c_copy = c.copy()
        c_copy["is_relevant"] = results_map.get(c["chunk_id"], False)
        evaluated_chunks.append(c_copy)
        
    return evaluated_chunks

def process_query(query: Dict[str, Any], batch_size: int = 20) -> Dict[str, Any]:
    print(f"  -> Processing query {query['query_id']}...")
    candidates = process_batch(query, query.get("retrieved_candidates", []), batch_size)
    
    ground_truth_child_chunks = []
    extended_child_chunks = []
    
    parent_source = query.get("parent_id_source", "")
    
    for candidate in candidates:
        if candidate.get("is_relevant", False):
            chunk_data = {
                "chunk_id": candidate["chunk_id"],
                "text": candidate["text"],
                "rank": candidate.get("rank")
            }
            
            if candidate.get("parent_id") == parent_source:
                ground_truth_child_chunks.append(chunk_data)
            else:
                extended_child_chunks.append(chunk_data)
            
    final_query = {
        "query_id": query["query_id"],
        "query_group": query.get("query_group", ""),
        "query_type": query.get("query_type", ""),
        "raw_question": query["raw_question"],
        "paraphrased_question": query["paraphrased_question"],
        "golden_answer": query["golden_answer"],
        "doc_id_source": query.get("doc_id_source", ""),
        "parent_id_source": parent_source,
        "ground_truth_child_chunks": ground_truth_child_chunks,
        "extended_child_chunks": extended_child_chunks
    }
    
    total_found = len(ground_truth_child_chunks) + len(extended_child_chunks)
    print(f"  -> Query {query['query_id']}: Found {total_found} relevant chunks ({len(ground_truth_child_chunks)} core, {len(extended_child_chunks)} extended)")
    return final_query

def main():
    parser = argparse.ArgumentParser(description="Step 5: Oracle Filtering (LLM-as-a-Judge)")
    parser.add_argument("--input-file", type=str, default="../../tmp_test_ai/retrieval_candidates.json", 
                        help="Path to retrieval candidates JSON")
    parser.add_argument("--output-file", type=str, default="../../tmp_test_ai/benchmark_dataset_final.json", 
                        help="Path to save the final benchmark JSON")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of chunks to send to the LLM per prompt")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(script_dir, args.input_file))
    output_file = os.path.abspath(os.path.join(script_dir, args.output_file))
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
        
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY environment variable not found.")
        return
        
    print(f"Loading candidates from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print(f"Found {len(queries)} queries to process.")
    
    final_dataset = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}]")
        final_query = process_query(query, args.batch_size)
        final_dataset.append(final_query)
        
        # Save incrementally to avoid losing data if it hangs
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Saved final benchmark dataset to {output_file}")
    
    total_relevant = sum(len(q["ground_truth_child_chunks"]) + len(q["extended_child_chunks"]) for q in final_dataset)
    print(f"Total Queries: {len(final_dataset)}")
    print(f"Total Relevant Chunks Found: {total_relevant}")

if __name__ == "__main__":
    main()
