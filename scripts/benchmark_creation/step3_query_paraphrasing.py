import os
import glob
import json
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """\
You are an expert linguist and evaluation dataset creator. Your task is to paraphrase a given question to create a realistic "Lexical Gap" for search engine testing.

You will be provided with:
1. Original Question
2. Golden Answer (for context, to ensure your paraphrased question still perfectly matches this answer)

Your Goal:
Rewrite the "Original Question" so that it has the exact same meaning and intent, but uses natural synonymous phrasing. Imagine a different user asking the exact same question in their own words. 
You should introduce a slight lexical gap (using different vocabulary or sentence structure where appropriate) to test information retrieval robustness, but DO NOT make it sound unnatural, forced, or overly complex.

Output ONLY the raw paraphrased string, nothing else. Do not use quotes or markdown formatting.
"""
def jaccard_similarity(str1: str, str2: str) -> float:
    import re
    set1 = set(re.findall(r'\w+', str1.lower()))
    set2 = set(re.findall(r'\w+', str2.lower()))
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def call_deepseek_paraphrase(original_question: str, golden_answer: str, temperature: float = 0.7) -> str:
    user_prompt = f"Original Question: {original_question}\nGolden Answer: {golden_answer}\n\nPlease provide the paraphrased question:"
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    raw = response.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    elif "<think>" in raw:
        pos = raw.find("<think>")
        end_pos = raw.find(">", pos)
        if end_pos != -1:
            raw = raw[end_pos+1:].strip()
    return raw

def process_file(file_path: str, output_dir: str):
    with open(file_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    if not queries:
        return
        
    doc_id = queries[0].get("doc_id", "unknown")
    out_path = os.path.join(output_dir, f"{doc_id}_paraphrased.json")
    
    if os.path.exists(out_path):
        print(f"[SKIP] Paraphrased queries already exist for {doc_id}.")
        return

    print(f"\n{'='*50}")
    print(f"[INFO] Processing Document: {doc_id} ({len(queries)} queries)")
    
    for idx, q in enumerate(queries):
        print(f"  [{idx+1}/{len(queries)}] Paraphrasing query {q['query_id']}...")
        
        # Safely extract question (D3.2)
        original_q = q.get("question")
        if not original_q:
            original_q = q.get("raw_question")
            
        golden_answer = q.get("golden_answer", "")
        
        try:
            t0 = time.time()
            paraphrased_q = call_deepseek_paraphrase(original_q, golden_answer)
            t1 = time.time()
            
            # Reconstruct the dict with the new fields
            q["raw_question"] = original_q
            q["paraphrased_question"] = paraphrased_q
            
            if "question" in q:
                del q["question"]
                
            # Lexical overlap (D3.1)
            overlap = jaccard_similarity(original_q, paraphrased_q)
            q["lexical_overlap"] = round(overlap, 3)
            
            print(f"  -> Success in {t1-t0:.1f}s. (Overlap: {overlap:.2f})")
            print(f"     Old: {original_q}")
            print(f"     New: {paraphrased_q}")
            
            if overlap > 0.8:
                print(f"     [WARN] Lexical overlap is very high ({overlap:.2f}). Paraphrase quality may be low.")
                q["paraphrase_quality"] = "low_gap"
            elif overlap < 0.2:
                print(f"     [WARN] Lexical overlap is very low ({overlap:.2f}). Meaning might have changed.")
                q["paraphrase_quality"] = "high_gap"
            else:
                q["paraphrase_quality"] = "good"
            
        except Exception as e:
            print(f"  [ERROR] Failed to paraphrase query {q['query_id']}: {e}", flush=True)
            q["raw_question"] = original_q
            q["paraphrased_question"] = original_q  # Fallback
            q["lexical_overlap"] = 1.0
            q["paraphrase_quality"] = "failed"
            if "question" in q:
                del q["question"]
            
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
        
    print(f"[SUCCESS] Saved {len(queries)} paraphrased queries to {out_path}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Step 3: Query Paraphrasing")
    parser.add_argument("--input-dir", type=str, default="../../tmp_test_ai/seeds", 
                        help="Directory containing output JSON from Step 2")
    parser.add_argument("--output-dir", type=str, default="../../tmp_test_ai/paraphrased", 
                        help="Directory to save the paraphrased JSON files")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_dir, "*_queries.json"))
    if not json_files:
        print(f"No JSON chunk files found in {input_dir}")
        return
        
    print(f"Found {len(json_files)} document query files. Starting paraphrasing...", flush=True)
    
    for file_path in json_files:
        process_file(file_path, output_dir)
        
    print("\n--- Query Paraphrasing Complete ---", flush=True)

if __name__ == "__main__":
    main()
