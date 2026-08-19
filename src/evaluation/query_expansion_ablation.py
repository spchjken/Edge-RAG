import os
import csv
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def compute_aspect_coverage(generated_terms, gt_aspects):
    gen_tokens = set()
    for t in generated_terms:
        gen_tokens.update(t.lower().split())
        
    covered = 0
    for gt in gt_aspects:
        gt_tokens = set(gt.lower().split())
        if not gt_tokens: continue
        overlap = len(gt_tokens.intersection(gen_tokens)) / len(gt_tokens.union(gen_tokens))
        if overlap >= 0.5:
            covered += 1
            
    return covered / len(gt_aspects) if gt_aspects else 1.0

def compute_keyword_jaccard(generated_terms, gt_keywords):
    gen_tokens = set()
    for t in generated_terms:
        gen_tokens.update(t.lower().split())
        
    gt_tokens = set()
    for kw in gt_keywords:
        gt_tokens.update(kw.lower().split())
        
    if not gt_tokens and not gen_tokens: return 1.0
    if not gt_tokens or not gen_tokens: return 0.0
    return len(gen_tokens.intersection(gt_tokens)) / len(gen_tokens.union(gt_tokens))


def analyze_output(method_name, query, output, ground_truth_item, elapsed_time):
    num_aspects = 0
    num_keywords = 0
    all_terms = []
    
    if isinstance(output, dict) and "aspects" in output:
        num_aspects = len(output["aspects"])
        for asp in output["aspects"]:
            all_terms.append(asp.get("name", "").lower())
            for kw in asp.get("keywords", []):
                num_keywords += 1
                all_terms.append(kw.get("term", "").lower())
    elif isinstance(output, list):
        num_aspects = len(output)
        num_keywords = 0
        for asp in output:
            if isinstance(asp, dict):
                all_terms.append(asp.get("name", "").lower())
            elif isinstance(asp, str):
                all_terms.append(asp.lower())
            
    stopword_count = 0
    for term in all_terms:
        words = term.split()
        if any(w in ENGLISH_STOP_WORDS for w in words):
            stopword_count += 1
            
    acronyms_in_query = [word for word in query.split() if word.isupper() and len(word) > 1]
    retained_acronyms = 0
    for acr in acronyms_in_query:
        if any(acr.lower() in term for term in all_terms):
            retained_acronyms += 1
            
    acronym_retention_rate = 1.0
    if len(acronyms_in_query) > 0:
        acronym_retention_rate = retained_acronyms / len(acronyms_in_query)
        
    # Evaluate against ground truth
    gt_aspects = [a.get("name", "") for a in ground_truth_item.get("detached_aspect_only", {}).get("aspects", [])]
    gt_keywords = []
    for a in ground_truth_item.get("detached_aspect_term", {}).get("aspects", []):
        gt_keywords.append(a.get("name", ""))
        for k in a.get("keywords", []):
            gt_keywords.append(k.get("term", ""))
            
    aspect_coverage = compute_aspect_coverage(all_terms, gt_aspects)
    keyword_jaccard = compute_keyword_jaccard(all_terms, gt_keywords)
        
    return {
        "query": query,
        "method": method_name,
        "latency": elapsed_time,
        "num_aspects": num_aspects,
        "num_keywords": num_keywords,
        "stopword_count": stopword_count,
        "acronym_retention": acronym_retention_rate,
        "aspect_coverage": aspect_coverage,
        "keyword_jaccard": keyword_jaccard,
        "raw_output": json.dumps(output)
    }

def main():
    print("Loading Ground Truth Dataset...")
    queries_dir = "data/processed/generated_queries_fintech"
    dataset = {}
    if os.path.exists(queries_dir):
        for fname in sorted(os.listdir(queries_dir)):
            if fname.endswith("_queries.json"):
                with open(os.path.join(queries_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        dataset[item["question"]] = item
                    
    if not dataset:
        print("No datasets found! Run generate_fintech_dataset.py first.")
        return
        
    print(f"Loaded {len(dataset)} unique queries for evaluation.")
    
    results = []
    methods = ["llm", "llm_aspect_only", "statistical", "vector", "yake_statistical", "yake_vector"]
    
    base_dir = "results/pipeline_test/query_expansion"
    
    print("Analyzing outputs from previous test runs...")
    
    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            continue
            
        # Find latest json file
        json_files = [f for f in os.listdir(method_dir) if f.endswith(".json")]
        if not json_files:
            continue
            
        json_files.sort()
        latest_file = os.path.join(method_dir, json_files[-1])
        
        with open(latest_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)
            
        print(f"Loaded {len(run_data.get('tests', []))} outputs for {method} from {json_files[-1]}")
        
        for test in run_data.get("tests", []):
            query = test.get("query")
            if query not in dataset:
                continue
                
            gt_item = dataset[query]
            latency = test.get("latency_sec", 0.0)
            output = test.get("output", {})
            
            # Use appropriate method name mapping
            method_display = method
            
            res = analyze_output(method_display, query, output, gt_item, latency)
            results.append(res)
            
    if not results:
        print("No results to save.")
        return
        
    # Save CSV
    csv_file = "results/pipeline_test/query_expansion/query_expansion_ablation.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "method", "latency", "num_aspects", "num_keywords", 
            "stopword_count", "acronym_retention", "aspect_coverage", "keyword_jaccard", "raw_output"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved ablation results to {csv_file}")
    
    # Calculate averages
    averages = {}
    found_methods = set(r["method"] for r in results)
    
    for name in found_methods:
        method_results = [r for r in results if r["method"] == name]
        if not method_results: continue
        avg_lat = sum(r["latency"] for r in method_results) / len(method_results)
        avg_cov = sum(r["aspect_coverage"] for r in method_results) / len(method_results)
        avg_jac = sum(r["keyword_jaccard"] for r in method_results) / len(method_results)
        avg_acr = sum(r["acronym_retention"] for r in method_results) / len(method_results)
        avg_stp = sum(r["stopword_count"] for r in method_results) / len(method_results)
        averages[name] = {
            "latency": avg_lat,
            "coverage": avg_cov,
            "jaccard": avg_jac,
            "acronym_ret": avg_acr,
            "stopword_cnt": avg_stp
        }
    
    # Generate Summary Markdown
    md_file = "results/pipeline_test/query_expansion/query_expansion_ablation_summary.md"
    with open(md_file, "w") as f:
        f.write("# Query Expansion Ablation Summary (Fintech Dataset)\n\n")
        f.write(f"Evaluated on {len(dataset)} complex, multi-hop queries from the fintech ground-truth dataset.\n\n")
        f.write("## Empirical Performance\n")
        f.write("| Method | Avg Latency (s) | Aspect Coverage | Keyword Jaccard | Acronym Retention | Stopword Count |\n")
        f.write("|--------|-----------------|-----------------|-----------------|-------------------|----------------|\n")
        for name, avg in averages.items():
            f.write(f"| {name} | {avg['latency']:.4f} | {avg['coverage']:.2%} | {avg['jaccard']:.2%} | {avg['acronym_ret']:.2%} | {avg['stopword_cnt']:.2f} |\n")
            
        f.write("\n## Metric Definitions\n")
        f.write("- **Aspect Coverage**: Percentage of `detached_aspect` ground truth concepts captured.\n")
        f.write("- **Keyword Jaccard**: Strict token overlap with ground truth synonyms (lower is normal, high means precise match).\n")
        f.write("- **Acronym Retention**: Percentage of acronyms in the query that were preserved in the output.\n")

    print(f"Saved summary to {md_file}")

if __name__ == "__main__":
    main()
