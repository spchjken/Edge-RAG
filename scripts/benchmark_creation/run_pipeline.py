import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()

def run_step(step_name: str, script_path: str, args: list):
    """Helper to run a python script via subprocess and stream output."""
    print(f"\n{'-'*60}")
    print(f"🚀 STARTING: {step_name}")
    print(f"   Command: python {script_path} {' '.join(args)}")
    print(f"{'-'*60}\n")
    
    result = subprocess.run([sys.executable, script_path] + args)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {step_name} failed with exit code {result.returncode}.")
        print("Pipeline aborted for this corpus.")
        sys.exit(result.returncode)
        
    print(f"\n✅ SUCCESS: {step_name} completed.")

def select_papers(input_dir: str, total_needed: int = 35) -> list:
    """Select papers of moderate size deterministically."""
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    valid_files = []
    for f in txt_files:
        size = os.path.getsize(os.path.join(input_dir, f))
        if 40000 <= size <= 150000: # ~40KB to 150KB
            valid_files.append(f)
            
    valid_files.sort()
    
    if len(valid_files) < total_needed:
        print(f"[WARN] Only found {len(valid_files)} moderate-sized papers. Falling back to all sizes.")
        txt_files.sort()
        return txt_files[:total_needed]
        
    return valid_files[:total_needed]

def setup_corpus_staging(base_out_dir: str, corpus_name: str, selected_txt_files: list, input_dir: str) -> str:
    """Creates an isolated staging directory containing symlinks to the chosen papers."""
    staging_dir = os.path.join(base_out_dir, corpus_name, "input_staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    for txt_file in selected_txt_files:
        src_txt = os.path.join(input_dir, txt_file)
        dst_txt = os.path.join(staging_dir, txt_file)
        if not os.path.exists(dst_txt):
            os.symlink(src_txt, dst_txt)
            
        pdf_file = txt_file.replace('.txt', '.pdf')
        src_pdf = os.path.join(input_dir, pdf_file)
        dst_pdf = os.path.join(staging_dir, pdf_file)
        if os.path.exists(src_pdf) and not os.path.exists(dst_pdf):
            os.symlink(src_pdf, dst_pdf)
            
    return staging_dir

def run_pipeline_for_corpus(corpus_name: str, staging_dir: str, output_dir: str, scripts_dir: str, limit_blocks=None):
    """Runs the 5-step pipeline for a single isolated corpus."""
    print(f"\n{'='*70}")
    print(f"🌟 RUNNING PIPELINE FOR CORPUS: {corpus_name} 🌟")
    print(f"{'='*70}")

    step1_chunks_dir = os.path.join(output_dir, "step1_chunks")
    step2_seeds_dir = os.path.join(output_dir, "step2_seeds")
    step3_paraphrased_dir = os.path.join(output_dir, "step3_paraphrased")
    step4_candidates_file = os.path.join(output_dir, "step4_candidates.json")
    final_benchmark_file = os.path.join(output_dir, f"final_benchmark_{corpus_name}.json")

    # Step 1
    run_step("Step 1: Document Parsing", os.path.join(scripts_dir, "step1_chunking.py"),
             ["--input-dir", staging_dir, "--output-dir", step1_chunks_dir])

    # Step 2
    step2_args = ["--input-dir", step1_chunks_dir, "--output-dir", step2_seeds_dir]
    if limit_blocks is not None:
        step2_args.extend(["--limit-blocks", str(limit_blocks)])
    run_step("Step 2: Seed Query Generation", os.path.join(scripts_dir, "step2_seed_generation.py"), step2_args)

    # Step 3
    run_step("Step 3: Query Paraphrasing", os.path.join(scripts_dir, "step3_query_paraphrasing.py"),
             ["--input-dir", step2_seeds_dir, "--output-dir", step3_paraphrased_dir])

    # Step 4
    run_step("Step 4: Offline Global Recall", os.path.join(scripts_dir, "step4_global_recall.py"),
             ["--processed-dir", step1_chunks_dir, "--paraphrased-dir", step3_paraphrased_dir, "--output-file", step4_candidates_file])

    # Step 5
    run_step("Step 5: Oracle Filtering", os.path.join(scripts_dir, "step5_oracle_filtering.py"),
             ["--input-file", step4_candidates_file, "--output-file", final_benchmark_file])

def main():
    parser = argparse.ArgumentParser(description="Multi-Corpus Edge-RAG Orchestrator")
    parser.add_argument("--input-dir", type=str, default="data/raw/latest_arxiv/ai", 
                        help="Path to the raw paper directory.")
    parser.add_argument("--output-dir", type=str, default="data/benchmarks", 
                        help="Path to store benchmark corpora.")
    parser.add_argument("--limit-blocks", type=int, default=5,
                        help="Limit blocks per document to cap API costs (default: 5).")
    parser.add_argument("--stress-only", action="store_true", default=True,
                        help="Only generate corpus_stress_1 (skips single and multi corpora).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.abspath(os.path.join(project_root, args.input_dir))
    output_base_dir = os.path.abspath(os.path.join(project_root, args.output_dir))
    scripts_dir = os.path.join(project_root, "scripts", "benchmark_creation")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY environment variable not found.")
        print("Steps 2, 3, and 5 require the API key. Ensure it is exported.")

    if args.stress_only:
        corpora_config = [("corpus_stress_1", 15)]
        total_needed = 15
    else:
        corpora_config = [
            ("corpus_single_1", 1), ("corpus_single_2", 1), ("corpus_single_3", 1),
            ("corpus_single_4", 1), ("corpus_single_5", 1),
            ("corpus_multi_1", 5), ("corpus_multi_2", 5), ("corpus_multi_3", 5),
            ("corpus_stress_1", 15)
        ]
        total_needed = 35

    # 1. Select needed papers
    selected_papers = select_papers(input_dir, total_needed=total_needed)
    if len(selected_papers) < total_needed:
        print(f"Error: Only found {len(selected_papers)} papers in {input_dir}. Need {total_needed}.")
        sys.exit(1)

    # 3. Iterate and run
    paper_idx = 0
    for corpus_name, count in corpora_config:
        corpus_files = selected_papers[paper_idx : paper_idx + count]
        paper_idx += count
        
        corpus_out_dir = os.path.join(output_base_dir, corpus_name)
        staging_dir = setup_corpus_staging(output_base_dir, corpus_name, corpus_files, input_dir)
        
        run_pipeline_for_corpus(corpus_name, staging_dir, corpus_out_dir, scripts_dir, limit_blocks=args.limit_blocks)

    print("\n🎉 ALL CORPORA BENCHMARKS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
