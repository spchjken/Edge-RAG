import os
import glob
import json
import argparse
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    tiktoken = None

class WordTokenizer:
    def encode(self, text, disallowed_special=()):
        return text.split()
    def decode(self, tokens):
        return " ".join(tokens)

def get_tokenizer(model_name: str = "cl100k_base"):
    """
    Returns a tokenizer instance. 
    Uses tiktoken's cl100k_base (used by GPT-4) as a fast standard.
    """
    if tiktoken is None or model_name == "word":
        print(f"[INFO] Using fallback WordTokenizer (model_name={model_name}, tiktoken_installed={tiktoken is not None})")
        return WordTokenizer()
    return tiktoken.get_encoding(model_name)

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, tokenizer) -> List[str]:
    """
    Chunks text into pieces of `chunk_size` tokens with `chunk_overlap`.
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    
    if not tokens:
        return chunks
        
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunks.append(tokenizer.decode(chunk_tokens))
        
        if end_idx == len(tokens):
            break
            
        start_idx += (chunk_size - chunk_overlap)
        
    return chunks

def process_document(
    text: str, 
    doc_id: str, 
    parent_chunk_size: int = 3000, 
    child_chunk_size: int = 1000, 
    child_overlap: int = 100,
    tokenizer=None
) -> Dict[str, Any]:
    """
    Processes a document into non-overlapping parent blocks and overlapping child chunks.
    Child chunks are strictly derived from their parent blocks.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    # 1. Split into Parent Blocks (Non-Overlapping)
    # Using chunk_overlap=0 to ensure they are disjoint, avoiding zero token-overlap waste for seeds
    parent_texts = chunk_text(text, chunk_size=parent_chunk_size, chunk_overlap=0, tokenizer=tokenizer)
    
    parent_blocks = []
    all_child_chunks = []
    
    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{doc_id}_block{p_idx}"
        
        parent_blocks.append({
            "parent_id": parent_id,
            "doc_id": doc_id,
            "text": p_text,
            "token_count": count_tokens(p_text, tokenizer)
        })
        
        # 2. Split into Child Chunks (Overlapping), derived strictly from the parent block
        # This guarantees that each child chunk has a single parent_id linking it back.
        c_texts = chunk_text(p_text, chunk_size=child_chunk_size, chunk_overlap=child_overlap, tokenizer=tokenizer)
        
        for c_idx, c_text in enumerate(c_texts):
            child_id = f"{parent_id}_chunk{c_idx}"
            all_child_chunks.append({
                "chunk_id": child_id,
                "parent_id": parent_id,
                "doc_id": doc_id,
                "text": c_text,
                "token_count": count_tokens(c_text, tokenizer)
            })
            
    # 3. Create Bridge Chunks (D1.2)
    # To handle information spanning parent blocks, create overlapping bridge chunks
    for i in range(len(parent_blocks) - 1):
        prev_parent = parent_blocks[i]
        next_parent = parent_blocks[i + 1]
        
        prev_tokens = tokenizer.encode(prev_parent["text"], disallowed_special=())
        next_tokens = tokenizer.encode(next_parent["text"], disallowed_special=())
        
        half_size = child_chunk_size // 2
        bridge_tokens = prev_tokens[-half_size:] + next_tokens[:half_size]
        bridge_text = tokenizer.decode(bridge_tokens)
        
        bridge_id = f"{doc_id}_bridge{i}-{i+1}"
        
        all_child_chunks.append({
            "chunk_id": bridge_id,
            "parent_id": bridge_id, # Synthetic parent ID
            "doc_id": doc_id,
            "text": bridge_text,
            "token_count": count_tokens(bridge_text, tokenizer)
        })
            
    return {
        "doc_id": doc_id,
        "parent_blocks": parent_blocks,
        "child_chunks": all_child_chunks
    }

def main():
    parser = argparse.ArgumentParser(description="Step 1: Document Parsing & Hierarchical Chunking")
    parser.add_argument("--input-dir", type=str, default="../../data/raw/latest_arxiv", 
                        help="Directory containing raw .txt files")
    parser.add_argument("--output-dir", type=str, default="../../data/processed/benchmark_chunks", 
                        help="Directory to save the chunked JSON files")
    parser.add_argument("--parent-size", type=int, default=3000, 
                        help="Token length for parent blocks (default: 3000)")
    parser.add_argument("--child-size", type=int, default=1000, 
                        help="Token length for child chunks (default: 1000)")
    parser.add_argument("--child-overlap", type=int, default=100, 
                        help="Token overlap for child chunks (default: 100)")
    parser.add_argument("--tokenizer", type=str, default="cl100k_base", 
                        help="Tokenizer model name (or 'word' for fallback)")
    
    args = parser.parse_args()
    
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print("Please ensure the directory exists and contains the raw text files.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
        
    tokenizer = get_tokenizer(args.tokenizer)
        
    print(f"Found {len(txt_files)} document(s). Starting hierarchical chunking...")
    print(f"Parameters: Parent={args.parent_size} tokens, Child={args.child_size} tokens (Overlap={args.child_overlap})")
    
    total_parents = 0
    total_children = 0
    
    for file_path in txt_files:
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        result = process_document(
            text=text, 
            doc_id=doc_id, 
            parent_chunk_size=args.parent_size, 
            child_chunk_size=args.child_size, 
            child_overlap=args.child_overlap,
            tokenizer=tokenizer
        )
        
        output_file = os.path.join(output_dir, f"{doc_id}_chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        p_count = len(result['parent_blocks'])
        c_count = len(result['child_chunks'])
        total_parents += p_count
        total_children += c_count
        
        print(f"[{doc_id}] Created {p_count} parent blocks and {c_count} child chunks -> Saved to {output_file}")
        
    print("\n--- Chunking Complete ---")
    print(f"Total Documents: {len(txt_files)}")
    print(f"Total Parent Blocks: {total_parents}")
    print(f"Total Child Chunks: {total_children}")

if __name__ == "__main__":
    main()
