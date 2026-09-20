import os
import json
import math
from collections import Counter
import fasttext
import numpy as np

# Try to import BGE and FAISS for vector pathway assets
try:
    from FlagEmbedding import FlagModel
    import faiss
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False

def compute_idf(corpus):
    N = len(corpus)
    df = Counter()
    for doc in corpus:
        # Simple tokenization
        words = set([w.lower() for w in doc.split() if w.isalpha()])
        for w in words:
            df[w] += 1
            
    idf_dict = {}
    for word, count in df.items():
        # Standard IDF formula
        idf_dict[word] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)
    return idf_dict

def prepare_assets():
    print("Preparing test assets from Adobe 10-K document...")
    
    # 1. Load the document
    doc_path = "data/raw/10-K_2026/10-K_2026/2026-01-15_0000796343-26-000003/10-K.md"
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found at {doc_path}")
        
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Split into paragraphs to form the corpus
    corpus = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    print(f"Loaded {len(corpus)} paragraphs.")
    
    # Create assets directory
    os.makedirs("tests/assets", exist_ok=True)
    
    # Save corpus
    with open("tests/assets/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f)
        
    # 2. Compute and save IDF dictionary
    idf_dict = compute_idf(corpus)
    with open("tests/assets/idf_dict.json", "w", encoding="utf-8") as f:
        json.dump(idf_dict, f)
    print(f"Saved IDF dictionary with {len(idf_dict)} words.")
    
    # 3. Train FastText model
    # FastText requires input as a text file with one document per line
    train_file = "tests/assets/fasttext_train.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(doc.replace('\n', ' ') + '\n')
            
    print("Training lightweight FastText model on the 10-K corpus...")
    # Train unsupervised (CBOW or skipgram). 
    # Use dim=50 for speed, minCount=2 to capture domain terms.
    model = fasttext.train_unsupervised(train_file, model='cbow', dim=50, epoch=10, minCount=2)
    model.save_model("tests/assets/fasttext_10k.bin")
    print("Saved FastText model to tests/assets/fasttext_10k.bin")
    
    # Cleanup train file
    os.remove(train_file)
    
    # 4. Build FAISS Index for Vector Pathway (if dependencies available)
    if HAS_VECTOR_DEPS:
        print("Building FAISS index for Vector Pathway...")
        bge = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=True)
        
        # Build vocabulary from IDF dict (filter out extremely rare words to save time, keep those with count > 2)
        # Let's just use all words in idf_dict that appear at least 3 times
        # Since IDF doesn't store counts, we'll re-calculate frequency simply
        word_counts = Counter()
        for doc in corpus:
            for w in doc.lower().split():
                if w.isalpha():
                    word_counts[w] += 1
                    
        vocab = [w for w, c in word_counts.items() if c >= 2]
        print(f"Encoding {len(vocab)} vocabulary words for FAISS...")
        
        # Encode in batches to avoid OOM
        batch_size = 512
        embeddings = []
        for i in range(0, len(vocab), batch_size):
            batch = vocab[i:i+batch_size]
            emb = bge.encode(batch)
            embeddings.append(emb)
            
        all_embeddings = np.vstack(embeddings).astype('float32')
        
        # Normalize for inner product (cosine similarity)
        faiss.normalize_L2(all_embeddings)
        
        dim = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_embeddings)
        
        faiss.write_index(index, "tests/assets/faiss_10k.index")
        with open("tests/assets/faiss_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f)
            
        print("Saved FAISS index and vocabulary.")
    else:
        print("Skipping FAISS index generation (dependencies not found).")
        
    print("Asset preparation complete.")

if __name__ == "__main__":
    prepare_assets()
