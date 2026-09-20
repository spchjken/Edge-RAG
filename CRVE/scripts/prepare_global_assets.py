import os
import json
import math
from collections import Counter
import fasttext
import numpy as np
import urllib.request
import zipfile
import shutil

# Try to import BGE and FAISS for vector pathway assets
try:
    from FlagEmbedding import FlagModel
    import faiss
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False

def prepare_global_assets():
    print("Preparing GLOBAL test assets (Simulating production environment)...")
    
    os.makedirs("models/global", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    text8_zip = "data/raw/text8.zip"
    text8_file = "data/raw/text8"
    
    # 1. Download global corpus (text8 - 100MB of clean Wikipedia text)
    if not os.path.exists(text8_file):
        if not os.path.exists(text8_zip):
            print("Downloading text8 global corpus (31MB)...")
            urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", text8_zip)
        
        print("Extracting text8...")
        with zipfile.ZipFile(text8_zip, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
    
    # Read a portion of text8 for vocab/IDF to keep it fast, but enough to be a generic english baseline
    # Text8 is a single line of text.
    print("Processing global text...")
    with open(text8_file, "r") as f:
        text = f.read(50000000) # Read 50MB of the 100MB file
        
    words = text.split()
    
    # We pretend each 100 words is a "document" for IDF purposes
    chunk_size = 100
    global_docs = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Generated {len(global_docs)} generic global documents for IDF computation.")
    
    # 2. Compute Global IDF Dictionary
    print("Computing Global IDF...")
    df = Counter()
    for doc in global_docs:
        doc_words = set(doc.split())
        for w in doc_words:
            df[w] += 1
            
    N = len(global_docs)
    idf_dict = {}
    for word, count in df.items():
        if count >= 5:  # Only keep words that appear at least a few times globally
            idf_dict[word] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)
            
    with open("models/global/global_idf_dict.json", "w", encoding="utf-8") as f:
        json.dump(idf_dict, f)
    print(f"Saved Global IDF dictionary with {len(idf_dict)} common English words.")
    
    # 3. Train Global FastText model
    fasttext_path = "models/global/global_fasttext.bin"
    if not os.path.exists(fasttext_path):
        print("Training Global FastText model on Wikipedia text (this takes ~30 seconds)...")
        # Train unsupervised CBOW model
        model = fasttext.train_unsupervised(text8_file, model='cbow', dim=50, epoch=5, minCount=5)
        model.save_model(fasttext_path)
        print(f"Saved Global FastText model to {fasttext_path}")
    else:
        print(f"Global FastText model already exists at {fasttext_path}")
        
    # 4. Build Global FAISS Index (Optional, for Vector Pathway)
    if HAS_VECTOR_DEPS:
        faiss_index_path = "models/global/global_faiss.index"
        if not os.path.exists(faiss_index_path):
            print("Building Global FAISS index...")
            bge = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=True)
            
            # Use top 20,000 global words for the FAISS index to keep it fast
            word_counts = Counter(words)
            top_words = [w for w, c in word_counts.most_common(20000)]
            
            print(f"Encoding top {len(top_words)} global English words for FAISS...")
            batch_size = 512
            embeddings = []
            for i in range(0, len(top_words), batch_size):
                batch = top_words[i:i+batch_size]
                embeddings.append(bge.encode(batch))
                
            all_embeddings = np.vstack(embeddings).astype('float32')
            faiss.normalize_L2(all_embeddings)
            
            dim = all_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(all_embeddings)
            
            faiss.write_index(index, faiss_index_path)
            with open("models/global/global_faiss_vocab.json", "w", encoding="utf-8") as f:
                json.dump(top_words, f)
            print("Saved Global FAISS index and vocabulary.")
        else:
            print(f"Global FAISS index already exists at {faiss_index_path}")
    
    print("Global asset preparation complete.")

if __name__ == "__main__":
    prepare_global_assets()
