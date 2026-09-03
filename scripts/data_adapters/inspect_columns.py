import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.data_adapters.convert_external_benchmarks import read_arrow_file

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("--- LiveRAG Columns & Sample ---")
    liverag_files = glob.glob(os.path.join(base_dir, "data", "raw", "liverag", "**", "*.arrow"), recursive=True)
    if liverag_files:
        records = read_arrow_file(liverag_files[0])
        if records:
            print("Total records:", len(records))
            print("Keys:", list(records[0].keys()))
            print("Sample record[0]:", {k: (str(v)[:100] if isinstance(v, str) else type(v)) for k, v in records[0].items()})
            
    print("\n--- EnterpriseRAG Documents Columns & Sample ---")
    ent_doc_files = glob.glob(os.path.join(base_dir, "data", "raw", "enterpriserag-bench", "**", "documents", "**", "*.arrow"), recursive=True)
    if ent_doc_files:
        doc_records = read_arrow_file(ent_doc_files[0])
        if doc_records:
            print("Total doc records:", len(doc_records))
            print("Keys:", list(doc_records[0].keys()))
            print("Sample doc[0]:", {k: (str(v)[:100] if isinstance(v, str) else type(v)) for k, v in doc_records[0].items()})

    print("\n--- EnterpriseRAG Questions Columns & Sample ---")
    ent_q_files = glob.glob(os.path.join(base_dir, "data", "raw", "enterpriserag-bench", "**", "questions", "**", "*.arrow"), recursive=True)
    if ent_q_files:
        q_records = read_arrow_file(ent_q_files[0])
        if q_records:
            print("Total question records:", len(q_records))
            print("Keys:", list(q_records[0].keys()))
            print("Sample q[0]:", {k: (str(v)[:100] if isinstance(v, str) else type(v)) for k, v in q_records[0].items()})

if __name__ == "__main__":
    main()
