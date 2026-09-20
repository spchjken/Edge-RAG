import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_enterpriserag():
    q_file = os.path.join(base_dir, "data", "raw", "enterpriserag_bench", "data", "questions", "test.parquet")
    d_file = os.path.join(base_dir, "data", "raw", "enterpriserag_bench", "data", "documents", "test.parquet")
    
    try:
        import pandas as pd
        if os.path.exists(q_file):
            df_q = pd.read_parquet(q_file)
            print(f"[EnterpriseRAG] Total Questions: {len(df_q)}")
            print(f"[EnterpriseRAG] Columns in questions: {list(df_q.columns)}")
        if os.path.exists(d_file):
            df_d = pd.read_parquet(d_file)
            print(f"[EnterpriseRAG] Total Documents: {len(df_d)}")
    except Exception as e:
        print(f"Error reading EnterpriseRAG parquet: {e}")

def count_liverag():
    p_file = os.path.join(base_dir, "data", "raw", "liverag_bench", "LiveRAG_banchmark_20250910.parquet")
    try:
        import pandas as pd
        if os.path.exists(p_file):
            df_l = pd.read_parquet(p_file)
            print(f"[LiveRAG] Total Queries/Records: {len(df_l)}")
            print(f"[LiveRAG] Columns: {list(df_l.columns)}")
    except Exception as e:
        print(f"Error reading LiveRAG parquet: {e}")

if __name__ == "__main__":
    count_enterpriserag()
    count_liverag()
