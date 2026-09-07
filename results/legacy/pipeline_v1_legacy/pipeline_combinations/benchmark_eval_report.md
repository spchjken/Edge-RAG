# Final Benchmark Evaluation
| Combination | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Avg VRAM |
|-------------|---------------|------------|-----------|-------------|----------------|-------------|----------|
| SimilarKW_Statistical_IDF_Adaptive | 100.0% | 100.0% | 2.7% | 30 | 0 | 9.87s | 0.09 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive | 90.0% | 90.0% | 17.2% | 14 | 11 | 5.44s | 0.09 GB |
| AspectOnly_Dense_Vocab_V3_Cascade | 90.0% | 90.0% | 18.5% | 4 | 21 | 9.09s | 0.09 GB |
| SimilarKW_Statistical_IDF_Cascade | 90.0% | 90.0% | 15.4% | 8 | 17 | 11.06s | 0.09 GB |
| AspectOnly_Dense_Vocab_V1_Cascade | 85.0% | 85.0% | 23.7% | 0 | 23 | 5.88s | 0.09 GB |
| AspectOnly_Dense_Vocab_V1_Adaptive | 80.0% | 80.0% | 25.3% | 3 | 18 | 3.48s | 0.09 GB |
| AspectOnly_Dense_Vocab_V2_Cascade | 75.0% | 75.0% | 24.1% | 0 | 20 | 7.92s | 0.09 GB |
| AspectOnly_Dense_Vocab_V2_Adaptive | 60.0% | 60.0% | 23.3% | 3 | 14 | 2.77s | 0.09 GB |
| SimilarKW_Vector_Projection_Cascade | 50.0% | 50.0% | 31.7% | 0 | 13 | 3.63s | 0.16 GB |
| SimilarKW_LLM_Query_Expander_Adaptive | 40.0% | 50.0% | 27.0% | 2 | 8 | 5.31s | 0.09 GB |
| SimilarKW_LLM_Query_Expander_Cascade | 45.0% | 50.0% | 30.8% | 5 | 7 | 6.21s | 0.09 GB |
| SimilarKW_Vector_Projection_Adaptive | 35.0% | 40.0% | 28.6% | 0 | 8 | 1.65s | 0.16 GB |
| AspectOnly_LLM_Aspect_Adaptive | 35.0% | 35.0% | 20.9% | 5 | 4 | 2.73s | 0.09 GB |
| AspectOnly_LLM_Aspect_Cascade | 35.0% | 35.0% | 20.0% | 4 | 5 | 3.00s | 0.09 GB |
| AspectOnly_YAKE_Cascade | 15.0% | 20.0% | 40.0% | 0 | 6 | 0.70s | 0.09 GB |
| AspectOnly_YAKE_Adaptive | 5.0% | 5.0% | 66.7% | 0 | 2 | 0.52s | 0.09 GB |
