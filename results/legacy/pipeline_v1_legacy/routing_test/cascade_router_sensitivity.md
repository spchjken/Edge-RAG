# Cascade Router Threshold Sensitivity

| Setting | Bypass | Rerank | Discard | Compression Ratio |
|---------|--------|--------|---------|-------------------|
| Strict | 0 | 1 | 4 | 0.12 |
| Balanced | 0 | 4 | 1 | 0.19 |
| Permissive | 1 | 3 | 1 | 0.19 |

**Conclusion**: The 'Balanced' setting is preferred as it safely discards irrelevant low-density chunks while keeping the compression ratio optimal. The 'Strict' setting drops too many chunks into Discard, risking recall, while 'Permissive' causes unnecessary redundancy by overloading the Rerank queue.
