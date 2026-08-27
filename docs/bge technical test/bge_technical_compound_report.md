# 🧪 BGE Technical Compound Similarity Evaluation Report

- **Model Evaluated:** `BAAI/bge-small-en-v1.5`
- **Input Testset:** `docs/technical_compound_bge_testset.md`
- **Total Pairs Tested:** 6308

## 1. Summary Statistics & Pass Rates Across Thresholds

| Evaluation Group | Count | Mean ± Std | Median | τ ≥ 0.50 | τ ≥ 0.55 | τ ≥ 0.60 | τ ≥ 0.70 | τ ≥ 0.80 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (All Pairs)** | 6308 | 0.8439 ± 0.0263 | 0.8431 | 100.0% | 100.0% | 100.0% | 100.0% | 96.8% |
| **AI Models (Total)** | 931 | 0.8380 ± 0.0395 | 0.8329 | 100.0% | 100.0% | 100.0% | 100.0% | 86.4% |
| **  ↳ AI Models (Real-World)** | 432 | 0.8591 ± 0.0441 | 0.8675 | 100.0% | 100.0% | 100.0% | 100.0% | 91.7% |
| **  ↳ AI Models (Bielik Stress)** | 499 | 0.8197 ± 0.0223 | 0.8202 | 100.0% | 100.0% | 100.0% | 100.0% | 81.8% |
| **Frameworks & Libraries (Total)** | 5377 | 0.8450 ± 0.0231 | 0.8439 | 100.0% | 100.0% | 100.0% | 99.9% | 98.6% |
| **  ↳ Frameworks & Libraries (Real-World)** | 475 | 0.8654 ± 0.0463 | 0.8741 | 100.0% | 100.0% | 100.0% | 99.4% | 91.4% |
| **  ↳ Frameworks & Libraries (Lucene Stress)** | 4902 | 0.8430 ± 0.0182 | 0.8429 | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% |

## 2. Key Findings & Analysis

### Real-World Technical Pairs Analysis (N=907)
- **Mean Cosine Similarity:** `0.8624` (Median: `0.8710`)
- **Pass rate at standard Edge-RAG threshold (τ = 0.55):** **`100.0%`** (907/907 pairs)
- **Pass rate at high confidence threshold (τ = 0.70):** **`99.7%`** (904/907 pairs)
- **Pass rate at strict threshold (τ = 0.80):** **`91.5%`** (830/907 pairs)

### Architectural Implication for Edge-RAG Pipeline

> [!NOTE]
> **Option 1 Supported:** With a high pass rate (99.7% at τ=0.70, 100.0% at τ=0.55), BGE dense probing alone can reliably discover technical compounds from base terms without requiring explicit lexical boundary bailout rules.

## 3. Sample High & Low Similarity Pairs

### Top 20 Highest Similarity Pairs

| Category | Base Term | Compound Term | Cosine Similarity |
| :--- | :--- | :--- | :---: |
| `ai-models` | `wav2lip` | `wav2lip-2` | `0.9836` |
| `frameworks-libraries` | `xgrammar` | `xgrammar-0.1` | `0.9584` |
| `ai-models` | `gpt-image` | `gpt-image-2` | `0.9523` |
| `frameworks-libraries` | `huggingface` | `huggingface-hub` | `0.9510` |
| `frameworks-libraries` | `imagemagick` | `imagemagick-7.1` | `0.9508` |
| `frameworks-libraries` | `llama-cpp` | `llama-cpp-python` | `0.9492` |
| `ai-models` | `s2t` | `s2t-2` | `0.9488` |
| `frameworks-libraries` | `gstreamer` | `gstreamer-1.24` | `0.9477` |
| `ai-models` | `openhermes` | `openhermes-2.5` | `0.9452` |
| `ai-models` | `elevenlabs` | `elevenlabs-2` | `0.9445` |
| `frameworks-libraries` | `mxnet` | `mxnet-2.0` | `0.9441` |
| `ai-models` | `minicpm` | `minicpm-v` | `0.9432` |
| `ai-models` | `mixlab` | `mixlab-2` | `0.9418` |
| `frameworks-libraries` | `sentence-transformers` | `sentence-transformers-3` | `0.9377` |
| `ai-models` | `layoutlm` | `layoutlm-3` | `0.9375` |
| `frameworks-libraries` | `gstreamer` | `gstreamer-1.26` | `0.9356` |
| `frameworks-libraries` | `certmanager` | `cert-manager` | `0.9350` |
| `frameworks-libraries` | `cert-manager` | `cert-manager-1.16` | `0.9347` |
| `ai-models` | `nano-banana` | `nano-banana-2` | `0.9332` |
| `frameworks-libraries` | `vllm` | `vllm-1.0` | `0.9330` |

### Top 20 Lowest Similarity Pairs

| Category | Base Term | Compound Term | Cosine Similarity |
| :--- | :--- | :--- | :---: |
| `ai-models` | `opt` | `opt-350m` | `0.7405` |
| `ai-models` | `phi` | `phi-4-mini` | `0.7380` |
| `ai-models` | `command` | `command-r-plus` | `0.7379` |
| `frameworks-libraries` | `bq` | `bq-2025` | `0.7355` |
| `frameworks-libraries` | `java` | `java-25` | `0.7344` |
| `ai-models` | `phi` | `phi-5-mini` | `0.7273` |
| `frameworks-libraries` | `torch` | `torchaudio-2.5` | `0.7253` |
| `frameworks-libraries` | `go` | `go-1.23` | `0.7224` |
| `ai-models` | `stable` | `stable-diffusion-4` | `0.7206` |
| `frameworks-libraries` | `torch` | `torchaudio-2.6` | `0.7159` |
| `frameworks-libraries` | `next` | `next-16-canary` | `0.7149` |
| `ai-models` | `hmm` | `hmm-2025` | `0.7143` |
| `frameworks-libraries` | `go` | `go-1.24` | `0.7132` |
| `ai-models` | `stable` | `stable-diffusion-3.5` | `0.7106` |
| `ai-models` | `yolo` | `yolov8` | `0.7085` |
| `ai-models` | `yolo` | `yolov11` | `0.7069` |
| `ai-models` | `phi` | `phi-4-14b` | `0.7059` |
| `frameworks-libraries` | `spring` | `spring-framework-7` | `0.6814` |
| `frameworks-libraries` | `spring` | `spring-boot-3.3` | `0.6345` |
| `frameworks-libraries` | `spring` | `spring-boot-4.0` | `0.6099` |
