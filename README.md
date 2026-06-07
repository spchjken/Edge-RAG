# Edge-RAG: Extractive-Compression RAG for Resource-Constrained Edge Devices

This repository contains the experimental implementation and evaluation infrastructure for **Edge-RAG**, a local-first, Extractive-Compression (EC) RAG framework designed for modern consumer-grade edge hardware (tested on an RTX 5060 Ti 16GB VRAM profile).

By leveraging query-guided lexical anchors, 1D interval merging, and dual-bypass density routing, Edge-RAG reduces context token footprints and improves Time to First Token (TTFT) while maintaining high factual retrieval recall.

---

## 🚀 Key Features
- **Extractive-Compression Pipeline**: Aho-Corasick lexical search, interval merging, and dual-bypass density routing.
- **2026 Model Support**: Integrated with Google's Gemma-4-E2B/E4B, Alibaba's Qwen3.5-2B/4B, and Zyphra's ZAYA1-8B (sparse MoE).
- **Multi-Backend Support**: Runs dense models via **Ollama** and sparse MoE models via custom **llama.cpp** servers (with logic to bypass forced reasoning).
- **Hardware Simulation**: Built-in VRAM constraint simulation (e.g., targeting 8GB and 16GB limits) using PyTorch memory hooks.

---

## 📁 Repository Structure

```
Edge-RAG/
├── docs/                             # Metric definitions and design documents
├── configs/                          # Hyperparameters and hardware profiles
├── src/
│   ├── pipeline/                     # Core EC-RAG algorithm (Aho-Corasick, Router, etc.)
│   ├── baselines/                    # BM25, Dense RAG, and LLMLingua-2
│   ├── evaluation/                   # Benchmarking and VRAM simulation
│   └── utils/                        # OpenAI-compatible API client & logs
├── scripts/                          # Executable benchmarks, ablations, and setup scripts
├── tests/                            # Unit and integration test suite
├── data/                             # Raw and processed datasets
└── results/                          # CSV output logs and performance tables
```

---

## 🛠️ Local Setup

### 1. Prerequisites
- **GPU**: NVIDIA GPU with >= 16GB VRAM (e.g., RTX 5060 Ti).
- **Ollama**: Installed and running on your system.
- **Python**: Version 3.11 or higher.

### 2. Environment Setup
Create a Python virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Model Backends Setup
- **Ollama Models**: Pull the required Qwen3.5 and Gemma-4 models:
  ```bash
  ollama pull qwen3.5:2b
  ollama pull qwen3.5:4b
  ollama pull gemma4:e2b
  ollama pull gemma4:e4b
  ```
- **ZAYA1-8B MoE Model**: Compile llama.cpp from PR #23112 and download the GGUF weights:
  ```bash
  chmod +x scripts/setup_zaya.sh
  ./scripts/setup_zaya.sh
  ```
  *Note: To use the ZAYA1-8B model during benchmarking, you must first start the compiled `llama-server` in a separate terminal:*
  ```bash
  ./vendor/llama.cpp/build/bin/llama-server -m data/models/ZAYA1-8B-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
  ```

### 4. Datasets Acquisition
Run the script to download MS MARCO and prepare document directories:
```bash
python scripts/download_datasets.py
```
*Note: For the medical dataset, manually download the IFRC First Aid 2020 PDF and place it under `data/raw/medical/`.*

---

## 📊 Running Benchmarks

To execute the full evaluation suite and reproduce the results in Table 1 and Table 2:
```bash
python scripts/run_benchmarks.py
```

To run the ablation studies:
```bash
python scripts/run_ablations.py
```
