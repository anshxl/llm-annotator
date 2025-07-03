# LLM Annotation Pipeline

This repository provides a zero-cost, local annotation pipeline using a quantized LLaMA model via `llama.cpp` and the `llama-cpp-python` bindings. It enables batched, cached inference on large text datasets (e.g., 100K rows) entirely on CPU.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Environment Setup](#python-environment-setup)
3. [System Dependencies](#system-dependencies)
4. [Building `llama.cpp`](#building-llama-cpp)
5. [Downloading a Quantized Model](#downloading-a-quantized-model)
6. [Running the Annotation Script](#running-the-annotation-script)

---

## Prerequisites

* **Python 3.8+**
* At least **4 CPU threads** recommended
* \~8 GB free disk space for the model (.gguf \~4 GB)

## Python Environment Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install Python dependencies
pip install -r requirements.txt
```

**`requirements.txt`** should include:

```
llama-cpp-python
pandas
tqdm
langchain-core
langchain-community
```

*(Note: `sqlite3` is part of the Python stdlib.)*

## System Dependencies

Install CMake and a C/C++ compiler (these commands can be run from any directory):

* **macOS (Homebrew):**

  ```bash
  brew install cmake gcc
  ```

* **Ubuntu/Debian:**

  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake
  ```

* **Fedora/RHEL:**

  ```bash
  sudo dnf install -y gcc gcc-c++ make cmake
  ```

## Building `llama.cpp`

```bash
# 1. Clone the repo
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. Create build directory and compile
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. Verify binaries
./build/bin/quantize --help
```

Then return to your project root:

```bash
cd ../../
```

## Downloading a Quantized Model

We recommend Meta’s LLaMA 2‑7B in GGUF Q4\_K\_M format (4 GB). For even smaller model options, we recommend TinyLlama 1.1B with GGUF Q4\_K\_M format (700 MB). Example using the Hugging Face CLI:

```bash
# Install and login
pip install huggingface_hub
huggingface-cli login

# Download chat-optimized model
python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
  repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
  filename="llama-2-7b-chat.Q4_K_M.gguf",
  cache_dir="models/"
)
EOF
```

## Running the Annotation Script

Point your Python script at the downloaded model:

```python
from llama_cpp import Llama
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_threads=8
)
```

Then execute your batched annotation pipeline as usual:

```bash
python annotate.py --input data/gold_standard.csv \
                   --output data/annotated_gold_standard.csv
```
