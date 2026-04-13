# Attention-Summarizer-HPC

**A complete High Performance Computing project: Self-Attention based Text Summarizer in C**

This repository implements a **manual self-attention mechanism** for extractive text summarization using real pretrained GloVe word embeddings.  
Everything starts with a clean **serial C** baseline (no external ML libraries), then progressively adds parallel versions using MPI, OpenMP, and CUDA.

The core goal is to compute self-attention on tokenized text and generate a **meaningful extractive summary** by ranking sentences according to their average attention strength.

### Project Features
- Pure serial C implementation with explicit loops (easy to understand and parallelize)
- Real GloVe 50-dimensional pretrained embeddings (loaded from file)
- Manual tokenization, vectorization, Q/K/V formation, scaled dot-product attention, and softmax
- Attention-driven extractive summarization (sentences selected using calculated attention weights)
- Multiple parallel backends in the same repo:
  - **Serial** (baseline)
  - **MPI** (distributed memory)
  - **OpenMP** (shared memory multi-core)
  - **CUDA** (GPU acceleration)
- Clear, well-commented code suitable for HPC teaching and performance comparison
- Meaningful output: tokens, sample attention matrix, and real attention-based summary

### Overall Architecture

## Run The Serial Version

The serial baseline source is in `serial/attention_summarizer.c`.

### 1) Go to the project root

From PowerShell:

```powershell
cd "d:\7 SEM\HPC\Project\Attention-Summarizer-HPC"
```

From WSL/Linux:

```bash
cd /mnt/d/7\ SEM/HPC/Project/Attention-Summarizer-HPC
```

### 2) Compile the serial program

```bash
gcc serial/attention_summarizer.c -o serial/summarizer -lm
```

If your compiler needs C99 explicitly, use:

```bash
gcc -std=c99 serial/attention_summarizer.c -o serial/summarizer -lm
```

### 3) Run the executable

```bash
./serial/summarizer
```

On Windows PowerShell (native), run:

```powershell
.\serial\summarizer.exe
```

### 4) Verify inputs are present

Before running, keep these files in the repository root:

- `glove.6B.50d.txt` (or your selected embedding file)
- `examples/paragraphs.txt` (you can copy the example paragraphs from it)

### 5) Expected output

The program prints:

- tokenized text information
- attention values (sample matrix/weights)
- extracted summary sentences based on attention score

