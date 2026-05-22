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
- Deterministic output across CPU and GPU for numerical correctness verification

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
- complexity and timing report

## Run The CUDA Version

The CUDA accelerated version is in `cuda/attention_summarizer_cuda.cu`. It requires an NVIDIA GPU (the Makefile defaults to targeting the RTX 5090 using `-arch=sm_120`, adjust if using a different card).

### 1) Compile the CUDA program

```bash
make cuda_app
```
*(Or use `nvcc -O2 -arch=sm_120 cuda/attention_summarizer_cuda.cu -o cuda/summarizer_cuda -lm`)*

### 2) Run the executable

```bash
./cuda/summarizer_cuda
```
On Windows PowerShell:
```powershell
.\cuda\summarizer_cuda.exe
```

## Compare Execution Time

You can build and run both versions consecutively using Make:

```bash
make compare
```

This will run the serial version followed immediately by the CUDA version, allowing you to easily compare their runtime metrics on the generated report.

dataset: https://www.kaggle.com/datasets/incorpes/glove6b200d



# results

## Serial
  TOP 10 IMPORTANT WORDS  (ranked by attention output norm)
  Rank  Word                  Score       Relevance
------------------------------------------------------------
  #1    com                      12.8472  [********************]
  #2    temperatures             12.6819  [******************* ]
  #3    systems                  12.4401  [******************* ]
  #4    universities             12.3294  [******************* ]
  #5    intelligence             12.3060  [******************* ]
  #6    states                   12.1738  [******************  ]
  #7    network                  12.1395  [******************  ]
  #8    research                 12.1290  [******************  ]
  #9    mobile                   12.1018  [******************  ]
  #10   quantum                  12.0646  [******************  ]


  COMPLEXITY & TIMING REPORT
  Stage                               Time (s)          Operations  % Total
----------------------------------------------------------------------
  load_glove                            4.5549            80000000   15.6% [###                 ]
  covariance_matrix                    12.0038          8040000000   41.2% [########            ]
  jacobi_eigen                          0.2424           210207680    0.8% [                    ]
  build_whitening_W                     0.0002               40000    0.0% [                    ]
  tokenize+vectorize                    0.0165             2230681    0.1% [                    ]
  positional_encoding                   0.0011              102400    0.0% [                    ]
  self_attention                        0.3098           145817600    1.1% [                    ]
  sentence_scoring                      0.0012               15360    0.0% [                    ]
----------------------------------------------------------------------
  TOTAL                                29.1571

## parallel


  TOP 10 IMPORTANT WORDS  (ranked by attention output norm)
  Rank  Word                  Score       Relevance
------------------------------------------------------------
  #1    com                      12.8309  [********************]
  #2    temperatures             12.6816  [******************* ]
  #3    systems                  12.4406  [******************* ]
  #4    universities             12.3286  [******************* ]
  #5    intelligence             12.3061  [******************* ]
  #6    states                   12.1731  [******************  ]
  #7    network                  12.1393  [******************  ]
  #8    research                 12.1289  [******************  ]
  #9    mobile                   12.1053  [******************  ]
  #10   quantum                  12.0672  [******************  ]

  COMPLEXITY & TIMING REPORT  [CUDA]
  Stage                               Time (s)          Operations  % Total
----------------------------------------------------------------------
  load_glove                            3.7923            80000000   37.0% [#######             ]
  covariance_matrix                     0.4431          8040000000    4.3% [                    ]
  jacobi_eigen                          0.0543           221165416    0.5% [                    ]
  build_whitening_W                     0.0001               40000    0.0% [                    ]
  tokenize+vectorize                    0.0128             2230681    0.1% [                    ]
  positional_encoding                   0.0009              102400    0.0% [                    ]
  self_attention                        0.0038           145817600    0.0% [                    ]
  sentence_scoring                      0.0002              262656    0.0% [                    ]
----------------------------------------------------------------------
  TOTAL                                10.2435
## OpenMP Notes

The OpenMP implementation is documented in [docs/openmp_notes.md](docs/openmp_notes.md).

That file includes:

- the OpenMP source file location
- compile and run commands
- `OMP_NUM_THREADS` usage
- parallelized sections
- correctness and performance notes



## Running in Ninada's Laptop:
su - mpiuser
cd "/mnt/d/7 SEM/HPC/Project/Attention-Summarizer-HPC"

# Run MPI
Using make file:

make mpi-mode
mpirun -np 2 ./mpi/summarizer_mpi_mode

Mannualy:

mpicc -std=gnu11 -O2 mpi/attention_summarizer_mpi_new.c -o mpi/summarizer_mpi_mode -lm
mpirun -np 2 ./mpi/summarizer_mpi_mode

# Run MPI/OpenMP Hybrid
Using make file:

make mpi_openmp_hybrid
OMP_NUM_THREADS=4 mpirun -np 2 ./mpi/mpi_openmp_hybrid

Mannualy:

make run-mpi_openmp_hybrid MPI_RANKS=2 OMP_THREADS=4

# Run OpenMP
Using make file:

make openmp-mode
make check-openmp-mode OMP_THREADS=4

Mannualy:

OMP_NUM_THREADS=4 ./openmp/summarizer_openmp_mode





