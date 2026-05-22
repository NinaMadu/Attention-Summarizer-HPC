# OpenMP Notes

## OpenMP Version

The OpenMP implementation is provided in `openmp/attention_summarizer_new_omp.c`.

This version was created to preserve the same structure and core logic as the serial baseline in `serial/attention_summarizer_new.c`, so that results and timing can be compared directly.

## Compilation

Compile the OpenMP version with:

```bash
gcc -std=gnu11 openmp/attention_summarizer_new_omp.c -o openmp/summarizer_new_omp -fopenmp -lm
```

Compile the serial baseline with:

```bash
gcc -std=gnu11 serial/attention_summarizer_new.c -o serial/summarizer_new -lm
```

## Execution

Run the OpenMP version with:

```bash
./openmp/summarizer_new_omp
```

Run the serial version with:

```bash
./serial/summarizer_new
```

To test different OpenMP thread counts:

```bash
export OMP_NUM_THREADS=1
./openmp/summarizer_new_omp

export OMP_NUM_THREADS=2
./openmp/summarizer_new_omp

export OMP_NUM_THREADS=4
./openmp/summarizer_new_omp
```

## Parallelized Sections

OpenMP was added only to loop-based compute-heavy sections where iterations are largely independent.

The following parts were parallelized:

- mean vector computation during PCA preprocessing
- covariance matrix construction using thread-local accumulation and merging
- whitening matrix construction (`W_Q` and `W_K`)
- embedding projection
- token vectorization across input tokens
- positional encoding
- multi-head attention score computation
- softmax normalization across attention rows
- attention output accumulation
- final attention-weight averaging across heads
- token-level attention centrality scoring

## Sections Kept Serial

Some parts were intentionally left serial to preserve correctness and keep the structure close to the serial baseline:

- GloVe file loading
- Jacobi eigendecomposition
- sentence-level accumulation and final sentence selection
- output formatting and reporting

## Correctness Notes

- The OpenMP version is expected to produce the same summary sentences as the serial version for the same input paragraph.
- Small floating-point differences may appear because parallel reductions can change the order of accumulation.
- Attention row sums should remain close to `1.0`, confirming softmax normalization is still correct.

## Performance Notes

- The covariance computation benefits noticeably from OpenMP parallelization.
- Total speedup may still be limited because `load_glove` remains one of the largest runtime costs.
- For short paragraphs, some small stages such as self-attention may show limited speedup or even slight slowdown because thread-management overhead can exceed the amount of work.

## Summary

The OpenMP version keeps the same algorithm, output style, and comparison-friendly structure as the serial implementation while parallelizing the most expensive independent loops. This makes it suitable for HPC performance analysis against the serial baseline.
