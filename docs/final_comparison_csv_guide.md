# Final Comparison CSV Guide

The comparison CSVs are now generated automatically by:

```bash
python3 scripts/run_comparison_experiment.py --input-file examples/paragraphs.txt
```

The script runs each implementation with the same input paragraph, saves raw logs, and fills:

```text
docs/final_performance_comparison.csv
docs/final_accuracy_comparison.csv
```

Raw logs are saved under:

```text
results/comparison_runs/<experiment_id>/
```

## Performance CSV

File:

```text
docs/final_performance_comparison.csv
```

This file is for speed and resource comparison.

Important columns:

```text
mode
total_s
load_s
compute_total_s
covariance_s
jacobi_s
whitening_s
vectorize_s
positional_s
attention_s
scoring_s
cpu_util_percent
peak_memory_mb
speedup_vs_best_serial300
compute_speedup_vs_best_serial300
efficiency_percent
compute_efficiency_percent
```

Use this table to answer:

```text
Which implementation is fastest?
Which stage takes the most time?
How much speedup did OpenMP, MPI, and hybrid achieve?
How much memory was used?
```

For the main HPC performance comparison, use:

```text
compute_total_s = total_s - load_s
compute_speedup_vs_best_serial300
compute_efficiency_percent
```

`load_s` is kept separately because GloVe loading is mostly a fixed file I/O startup cost. `compute_total_s` is better for comparing the actual computation after the embeddings are loaded.

## Accuracy CSV

File:

```text
docs/final_accuracy_comparison.csv
```

This file is for output consistency comparison.

In this project, "accuracy" means:

```text
Do different implementations select the same summary and produce close numerical output?
```

Important columns:

```text
mode
selected_sentences
same_selected_as_serial300
config_same_as_serial300
sentence_counts_same_as_serial300
token_sequence_same_as_serial300
attention_checksum
attention_checksum_diff
output_checksum
output_checksum_diff
sentence_score_rmse
token_centrality_rmse
output_l2_rmse
attention_sample_rmse
attention_rows_valid
accuracy_status
```

The baseline is:

```text
serial_300d
```

For OpenMP, MPI, and hybrid, the expected result is usually:

```text
same selected sentences
same configuration
very small RMSE / MAE / max-error values
attention rows valid
```

The easiest accuracy columns to explain in the report are:

```text
sentence_score_rmse
token_centrality_rmse
output_l2_rmse
attention_sample_rmse
```

These are calculated against the `serial_300d` baseline using the `COMPARE_SENTENCE_TABLE`, `COMPARE_TOKEN_TABLE`, and `COMPARE_ATTENTION_SAMPLE` values printed by each program. Lower is better. A value close to zero means the implementation produced almost the same numerical output as serial.

## Accuracy Status Meaning

The script fills `accuracy_status` as:

```text
baseline
match
same_summary_numeric_diff
numeric_close_match
different_output
different_config
different_tokenization
```

Meaning:

```text
baseline                  serial_300d reference row
match                     same selected sentences and near-zero RMSE values
numeric_close_match       same selected sentences and acceptably small RMSE values
same_summary_numeric_diff same selected sentences, but small numeric differences
different_output          selected sentences differ
different_config          configuration differs, for example a legacy 200D run vs serial 300D
different_tokenization    token or sentence alignment differs, so MSE is not directly comparable
```

## CUDA Note

The CUDA code now uses the same main embedding configuration as the 300D CPU versions:

```text
glove.6B.300d.txt
EMBED_DIM = 300
```

This means CUDA can be compared directly against:

```text
serial_300d
openmp
mpi
hybrid
cuda_openmp_hybrid
```

For CUDA, a fair result should usually show:

```text
config_same_as_serial300 = yes
accuracy_status = match
```

Small numerical differences are still possible because GPU floating-point operations may happen in a different order.

The CUDA + OpenMP hybrid version keeps the heavy covariance and attention kernels on the GPU, then uses OpenMP threads for CPU-side preparation and scoring work such as vector lookup, positional encoding, whitening matrix construction, and sentence scoring.

To include the old 200D serial baseline as an extra legacy comparison, run:

```bash
python3 scripts/run_comparison_experiment.py --include-serial-200d
```

## Common Commands

Run all default modes:

```bash
python3 scripts/run_comparison_experiment.py --input-file examples/paragraphs.txt
```

Skip CUDA:

```bash
python3 scripts/run_comparison_experiment.py --skip-cuda
```

Use 4 MPI ranks and 8 OpenMP threads:

```bash
python3 scripts/run_comparison_experiment.py --mpi-ranks 4 --omp-threads 8
```

Use a paragraph directly:

```bash
python3 scripts/run_comparison_experiment.py --input-text "HPC improves performance. MPI distributes work. OpenMP uses threads."
```
