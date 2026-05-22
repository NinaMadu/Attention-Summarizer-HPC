# Compiler variables
CC          ?= gcc
MPICC       ?= mpicc
MPIRUN      ?= mpirun
NVCC        ?= nvcc

CFLAGS      ?= -std=gnu11 -O2
NVCC_FLAGS  ?= -O2
LDFLAGS     ?= -lm

MPI_RANKS   ?= 2
OMP_THREADS ?= 4
SAMPLE_TEXT ?= High performance computing speeds up scientific workloads. MPI distributes work across processes. OpenMP uses threads inside each process. Hybrid programming combines both models.

# Source files currently kept in the project
SERIAL_200D_SRC = serial/attention_summarizer_200d.c
SERIAL_300D_SRC = serial/attention_summarizer_300d.c
OPENMP_SRC      = openmp/attention_summarizer_omp.c
MPI_SRC         = mpi/attention_summarizer_mpi.c
HYBRID_SRC      = mpi/attention_summarizer_hybrid.c
CUDA_SRC        = cuda/attention_summarizer_cuda.cu

# Output binaries
SERIAL_200D_BIN = serial/summarizer_200d
SERIAL_300D_BIN = serial/summarizer_300d
OPENMP_BIN      = openmp/summarizer_openmp_mode
MPI_BIN         = mpi/summarizer_mpi_mode
HYBRID_BIN      = mpi/mpi_openmp_hybrid
CUDA_BIN        = cuda/summarizer_cuda

.PHONY: all cpu serial_app cuda_app
.PHONY: serial-mode serial-200d serial-300d openmp-mode mpi-mode mpi_openmp_hybrid
.PHONY: serial openmp mpi hybrid cuda
.PHONY: run-serial-mode run-serial-200d run-serial-300d run-openmp-mode run-mpi-mode run-mpi_openmp_hybrid run-cuda
.PHONY: check-serial-mode check-serial-200d check-serial-300d check-openmp-mode check-mpi-mode check-mpi_openmp_hybrid check-cuda
.PHONY: compare compare-cpu compare-cuda clean

all: cpu

cpu: serial-200d serial-300d openmp-mode mpi-mode mpi_openmp_hybrid

# Backward-compatible aliases. The current default serial build is the 300D version.
serial_app: serial-300d

serial-mode: serial-300d

serial: serial-300d

openmp: openmp-mode

mpi: mpi-mode

hybrid: mpi_openmp_hybrid

cuda: cuda_app

cuda_app: $(CUDA_BIN)

serial-200d: $(SERIAL_200D_BIN)

serial-300d: $(SERIAL_300D_BIN)

openmp-mode: $(OPENMP_BIN)

mpi-mode: $(MPI_BIN)

mpi_openmp_hybrid: $(HYBRID_BIN)

$(SERIAL_200D_BIN): $(SERIAL_200D_SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(SERIAL_300D_BIN): $(SERIAL_300D_SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(OPENMP_BIN): $(OPENMP_SRC)
	$(CC) $(CFLAGS) -fopenmp $< -o $@ $(LDFLAGS)

$(MPI_BIN): $(MPI_SRC)
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(HYBRID_BIN): $(HYBRID_SRC)
	$(MPICC) $(CFLAGS) -fopenmp $< -o $@ $(LDFLAGS)

$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

run-serial-mode: run-serial-300d

run-serial-200d: serial-200d
	./$(SERIAL_200D_BIN)

run-serial-300d: serial-300d
	./$(SERIAL_300D_BIN)

run-openmp-mode: openmp-mode
	OMP_NUM_THREADS=$(OMP_THREADS) ./$(OPENMP_BIN)

run-mpi-mode: mpi-mode
	$(MPIRUN) -np $(MPI_RANKS) ./$(MPI_BIN)

run-mpi_openmp_hybrid: mpi_openmp_hybrid
	OMP_NUM_THREADS=$(OMP_THREADS) $(MPIRUN) -np $(MPI_RANKS) ./$(HYBRID_BIN)

run-cuda: cuda_app
	./$(CUDA_BIN)

check-serial-mode: check-serial-300d

check-serial-200d: serial-200d
	printf '%s\n' "$(SAMPLE_TEXT)" | ./$(SERIAL_200D_BIN)

check-serial-300d: serial-300d
	printf '%s\n' "$(SAMPLE_TEXT)" | ./$(SERIAL_300D_BIN)

check-openmp-mode: openmp-mode
	printf '%s\n' "$(SAMPLE_TEXT)" | OMP_NUM_THREADS=$(OMP_THREADS) ./$(OPENMP_BIN)

check-mpi-mode: mpi-mode
	printf '%s\n' "$(SAMPLE_TEXT)" | $(MPIRUN) -np $(MPI_RANKS) ./$(MPI_BIN)

check-mpi_openmp_hybrid: mpi_openmp_hybrid
	printf '%s\n' "$(SAMPLE_TEXT)" | OMP_NUM_THREADS=$(OMP_THREADS) $(MPIRUN) -np $(MPI_RANKS) ./$(HYBRID_BIN)

check-cuda: cuda_app
	printf '%s\n' "$(SAMPLE_TEXT)" | ./$(CUDA_BIN)

compare: compare-cuda

compare-cpu: check-serial-300d check-openmp-mode check-mpi-mode check-mpi_openmp_hybrid

compare-cuda: check-serial-300d check-cuda

clean:
	$(RM) $(SERIAL_200D_BIN) $(SERIAL_300D_BIN) $(OPENMP_BIN) $(MPI_BIN) $(HYBRID_BIN) $(CUDA_BIN)
	$(RM) serial/summarizer serial/summarizer_new serial/summarizer_serial_mode
	$(RM) openmp/summarizer_new_omp openmp/summarizer_openmp
	$(RM) mpi/summarizer_mpi mpi/summarizer_mpi_new
	$(RM) serial/*.exe openmp/*.exe mpi/*.exe cuda/*.exe
