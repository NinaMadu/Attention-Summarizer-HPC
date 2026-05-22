# Compiler variables
CC = gcc
CFLAGS = -O2 -std=c99

# CUDA Compiler variables
NVCC = nvcc
# NVIDIA RTX 5090 uses sm_120, but if nvcc doesn't support it, leave it to default or adjust it.
NVCC_FLAGS = -O2

all: serial_app cuda_app

serial_app: serial/summarizer
cuda_app: cuda/summarizer_cuda

serial/summarizer: serial/attention_summarizer_new.c
	$(CC) $(CFLAGS) $< -o $@ -lm

cuda/summarizer_cuda: cuda/attention_summarizer_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lm

compare: serial/summarizer cuda/summarizer_cuda
	@echo "=== SERIAL ===" && ./serial/summarizer
	@echo "=== CUDA ===" && ./cuda/summarizer_cuda

clean:
	rm -f serial/summarizer cuda/summarizer_cuda serial/*.exe cuda/*.exe

.PHONY: all serial_app cuda_app compare clean
CC          ?= gcc
MPICC       ?= mpicc
MPIRUN      ?= mpirun
CFLAGS      ?= -std=gnu11 -O2
LDFLAGS     ?= -lm
MPI_RANKS   ?= 2
OMP_THREADS ?= 4
SAMPLE_TEXT ?= High performance computing speeds up scientific workloads. MPI distributes work across processes. OpenMP uses threads inside each process. Hybrid programming combines both models.

SERIAL_BIN  = serial/summarizer_serial_mode
SERIAL_200D_BIN = serial/summarizer_200d
SERIAL_300D_BIN = serial/summarizer_300d
OPENMP_BIN  = openmp/summarizer_openmp_mode
MPI_BIN     = mpi/summarizer_mpi_mode
HYBRID_BIN  = mpi/mpi_openmp_hybrid

.PHONY: all serial-mode serial-200d serial-300d openmp-mode mpi-mode mpi_openmp_hybrid
.PHONY: serial openmp mpi hybrid
.PHONY: run-serial-200d run-serial-300d run-openmp-mode run-mpi-mode run-mpi_openmp_hybrid
.PHONY: check-serial-200d check-serial-300d check-openmp-mode check-mpi-mode check-mpi_openmp_hybrid clean

all: serial-mode serial-200d serial-300d openmp-mode mpi-mode mpi_openmp_hybrid

serial-mode:
	$(CC) $(CFLAGS) serial/attention_summarizer_new.c -o $(SERIAL_BIN) $(LDFLAGS)

serial-200d:
	$(CC) $(CFLAGS) serial/attention_summarizer_200d.c -o $(SERIAL_200D_BIN) $(LDFLAGS)

serial-300d:
	$(CC) $(CFLAGS) serial/attention_summarizer_300d.c -o $(SERIAL_300D_BIN) $(LDFLAGS)

openmp-mode:
	$(CC) $(CFLAGS) -fopenmp openmp/attention_summarizer_new_omp.c -o $(OPENMP_BIN) $(LDFLAGS)

mpi-mode:
	$(MPICC) $(CFLAGS) mpi/attention_summarizer_mpi_new.c -o $(MPI_BIN) $(LDFLAGS)

mpi_openmp_hybrid:
	$(MPICC) $(CFLAGS) -fopenmp mpi/attention_summarizer_hybrid.c -o $(HYBRID_BIN) $(LDFLAGS)

serial: serial-mode
openmp: openmp-mode
mpi: mpi-mode
hybrid: mpi_openmp_hybrid

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

check-openmp-mode: openmp-mode
	printf '%s\n' "$(SAMPLE_TEXT)" | OMP_NUM_THREADS=$(OMP_THREADS) ./$(OPENMP_BIN)

check-serial-200d: serial-200d
	printf '%s\n' "$(SAMPLE_TEXT)" | ./$(SERIAL_200D_BIN)

check-serial-300d: serial-300d
	printf '%s\n' "$(SAMPLE_TEXT)" | ./$(SERIAL_300D_BIN)

check-mpi-mode: mpi-mode
	printf '%s\n' "$(SAMPLE_TEXT)" | $(MPIRUN) -np $(MPI_RANKS) ./$(MPI_BIN)

check-mpi_openmp_hybrid: mpi_openmp_hybrid
	printf '%s\n' "$(SAMPLE_TEXT)" | OMP_NUM_THREADS=$(OMP_THREADS) $(MPIRUN) -np $(MPI_RANKS) ./$(HYBRID_BIN)

clean:
	$(RM) $(SERIAL_BIN) $(SERIAL_200D_BIN) $(SERIAL_300D_BIN) $(OPENMP_BIN) $(MPI_BIN) $(HYBRID_BIN)
