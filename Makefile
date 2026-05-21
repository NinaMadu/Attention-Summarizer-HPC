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
